# From Quadratic Bottleneck to Linear Generation: Implementing Correct Recurrent Inference for xLSTM-Based Symbolic Music Generation

---

## 1. Introduction: The Problem We Faced

During the generation phase of our xLSTM-based symbolic music generation system — trained on the Lakh MIDI Dataset (LMD) with the REMIGEN2 token encoding — we encountered severe practical obstacles. Generating music using the Helibrunna framework's built-in `LanguageModel.generate()` method was **prohibitively slow for long sequences** and **crashed with Out-Of-Memory (OOM) errors** when attempting to generate full-length musical pieces in a single pass.

For example, generating a piece of 2,000 tokens took nearly a minute, and generating the longer pieces required for our evaluation (8,000–12,000 tokens) was projected to take 14–32 minutes per piece. At the upper end, attempting to generate very long sequences in a single call caused CUDA Out-Of-Memory errors because the framework was accumulating the entire sequence as a growing tensor in GPU memory, and reprocessing it from scratch on every step.

These were not limitations of the xLSTM model itself — the model architecture is specifically designed for efficient, linear-time generation of arbitrarily long sequences. The problem lay entirely in how the Helibrunna inference code was calling the model. This document traces our investigation of the root cause, our implementation of a correct solution, and the resulting performance improvements.

---

## 2. Background: How xLSTM Is Designed to Work

### 2.1 The Dual Formulation of xLSTM

The xLSTM architecture (Beck et al., 2024) is a modern recurrent neural network that introduces two new memory cell variants — sLSTM (scalar memory with memory mixing) and mLSTM (matrix memory with a covariance update rule). A critical design feature of xLSTM, and one central to this investigation, is that the architecture supports **two mathematically equivalent formulations**:

**The Parallel Formulation** is used during training. It processes all T tokens of a sequence simultaneously, computing attention-like operations across the full sequence in parallel. This is computationally efficient for training on GPUs because it leverages matrix operations and avoids sequential dependencies. The parallel formulation for mLSTM constructs a gate activation matrix D ∈ R^(T×T) and computes all hidden states in one pass. This is analogous to how Transformers process an entire sequence at once during training.

**The Recurrent Formulation** is used during inference (text/music generation). It processes tokens one at a time, maintaining a compact "recurrent state" (consisting of cell states c_t, normalizer states n_t, and hidden states h_t for sLSTM; matrix memory C_t, normalizer n_t, and hidden state h_t for mLSTM). At each time step, the model takes the previous token and the current state, produces a prediction for the next token, and returns an updated state. The xLSTM paper explicitly states: *"After training we can still use the recurrent formulation for fast text generation."*

The key insight is that both formulations produce **identical outputs** for the same input — they are mathematically equivalent. The parallel version is faster for training (processes all tokens at once), while the recurrent version is faster for generation (O(1) per token, since it only processes one new token at each step rather than the full sequence).

### 2.2 The Recurrent State as Compressed Memory

In the recurrent formulation, the state acts as a **compressed summary of the entire sequence history**. The sLSTM cell state update rule:

```
c_t = f_t * c_{t-1} + i_t * z_t
```

shows that at every time step, the model selectively retains information from the past (controlled by the forget gate f_t) and incorporates new information (controlled by the input gate i_t). After processing 5,000 tokens, the recurrent state encodes a compressed representation of all 5,000 tokens. This is what allows xLSTM to extrapolate beyond its training context length — the state just keeps accumulating information indefinitely.

The xLSTM paper demonstrates this extrapolation capability experimentally: models trained on context length 2,048 maintain low perplexity when tested at context lengths up to 16,384, significantly outperforming Transformers and other methods at sequence length extrapolation.

### 2.3 Implications for Generation Speed and Memory

The recurrent formulation has fundamental computational advantages for autoregressive generation:

- **Recurrent generation**: Each new token requires one forward pass through the model with a single token input and the current state. The cost per token is **O(1)** — constant regardless of how many tokens have been generated so far. Total cost for N tokens: **O(N)**. GPU memory usage is also **constant** — only the fixed-size recurrent state and a single token are held on the GPU.

- **Parallel-mode generation** (reprocessing the full sequence): Each new token requires a forward pass through the model with the *entire sequence generated so far*. The cost of generating the k-th token is **O(k)**. Total cost for N tokens: **O(N²)**. GPU memory usage **grows linearly** with sequence length, since the full sequence tensor must be held in GPU memory and processed on every step.

This quadratic-vs-linear distinction becomes dramatic for long sequences. Generating 12,000 tokens of music (a full piece) would take roughly 25× longer with the parallel approach than with the recurrent approach. Additionally, the growing memory footprint of the parallel approach can exceed GPU capacity for long sequences, causing OOM crashes — which is exactly what we observed.

---

## 3. The Helibrunna Inference Implementation: Where Things Went Wrong

### 3.1 What Helibrunna Does

Our xLSTM model was trained using the Helibrunna framework, a HuggingFace-compatible xLSTM trainer. Helibrunna provides a `LanguageModel` class with a `generate()` method for autoregressive generation. Here is the core of that generation loop (from `languagemodel.py`):

```python
while inputs.shape[1] < max_length:
    # Feed the ENTIRE sequence through the model
    outputs = self.model(inputs.to(device=self.device))
    
    # Take logits from the last position only
    outputs = outputs / temperature
    outputs = torch.nn.functional.softmax(outputs, dim=-1)
    outputs = torch.multinomial(outputs[0, -1], num_samples=1)
    
    # Append the new token to the full sequence
    inputs = torch.cat([inputs, outputs.unsqueeze(0)], dim=1)
```

The critical line is `outputs = self.model(inputs)`. On every single iteration of the generation loop — for every single new token — the **entire accumulated sequence** is fed back through the model from scratch. This is the parallel formulation being used for generation.

### 3.2 Two Consequences of This Approach

**Quadratic Time Complexity:** Generating the k-th token requires processing a sequence of length k. The total work for N tokens is 1 + 2 + 3 + ... + N = N(N+1)/2, which is O(N²). For a 100-token sequence, this is manageable (5,050 forward steps). For a 12,000-token musical piece, it amounts to 72,006,000 forward steps. In practice, we observed that generating 2,000 tokens took 53 seconds — extrapolating to 12,000 tokens would have required over 30 minutes per piece.

**Growing GPU Memory Usage:** The `inputs` tensor grows by one token on every iteration via `torch.cat`. By the time the model has generated several thousand tokens, this tensor — along with the intermediate activations computed during the forward pass through all layers — can exceed the available GPU memory. We experienced CUDA OOM errors when attempting to generate long single-shot sequences, which forced us to restart kernels and limited the sequence lengths we could evaluate.

### 3.3 The Mismatch

Helibrunna is primarily a **training** framework, and its generation method appears to have been implemented as a simple utility rather than an optimised inference pipeline. The parallel formulation (`model.forward()`) is the natural choice during training, and extending it to generation works correctly for short sequences. However, for the long sequences required in symbolic music generation (thousands to tens of thousands of tokens), this approach hits both time and memory walls.

The xLSTM model itself exposes a `model.step(token, state=state)` method that implements the intended recurrent inference. This method was simply not being used by Helibrunna's generation loop.

---

## 4. The Solution: Recurrent-State Inference (`xLSTMGenerator`)

### 4.1 Design Principles

Once the root cause was identified, the solution was clear: **bypass Helibrunna's generation loop entirely and use the xLSTM model's native recurrent `step()` function**. The xLSTM library (the underlying model code, separate from Helibrunna) exposes a `model.step(token, state=state)` method that implements the recurrent formulation directly. This method:

- Takes a single token and the current recurrent state as input.
- Returns the predicted logits for the next token and the **updated** recurrent state.
- Runs in O(1) time and O(1) memory regardless of how many tokens have been generated previously.

We created a new class, `xLSTMGenerator`, that:

1. Loads the exact same model weights and tokenizer as Helibrunna (no retraining required).
2. Does **not** modify any model architecture code or weights.
3. Implements its own generation loop using `model.step()` instead of `model.forward()`.

### 4.2 Implementation Details

The `xLSTMGenerator` class has three core methods:

#### `__init__`: Model Loading

The constructor loads the model configuration, weights, and tokenizer using the same logic as Helibrunna. It initializes the exact same `xLSTMLMModel` object and loads the same `model.safetensors` checkpoint. The only difference is that it does not use Helibrunna's `LanguageModel` wrapper — it holds a direct reference to the core xLSTM model.

#### `prefill(inputs)`: Building the Initial State

Before generation can begin, the model needs to process the initial prompt to build up its recurrent state. Since the xLSTM implementation does not expose a way to extract state from the parallel `forward()` pass, we use **sequential prefill**: iterating through the prompt tokens one by one using `model.step()`.

```python
def prefill(self, inputs: torch.Tensor):
    state = {}  # Initialize empty recurrent state
    for i in range(inputs.shape[1]):
        token = inputs[:, i:i+1]  # Shape (1, 1)
        logits, state = self.model.step(token, state=state)
    return state, logits
```

This takes O(P) time for a prompt of length P, which is a one-time cost at the start of generation. For our typical 3-token prompts (`s-9 o-0 t-38`), this is negligible.

#### `generate(prompt, max_length, ...)`: The Recurrent Generation Loop

The main generation loop is fundamentally different from Helibrunna's:

```python
# After prefill, we have: state (recurrent state), logits (prediction from last prompt token)

while sequence_length < max_length:
    # Apply temperature scaling and sample
    logits[:, :, special_ids] = float("-inf")
    if ids_to_mask:
        logits[:, :, ids_to_mask] = float("-inf")
    
    scaled_logits = logits / temperature
    probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
    next_token = torch.multinomial(probs[0, -1], num_samples=1)
    
    # The critical difference: step() carries state forward
    logits, state = self.model.step(next_token, state=state)
```

The key line is `logits, state = self.model.step(next_token, state=state)`. Instead of reprocessing the entire sequence, we pass only the newly generated token and the current state. The model updates its state and returns predictions. This is:

- **O(1) per token** — constant time regardless of sequence position.
- **O(N) total** — linear in the number of tokens generated.
- **Constant GPU memory** — the recurrent state is a fixed-size set of tensors, regardless of sequence length. There is no growing sequence tensor that could cause OOM errors.
- **State-preserving** — the recurrent state accumulates information about the entire sequence history, enabling the model to maintain long-range musical coherence.

### 4.3 What We Did NOT Change

It is important to emphasise that this work involved **no changes** to:

- The xLSTM model architecture or mathematical operations.
- The trained model weights (same checkpoint used throughout).
- The tokenizer or vocabulary.
- The REMIGEN2 encoding scheme.
- The training procedure or data.

The only change was in **how we call the model at inference time**. The model itself was always capable of correct recurrent generation — the Helibrunna framework simply was not using that capability.

---

## 5. Validation: Proving Mathematical Equivalence

### 5.1 The Equivalence Test

To confirm that our new `xLSTMGenerator` produces identical outputs to Helibrunna's generator (and is therefore not introducing any bugs or numerical errors), we ran a controlled equivalence test:

1. Load the same model checkpoint in both Helibrunna's `LanguageModel` and our `xLSTMGenerator`.
2. Set `torch.manual_seed(42)` immediately before each generation call to ensure the random number generator is in the same state.
3. Generate 100 tokens from the same prompt (`s-9 o-0 t-38`) at temperature 1.0 with both methods.
4. Compare the output strings character by character.

### 5.2 Seed Alignment

For this test to work, the random state must be perfectly aligned between the two methods. Both methods use `torch.multinomial` for sampling, and this is the only operation that consumes random numbers during generation. As long as:

- The seed is set at the same point relative to the first sampling operation.
- Both methods produce the same logits (which they must, since parallel and recurrent formulations are mathematically equivalent).
- No extraneous torch operations consume random numbers in one path but not the other.

...the sampled token sequences will be identical.

### 5.3 Result

**The test passed: both methods produced exactly the same output text at 100 tokens.** This confirms that `xLSTMGenerator` is a faithful, mathematically equivalent replacement for Helibrunna's generation — it simply uses the recurrent formulation instead of the parallel one.

---

## 6. Benchmarking: Performance Comparison

### 6.1 Test Setup

We benchmarked both generators on the same hardware (single GPU), using the same model checkpoint (`checkpoint-66000`), same prompt (`s-9 o-0 t-38`), and temperature 1.0. We measured wall-clock generation time at four sequence lengths: 100, 500, 1,000, and 2,000 tokens.

Note: We could not benchmark Helibrunna at longer sequence lengths (4,000+) due to the prohibitive time cost and risk of OOM errors. The projected figures below are extrapolated from the observed quadratic trend.

### 6.2 Results

| Sequence Length | Helibrunna (Parallel) | xLSTMGenerator (Recurrent) | Speedup |
|---|---|---|---|
| 100 tokens | 1.80 s | 0.65 s | 2.77× |
| 500 tokens | 6.45 s | 3.22 s | 2.00× |
| 1,000 tokens | 16.74 s | 6.45 s | 2.59× |
| 2,000 tokens | 53.45 s | 12.78 s | 4.18× |

### 6.3 Analysis of Scaling Behaviour

The results confirm the expected computational complexity:

**Helibrunna (O(N²) — Quadratic):** Going from 1,000 to 2,000 tokens (a 2× increase in sequence length), the generation time jumped from 16.74s to 53.45s — a 3.2× increase. This is consistent with quadratic scaling: doubling the sequence length should approximately quadruple the total work (since the cost is proportional to N²), with some fixed overhead that reduces the observed ratio at shorter lengths.

**xLSTMGenerator (O(N) — Linear):** Going from 1,000 to 2,000 tokens, the generation time went from 6.45s to 12.78s — almost exactly a 2× increase. This is textbook linear scaling: double the tokens, double the time.

### 6.4 Projected Performance for Full Music Pieces

Based on the observed scaling behaviour, we can project generation times for the sequence lengths used in our evaluation:

| Target Length | Helibrunna (Projected) | xLSTMGenerator (Projected) | Projected Speedup |
|---|---|---|---|
| 4,000 tokens | ~3.5 minutes | ~25 seconds | ~8× |
| 8,000 tokens | ~14 minutes | ~51 seconds | ~16× |
| 12,000 tokens | ~32 minutes | ~77 seconds | ~25× |

For our full evaluation set of 100 pieces across 5 target lengths (1K, 2K, 4K, 8K, 12K tokens), the total generation time would decrease from **dozens of hours** with Helibrunna to approximately **1–2 hours** with `xLSTMGenerator`. Additionally, the constant memory footprint of the recurrent approach eliminates the OOM errors that previously prevented generation of long pieces entirely.

---

## 7. Implications for Music Generation

### 7.1 Enabling Long-Form Generation

Prior to this work, generating full-length musical pieces (8,000–12,000 tokens) was impractical with our setup: it was either too slow to be feasible for batch evaluation, or it crashed with OOM errors. The `xLSTMGenerator` removes both obstacles. Long pieces can now be generated in roughly a minute each, and GPU memory usage remains constant regardless of sequence length.

### 7.2 Correct Use of Long-Range Memory

Beyond the speed and memory improvements, the recurrent formulation provides a qualitative benefit: **the model now has access to its full long-range memory during generation**. The recurrent state carries a compressed representation of the entire sequence history at every step. This means the model can maintain awareness of musical themes, key signatures, instrument choices, and structural patterns established early in the piece, even when generating tokens near the end of a long composition.

This is the capability that the xLSTM architecture was designed to provide. The paper's demonstration that xLSTM maintains low perplexity when extrapolating from training context length 2,048 to test context length 16,384 (Figure 3, Beck et al., 2024) is predicated on using the recurrent formulation, where the state accumulates indefinitely. Our `xLSTMGenerator` is the first time this capability has been correctly utilised in our music generation pipeline.

### 7.3 Comparison with Other Approaches

It is worth noting how other systems handle long-form generation with sequence models:

**Museformer (Yu et al., 2022)** uses a Transformer with fine- and coarse-grained attention to handle long music sequences. It addresses the quadratic cost of attention by attending at full resolution only to "structure-related bars" (those likely to be repeated) and summarising other bars at lower resolution. This architectural complexity is necessary because Transformers do not have a recurrent state — their "memory" is the explicit attention over all previous tokens, which grows quadratically.

**xLSTM does not need this mechanism.** The recurrent state inherently compresses the entire sequence history into a fixed-size representation. There is no attention matrix that grows with sequence length. The model can process arbitrarily long sequences with constant memory per step, which is a fundamental architectural advantage for long-form music generation.

---

## 8. Summary

The severe performance and memory issues we encountered during xLSTM music generation were caused by a **mismatch between the inference code and the model's intended usage**. The Helibrunna framework used the xLSTM's parallel formulation for generation — reprocessing the entire sequence from scratch for every new token. This resulted in O(N²) time complexity, growing GPU memory consumption, and OOM crashes for long sequences.

The fix was to implement `xLSTMGenerator`, a custom inference wrapper that uses the xLSTM's native recurrent `step()` function as intended by the architecture's designers. This carries a compact recurrent state forward across the entire generation, providing O(1) cost per token, constant memory usage, and correct long-range memory.

The new generator is:

- **Mathematically equivalent** to Helibrunna's output (verified via controlled seed-matched generation producing identical token sequences).
- **Significantly faster**, with O(N) linear scaling versus O(N²) quadratic scaling, yielding a 4.18× speedup at 2,000 tokens and projected 25×+ speedup at 12,000 tokens.
- **Memory-efficient**, with constant GPU memory usage that eliminates the OOM errors encountered with long sequences.
- **Architecturally correct**, using the recurrent formulation that xLSTM was designed for at inference time, providing the model with full long-range memory across the entire generated sequence.

No model weights, architecture code, or training procedures were modified. The improvement comes entirely from using the model correctly at inference time.

---

## References

- Beck, M., Pöppel, K., Spanring, M., Auer, A., Prudnikova, O., Kopp, M., Klambauer, G., Brandstetter, J., & Hochreiter, S. (2024). xLSTM: Extended Long Short-Term Memory. *Thirty-eighth Conference on Neural Information Processing Systems (NeurIPS 2024)*.
- Yu, B., et al. (2022). Museformer: Transformer with Fine- and Coarse-Grained Attention for Music Generation.
- Helibrunna — A HuggingFace compatible xLSTM trainer. Dr. Tristan Behrens.