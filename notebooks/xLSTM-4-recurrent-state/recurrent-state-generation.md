# xLSTM Fast Recurrent Generation Plan

## 1. Why are we doing this?
Currently, our music generation relies on the `LanguageModel.generate()` method provided by the `helibrunna` library. 

**The Problem:**
`helibrunna` implements a naive, **Parallel Formulation** for generation. For every single new coordinate/token it generates, it feeds the *entire sequence history* back into the model from scratch. 
- Generating the 1st token takes $x$ time. 
- Generating the 5,000th token takes $5000 \times x$ time. 
This results in an $O(N^2)$ quadratic slowdown, making it unnecessarily slow to generate long sequences and completely breaking chunkwise generation.

**The Solution:**
The original `xlstm` architecture is specifically designed with a **Recurrent Formulation** (`model.step()`). It can compress the entire sequence history into a small "Recurrent State" (like a KV-cache). 
Instead of recomputing everything, we just pass the last generated token and our current "State" to get the next token. This ensures an $O(N)$ linear generation time: generating the 5,000th token is just as fast as generating the 1st token.

## 2. What are we modifying?
**We are NOT modifying the original xLSTM architecture or the `helibrunna` training code.** 
The model weights and core mathematical operations are completely correct as-is.

Instead, we are creating a brand new, isolated wrapper—`xLSTMGenerator`—that bypasses `helibrunna` entirely during the inference phase. This new wrapper will correctly utilize the original `xlstm` model's `step()` function.

## 3. How will we implement this?
We will create a new file `inference.py` (or similar) inside the `xLSTM-4-recurrent-state` directory. 

### Step 3.1: The Custom Wrapper (`xLSTMGenerator`)
We will write a rigorous Python class that matches the HuggingFace generation style but uses recurrent state logic:

1. **`__init__(self, model_path)`**:
   - Loads the exact same model weights, configuration, and tokenizer you used in `helibrunna`.
   - Initializes the core `xLSTMLMModel` object (which we *have* confirmed naturally exposes the `.step(..., state=state)` method directly).
2. **`prefill(prompt)`**:
   - **Crucial Decision**: The xLSTM implementation currently does not appear to return `state` from `model.forward()`. So, we cannot do "parallel prefill" (processing the prompt all at once to extract a state).
   - Therefore, we will implement **sequential prefill**: we will iterate over the prompt token-by-token using `model.step()` to build the initial recurrent state. It takes $O(P)$ time for a prompt of length $P$, but it is mathematically guaranteed to be correct.
3. **`generate_step(last_token, state)`**:
   - Takes the last generated token and the current state, passes it through `model.step()`, and returns the predicted next token and the updated state.
4. **`generate(prompt, max_length)`**:
   - Orchestrates the loop: calls `prefill(prompt)`, then runs a `while` loop calling `generate_step()` until `max_length` is reached or an End-Of-Sequence token is hit.

### Step 3.2: Validating Correctness
Inside a test script, we will:
1. Generate 100 tokens using the old, slow `helibrunna` method with a fixed random seed.
2. Generate 100 tokens using our new `xLSTMGenerator` method with the exact same seed.
3. Assert that the outputs are **100% identical** at 100 tokens.
   - *Implementation Detail:* To ensure a 100% match, the torch random state must be perfectly aligned. We must ensure `torch.multinomial` is called the exact same number of times and at the exact same sequence step as `helibrunna`. Any extraneous torch random operations will cause RNG paths to diverge.
4. Stretch goal: test at 1000 tokens to check for floating-point drift accumulation (minor divergences are expected over very long sequences due to how fp32 accumulates in parallel vs recurrent modes).
5. Compare the generation speeds (Tokens Per Second). Our new method should be drastically faster.
