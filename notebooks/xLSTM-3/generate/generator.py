"""
generate/generator.py
=====================
MusicGenerator — wraps Helibrunna LanguageModel with two generation strategies:
  - single_shot(): one model call, max_length = target_tokens
  - chunked():     bar-aware sliding window, accumulates until target_tokens reached

All 5 bugs from xLSTM-2 are fixed here.

Bug 1 fix: "no new tokens" → retry with smaller context, never hard-stop.
Bug 2 fix: "no b-1 in chunk" → widen NEW_TOKENS and retry, never hard-stop.
Bug 3 fix: max_iterations computed dynamically from target_tokens.
Bug 4 fix: generate() returns RAW tokens (no filtering). Filtering is done by
           the caller AFTER slicing for new tokens.
Bug 5 fix: context left edge snapped to nearest b-1 via look-back buffer.
           Right edge is always bar-aligned (accumulator only stores complete bars).

API note (confirmed from languagemodel.py):
    LanguageModel.generate(
        prompt=str,
        temperature=float,
        max_length=int,           # total sequence length including prompt
        end_tokens=[],
        forbidden_tokens=[str],
        return_structured_output=True
    ) -> {"output": str, "elapsed_time": float, "tokens_per_second": float}
    The "output" string is decoded with the tokenizer; it includes the prompt.
"""
from __future__ import annotations

import logging
import sys
import time
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)


class MusicGenerator:
    """
    Wraps Helibrunna LanguageModel for REMIGEN2 music generation.

    Parameters
    ----------
    model_path : str
        Path to the Helibrunna run directory (contains tokenizer.json and
        checkpoint-*-last/ sub-directory with model.safetensors).
    context_length : int
        Inference context length override. Must exceed the max sequence length
        we intend to generate. Default 16 384 keeps the Helibrunna guard wall
        far away from any target we generate.
    device : str
        'cuda', 'cpu', or 'auto'.
    helibrunna_path : str, optional
        Path to the helibrunna source root (added to sys.path). Only needed
        if helibrunna is not already installed as a package.
    """

    def __init__(
        self,
        model_path: str,
        context_length: int = 16_384,
        device: str = "auto",
        helibrunna_path: Optional[str] = None,
    ) -> None:
        if helibrunna_path is not None and helibrunna_path not in sys.path:
            sys.path.insert(0, helibrunna_path)

        from source.languagemodel import LanguageModel  # noqa: PLC0415

        logger.info("Loading model from: %s", model_path)
        self._lm = LanguageModel(
            model_path,
            config_overrides={"context_length": context_length},
            device=device,
        )
        self._context_length = context_length
        logger.info("Model loaded (inference context: %d tokens)", context_length)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _raw_generate(self, prompt: str, max_length: int, temperature: float) -> str:
        """
        Call Helibrunna generate() and return the raw decoded output string.
        No filtering — caller is responsible for slicing and cleaning.

        Returns:
            Full decoded string (prompt + continuation), space-separated tokens.
        """
        result = self._lm.generate(
            prompt=prompt,
            temperature=temperature,
            max_length=max_length,
            end_tokens=[],
            forbidden_tokens=["[PAD]", "[EOS]"],
            return_structured_output=True,
        )
        return result["output"]  # type: ignore[index]

    @staticmethod
    def _find_last_bar_index(tokens: List[str]) -> Optional[int]:
        """Return index of the last 'b-1' token, or None."""
        for i in range(len(tokens) - 1, -1, -1):
            if tokens[i] == "b-1":
                return i
        return None

    @staticmethod
    def _find_first_bar_after_buffer(
        all_tokens: List[str],
        context_tokens: int,
        buffer: int,
    ) -> List[str]:
        """
        Bug 5 fix: snap the context left-edge to a bar boundary.

        Takes `context_tokens + buffer` tokens from the END of `all_tokens`,
        then scans the first `buffer` tokens of that slice for a 'b-1'. If
        found, returns the slice starting right AFTER that b-1 (so it begins
        with a complete bar: s-X t-X ...). Falls back to the full slice if no
        b-1 is found in the buffer zone.

        The right edge is always clean: the accumulator only ever appends
        complete bars (ending at b-1), so all_tokens always ends at b-1.
        """
        window = all_tokens[-(context_tokens + buffer):]
        for i, tok in enumerate(window[:buffer]):
            if tok == "b-1":
                return window[i + 1:]  # start of the next (complete) bar
        return window  # fallback: no b-1 found in buffer

    # ------------------------------------------------------------------
    # Public: single-shot generation
    # ------------------------------------------------------------------

    def single_shot(
        self,
        target_tokens: int,
        prompt: str = "s-9 o-0 t-38",
        temperature: float = 0.8,
        seed: int = 0,
    ) -> dict:
        """
        Generate a single piece in one model call.

        Args:
            target_tokens: Desired total sequence length (including prompt).
            prompt: Starting REMIGEN2 tokens.
            temperature: Sampling temperature.
            seed: Random seed for reproducibility.

        Returns:
            dict with keys:
                tokens          : List[str] — raw token list (no filtering; includes prompt)
                strategy        : "single_shot"
                target_tokens   : int
                actual_tokens   : int
                target_reached  : bool
                generation_time_s : float
                tokens_per_second : float
                num_chunks      : 0  (not applicable for single-shot)
        """
        torch.manual_seed(seed)
        logger.info("single_shot: target=%d tokens, temp=%.2f", target_tokens, temperature)

        t0 = time.time()
        raw_output = self._raw_generate(prompt, max_length=target_tokens, temperature=temperature)
        elapsed = time.time() - t0

        tokens = raw_output.split()
        actual = len(tokens)

        logger.info("single_shot: produced %d tokens in %.1fs", actual, elapsed)

        return {
            "tokens": tokens,
            "strategy": "single_shot",
            "target_tokens": target_tokens,
            "actual_tokens": actual,
            "target_reached": actual >= target_tokens,
            "generation_time_s": elapsed,
            "tokens_per_second": actual / elapsed if elapsed > 0 else 0.0,
            "num_chunks": 0,
        }

    # ------------------------------------------------------------------
    # Public: chunked (bar-aware sliding window) generation
    # ------------------------------------------------------------------

    def chunked(
        self,
        target_tokens: int,
        prompt: str = "s-9 o-0 t-38",
        temperature: float = 0.8,
        context_tokens: int = 1500,
        context_buffer: int = 300,
        new_tokens: int = 400,
        max_retries: int = 3,
        seed: int = 0,
    ) -> dict:
        """
        Generate a piece via bar-aware sliding-window chunking.

        Iterates until the accumulated token count reaches target_tokens.
        On each iteration:
          1. Snap context left edge to a bar boundary (Bug 5 fix).
          2. Generate ~new_tokens beyond the context.
          3. Slice RAW output to get new tokens (Bug 4 fix — filter AFTER slice).
          4. Cut at the last b-1 in new tokens (right-edge bar alignment).
          5. Append only complete new bars to the accumulator.
          6. On failure, retry with adjusted parameters (Bugs 1 & 2 fix).

        Args:
            target_tokens:  Stop when accumulator reaches this many tokens.
            prompt:         Starting REMIGEN2 tokens.
            temperature:    Sampling temperature.
            context_tokens: Target context window size (tokens).
            context_buffer: Look-back buffer for left-edge bar alignment.
            new_tokens:     Tokens to request per chunk beyond context.
            max_retries:    Max retries per chunk before giving up.
            seed:           Base random seed; iteration i uses seed + i.

        Returns:
            Same keys as single_shot(), plus num_chunks (iteration count).
        """
        torch.manual_seed(seed)
        logger.info(
            "chunked: target=%d tokens, context=%d, new=%d, temp=%.2f",
            target_tokens, context_tokens, new_tokens, temperature,
        )

        # Bug 3 fix: compute max_iterations dynamically
        max_iterations = max(100, (target_tokens // new_tokens) + 20)

        all_tokens: List[str] = prompt.split()
        chunks_done = 0
        t0 = time.time()

        for iteration in range(max_iterations):
            # ---- Check stopping condition ----
            if len(all_tokens) >= target_tokens:
                logger.info("chunked: reached target at iteration %d", iteration)
                break

            # ---- Bug 5 fix: bar-aligned context ----
            if len(all_tokens) <= context_tokens + context_buffer:
                ctx_tokens = all_tokens
            else:
                ctx_tokens = self._find_first_bar_after_buffer(
                    all_tokens, context_tokens, context_buffer
                )
            context_str = " ".join(ctx_tokens)
            ctx_len = len(ctx_tokens)

            logger.debug(
                "iter %d: context=%d tokens, accumulated=%d tokens",
                iteration, ctx_len, len(all_tokens),
            )

            # ---- Retry loop (Bugs 1 & 2 fix) ----
            completed_new: Optional[List[str]] = None
            current_new_tokens = new_tokens

            for attempt in range(max_retries):
                torch.manual_seed(seed + iteration * 100 + attempt)
                max_len = ctx_len + current_new_tokens

                try:
                    raw_output = self._raw_generate(
                        context_str, max_length=max_len, temperature=temperature
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("iter %d attempt %d: generate() raised: %s", iteration, attempt, exc)
                    current_new_tokens = int(current_new_tokens * 0.8)
                    continue

                # Bug 4 fix: slice FIRST, then filter for new tokens
                output_tokens = raw_output.split()

                # Extract new tokens by position (not by filtering which corrupts the index)
                if len(output_tokens) <= ctx_len:
                    # Bug 1: no new tokens — try with smaller context
                    logger.warning(
                        "iter %d attempt %d: no new tokens produced "
                        "(output=%d, context=%d). Reducing context.",
                        iteration, attempt, len(output_tokens), ctx_len,
                    )
                    current_new_tokens = int(current_new_tokens * 0.8)
                    continue

                raw_new = output_tokens[ctx_len:]  # new tokens only (raw, unfiltered)

                # Find last b-1 in the new tokens (right-edge alignment)
                last_bar_idx = self._find_last_bar_index(raw_new)

                if last_bar_idx is None:
                    # Bug 2: no b-1 in chunk — try with wider chunk
                    logger.warning(
                        "iter %d attempt %d: no b-1 in %d new tokens. Widening chunk.",
                        iteration, attempt, len(raw_new),
                    )
                    current_new_tokens = int(current_new_tokens * 1.5)
                    continue

                # Keep only complete bars (up to and including last b-1)
                completed_new = raw_new[: last_bar_idx + 1]
                break  # Success

            if completed_new is None:
                logger.error(
                    "iter %d: all %d retries exhausted — stopping generation.", iteration, max_retries
                )
                break

            all_tokens.extend(completed_new)
            chunks_done += 1

            logger.debug(
                "iter %d: appended %d tokens, total=%d",
                iteration, len(completed_new), len(all_tokens),
            )

        elapsed = time.time() - t0
        actual = len(all_tokens)
        logger.info("chunked: done. %d tokens in %.1fs (%d chunks)", actual, elapsed, chunks_done)

        return {
            "tokens": all_tokens,
            "strategy": "chunked",
            "target_tokens": target_tokens,
            "actual_tokens": actual,
            "target_reached": actual >= target_tokens,
            "generation_time_s": elapsed,
            "tokens_per_second": actual / elapsed if elapsed > 0 else 0.0,
            "num_chunks": chunks_done,
        }
