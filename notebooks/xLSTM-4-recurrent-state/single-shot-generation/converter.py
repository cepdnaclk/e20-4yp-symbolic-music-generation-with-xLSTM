"""
generate/converter.py
=====================
MIDIConverter — converts REMIGEN2 token lists to MIDI files.

Encoding confirmed from:
  - repos/MidiProcessor/midiprocessor/enc_remigen2_utils.py  (encoder used for LMD)
  - Tokenizer vocab inspection: tokenizer.json contains s-X, t-X, b-X, o-X tokens,
    with s-X and t-X emitted at the START OF EVERY BAR (REMIGEN2 behaviour).

We use MidiDecoder('REMIGEN2') because:
  1. The LMD data was encoded with enc_remigen2_utils (s-X t-X per bar, not per change).
  2. enc_remigen2_utils.generate_midi_obj_from_remigen_token_list() has extra robustness:
     it appends a fallback time_signature if none were found. enc_remigen_utils does not.
  3. xLSTM-2 used MidiDecoder('REMIGEN') by oversight — both decoders parse the same
     token strings, so it worked, but 'REMIGEN2' is semantically correct.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class MIDIConverter:
    """
    Converts REMIGEN2 token lists to MIDI files via midiprocessor.

    Parameters
    ----------
    midiprocessor_path : str, optional
        Path to the MidiProcessor source root. Added to sys.path if given
        and if midiprocessor is not already importable.
    """

    def __init__(self, midiprocessor_path: Optional[str] = None) -> None:
        if midiprocessor_path is not None and midiprocessor_path not in sys.path:
            sys.path.insert(0, midiprocessor_path)

        import midiprocessor as mp  # noqa: PLC0415
        # 'REMIGEN2': LMD data was encoded with enc_remigen2_utils (s-X t-X per bar).
        self._decoder = mp.MidiDecoder("REMIGEN2")
        logger.debug("MIDIConverter initialised with REMIGEN2 decoder.")

    def tokens_to_midi(
        self,
        tokens: List[str],
        output_path: str,
        *,
        use_clean_fallback: bool = True,
    ) -> bool:
        """
        Convert a REMIGEN2 token list to a MIDI file and write it to disk.

        Tries direct decode first. If that fails and use_clean_fallback=True,
        applies token_analysis.clean_tokens() and retries.

        Args:
            tokens:              List of raw REMIGEN2 token strings.
            output_path:         Destination .mid file path (directories created).
            use_clean_fallback:  Whether to retry with cleaned tokens on failure.

        Returns:
            True on success, False if decoding failed even after cleaning.
        """
        from token_analysis import clean_tokens  # noqa: PLC0415

        # First attempt: raw tokens
        success, cleaned_used = self._try_decode(tokens, output_path)
        if success:
            return True

        if not use_clean_fallback:
            logger.error("Decode failed for %s (no fallback).", output_path)
            return False

        # Second attempt: cleaned tokens
        logger.warning("Direct decode failed for %s — retrying with cleaned tokens.", output_path)
        cleaned = clean_tokens(tokens)
        if not cleaned:
            logger.error("Token list empty after cleaning — giving up.")
            return False

        success, _ = self._try_decode(cleaned, output_path)
        if not success:
            logger.error("Decode still failed after cleaning for %s.", output_path)
        return success

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _try_decode(self, tokens: List[str], output_path: str) -> tuple[bool, bool]:
        """
        Attempt to decode `tokens` and write to `output_path`.

        Returns:
            (success: bool, cleaned_used: bool)  — second element always False here.
        """
        try:
            midi_obj = self._decoder.decode_from_token_str_list(tokens)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            midi_obj.dump(output_path)
            return True, False
        except Exception as exc:  # noqa: BLE001
            logger.debug("Decode error (%s): %s", type(exc).__name__, exc)
            return False, False
