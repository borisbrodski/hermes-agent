"""Client-side reasoning-token watchdog.

This module provides :class:`ReasoningWatchdog`, a small, pure, side-effect-free
helper that counts reasoning ("thinking") tokens as they stream back from the
model and reports when a configured cap has been exceeded.

Why this exists
---------------
The model server (vLLM 0.21.0, Qwen3.6 + MTP speculative decoding) ships a
``thinking_token_budget`` request field that is *supposed* to bound the number of
reasoning tokens the model emits before it must answer.  On the MTP speculative-
decoding path that field is **not enforced** (vLLM issue #39573): the draft
tokens accepted by MTP slip past the budget state machine, so a "budget=64"
request can stream 5000+ reasoning tokens before the model ever emits
``</think>``.  Runaway reasoning is the proximate cause of agent drift (an
8744-token reasoning spiral that never reaches a useful answer).

Because the server-side enforcement is broken and the fix is intricate and
upgrade-fragile, we enforce the cap on the client.  As reasoning deltas arrive in
the streaming loop we feed their text to a watchdog; once the running token
estimate crosses ``max_reasoning_tokens`` we trip, and the caller interrupts the
in-flight stream so the turn ends bounded instead of spiralling.

This guard is intentionally model-agnostic and lives entirely in our code, so it
survives vLLM upgrades and model swaps and acts as cheap belt-and-suspenders even
once the upstream budget bug is fixed.

Design notes
------------
* **Pure / unit-testable.**  No global state, no I/O, no callbacks.  All state
  lives on the instance and is reset per turn via :meth:`reset`.
* **Token estimation.**  Exact token counts require the model tokenizer, which is
  *not* cheaply available inside the streaming hot loop (the agent talks to an
  OpenAI-compatible HTTP endpoint and does not hold the tokenizer in that scope).
  We therefore use a conservative character-to-token heuristic.  For the models
  in use (Qwen3.6 family, English+code reasoning) ~3.5-4 chars per token is
  typical; we use ``CHARS_PER_TOKEN = 4.0`` which slightly *under*-estimates the
  token count (i.e. the watchdog trips a little *later* than a perfect counter
  would), erring toward not cutting a borderline-legitimate reasoning turn too
  early.  The estimate is monotonic and stable across deltas because we count
  characters, not re-tokenized boundaries.  A pluggable ``token_counter`` hook is
  provided so a caller that *does* hold a tokenizer can pass an exact counter.
"""

from __future__ import annotations

from typing import Callable, Optional

# Conservative default cap.  Matches the ``thinking_token_budget`` historically
# requested from vLLM (greentech-coder ``config.yaml`` -> request_overrides ->
# extra_body -> thinking_token_budget: 4096) so the client guard mirrors the
# server budget the model was already asked to honour.
DEFAULT_MAX_REASONING_TOKENS = 4096

# Average characters per reasoning token used by the fallback estimator.  See the
# module docstring for the rationale; slightly under-estimates to avoid early
# cuts on legitimate reasoning.
CHARS_PER_TOKEN = 4.0


def estimate_tokens(text: str, chars_per_token: float = CHARS_PER_TOKEN) -> int:
    """Estimate the number of tokens in ``text`` from its character length.

    A defensible, dependency-free heuristic used when the real tokenizer is not
    available in the calling scope.  Rounds up so any non-empty text counts as at
    least one token (a stream of tiny 1-2 char deltas still accumulates).
    """
    if not text:
        return 0
    if chars_per_token <= 0:
        # Degenerate config; fall back to a 1-char-per-token upper bound rather
        # than dividing by zero.
        return len(text)
    # Ceil division so partial tokens round up.
    return int((len(text) + chars_per_token - 1) // chars_per_token)


class ReasoningWatchdog:
    """Counts streamed reasoning tokens and trips when a cap is exceeded.

    Typical usage inside a streaming loop::

        watchdog = ReasoningWatchdog(max_reasoning_tokens=4096)
        watchdog.reset()                       # once per turn / per stream call
        for chunk in stream:
            if reasoning_text:
                if watchdog.note_reasoning_delta(reasoning_text):
                    interrupt_the_stream()     # cap exceeded -> end turn bounded

    The watchdog is *pure*: it neither performs I/O nor mutates anything outside
    its own instance, which makes it trivially unit-testable.

    Parameters
    ----------
    max_reasoning_tokens:
        The reasoning-token cap.  A value ``<= 0`` (or ``None``) disables the
        watchdog entirely — :meth:`note_reasoning_delta` then always returns
        ``False`` and :attr:`enabled` is ``False``.
    token_counter:
        Optional callable ``(str) -> int`` used to count tokens in a delta.  When
        omitted, the character-based :func:`estimate_tokens` heuristic is used.
        Pass a real tokenizer-backed counter for exact accounting when one is
        cheaply available.
    """

    def __init__(
        self,
        max_reasoning_tokens: Optional[int] = DEFAULT_MAX_REASONING_TOKENS,
        token_counter: Optional[Callable[[str], int]] = None,
    ) -> None:
        # Normalise: treat None / non-positive as "disabled".
        self.max_reasoning_tokens: int = int(max_reasoning_tokens or 0)
        self._token_counter: Callable[[str], int] = token_counter or estimate_tokens
        self._tokens: int = 0
        self._tripped: bool = False

    # -- introspection ----------------------------------------------------

    @property
    def enabled(self) -> bool:
        """True when a positive cap is configured."""
        return self.max_reasoning_tokens > 0

    @property
    def tokens(self) -> int:
        """Estimated reasoning tokens accumulated since the last :meth:`reset`."""
        return self._tokens

    @property
    def tripped(self) -> bool:
        """True once the cap has been exceeded (until the next :meth:`reset`)."""
        return self._tripped

    # -- lifecycle --------------------------------------------------------

    def reset(self) -> None:
        """Clear accumulated state.  Call once at the start of every turn/stream."""
        self._tokens = 0
        self._tripped = False

    # -- hot path ---------------------------------------------------------

    def note_reasoning_delta(self, text: str) -> bool:
        """Account for one reasoning delta; return True iff the cap is exceeded.

        Returns ``True`` only on the *first* delta that pushes the running total
        strictly past the cap, and ``True`` for every subsequent delta while the
        watchdog stays tripped (it never auto-resets).  This lets callers guard on
        ``if not already_handled and watchdog.note_reasoning_delta(...)`` to fire
        the interrupt exactly once, or simply check the boolean each iteration.

        When the watchdog is disabled (cap ``<= 0``) this always returns ``False``
        and performs no accounting.
        """
        if not self.enabled:
            return False
        if text:
            self._tokens += self._token_counter(text)
        if self._tokens > self.max_reasoning_tokens:
            self._tripped = True
        return self._tripped
