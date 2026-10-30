import logging
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)


class KimiBasetenEncoder:
    def __init__(self, vocab_file, pattern, special_tokens, original):
        self.vocab_file = str(Path(vocab_file).resolve(strict=True))
        self.pattern = pattern
        self.special_tokens = dict(special_tokens)
        self._original = original
        self._backend = None
        self._lock = Lock()

    def __getstate__(self):
        return self.vocab_file, self.pattern, self.special_tokens, self._original

    def __setstate__(self, state):
        self.__init__(*state)

    def _get_backend(self):
        if self._backend is None:
            with self._lock:
                if self._backend is None:
                    from basetenkenizer import Tokenizer

                    self._backend = Tokenizer.from_tiktoken_model(
                        self.vocab_file,
                        pattern=self.pattern,
                        special_tokens=self.special_tokens,
                    )
        return self._backend

    def encode_segments(self, segments):
        pieces = [(segment.text, segment.allow_special) for segment in segments]
        try:
            return (
                self._get_backend()
                .encode_segments(pieces, add_special_tokens=False, tiktoken_safe=True)
                .ids
            )
        except UnicodeEncodeError:
            return [
                token
                for text, policy in pieces
                for token in self.encode_text_piece(text, policy)
            ]

    def encode_text_piece(self, text, allow_special_tokens=True):
        try:
            return (
                self._get_backend()
                .encode_segments(
                    [(text, allow_special_tokens)],
                    add_special_tokens=False,
                    tiktoken_safe=True,
                )
                .ids
            )
        except UnicodeEncodeError:
            return type(self._original)._encode_text_piece(
                self._original, text, allow_special_tokens=allow_special_tokens
            )


def enable_baseten_kimi(tokenizer):
    """Use Baseten for Kimi K3 encoding while retaining its original renderer."""
    if isinstance(
        getattr(tokenizer, "_sglang_baseten_encoder", None), KimiBasetenEncoder
    ):
        return tokenizer

    special_tokens = getattr(tokenizer, "special_tokens", {})
    if (
        type(tokenizer).__name__ != "TikTokenTokenizer"
        or "tokenization_kimi" not in type(tokenizer).__module__
        or not callable(getattr(tokenizer, "_encode_chat_segments", None))
        or not callable(getattr(tokenizer, "_encode_text_piece", None))
        or not isinstance(special_tokens, dict)
        or len(special_tokens) != 256
        or special_tokens.get("[BOS]") != 163584
        or special_tokens.get("<|open|>") != 163587
        or special_tokens.get("<|end_of_msg|>") != 163586
    ):
        raise ValueError(
            "--tokenizer-backend=baseten supports the original Kimi K3 tokenizer only"
        )

    try:
        from basetenkenizer import Tokenizer
    except ImportError as exc:
        raise ImportError(
            "Install the pinned Baseten wheel in the serving image before selecting baseten"
        ) from exc
    if not hasattr(Tokenizer, "from_tiktoken_model"):
        raise ImportError(
            "The baseten backend requires a Baseten build with native tiktoken.model loading"
        )

    encoder = KimiBasetenEncoder(
        tokenizer.vocab_file, tokenizer.pat_str, special_tokens, tokenizer
    )
    tokenizer._sglang_baseten_encoder = encoder
    tokenizer._encode_chat_segments = encoder.encode_segments
    tokenizer._encode_text_piece = encoder.encode_text_piece
    logger.info("Baseten Kimi K3 encoder selected: %s", encoder.vocab_file)
    return tokenizer
