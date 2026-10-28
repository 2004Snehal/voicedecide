"""Exceptions and warnings raised by voicedecide."""


class VoiceDecideError(Exception):
    """Base class for voicedecide errors."""


class PackError(VoiceDecideError):
    """A pack is missing, malformed, or incompatible."""


class UnknownLabelError(VoiceDecideError, KeyError):
    """A label was requested that the question does not have."""

    def __str__(self):  # KeyError would otherwise wrap the message in quotes
        return str(self.args[0]) if self.args else ""


class UnknownLanguageError(UnknownLabelError):
    """A language was requested that the pack does not contain."""


class OutOfSetError(VoiceDecideError):
    """Raised in strict mode when audio does not match any option of a question."""

    def __init__(self, question: str, similarity: float, min_similarity: float):
        self.question = question
        self.similarity = similarity
        self.min_similarity = min_similarity
        super().__init__(
            f"Audio does not match any option of question '{question}' "
            f"(best similarity {similarity:.3f} < pack minimum {min_similarity:.3f})."
        )


class VoiceDecideWarning(UserWarning):
    """Base class for voicedecide warnings."""


class UntestedLanguageWarning(VoiceDecideWarning):
    """A language without a benchmark was selected."""


class ZeroShotHint(VoiceDecideWarning):
    """One-time tip that prototypes usually beat zero-shot text anchors."""


class LeakageWarning(VoiceDecideWarning):
    """Validation clips overlap with the clips used to build prototypes."""


class ModelRevisionWarning(VoiceDecideWarning):
    """A question was built with a different model revision than the one loaded."""


class AudioTruncatedWarning(VoiceDecideWarning):
    """Audio was longer than the model's context and was cut."""
