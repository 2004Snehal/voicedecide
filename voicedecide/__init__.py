"""voicedecide: training-free audio decisions (call routing, language ID) on EmbeddingGemma 2.

    from voicedecide import Decider, Choice

    d = Decider()
    route = Choice.from_descriptions({"balance": "A caller asking for their account balance", ...})
    res = d.decide("call.wav", {"route": route})
    res["route"].choice, res["route"].probabilities, res["route"].action
"""

from .audio import load_audio
from .calibrate import FitReport
from .decide import Decider
from .errors import (
    LeakageWarning,
    ModelRevisionWarning,
    OutOfSetError,
    PackError,
    UnknownLabelError,
    UnknownLanguageError,
    UntestedLanguageWarning,
    VoiceDecideError,
    ZeroShotHint,
)
from .languages import TESTED as TESTED_LANGUAGES
from .model import MODEL_ID, Encoder
from .packs import list_languages
from .questions import Choice
from .results import Answer, DecisionResult

__version__ = "0.1.0a2"

__all__ = [
    "Answer", "Choice", "DecisionResult", "Decider", "Encoder", "FitReport", "LeakageWarning", "MODEL_ID",
    "ModelRevisionWarning", "OutOfSetError", "PackError", "TESTED_LANGUAGES", "UnknownLabelError",
    "UnknownLanguageError", "UntestedLanguageWarning", "VoiceDecideError", "ZeroShotHint", "list_languages",
    "load_audio",
]
