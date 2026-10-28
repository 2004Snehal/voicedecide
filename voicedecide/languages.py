"""Language codes, names, aliases, and the tested-language policy.

Codes follow FLEURS config names (``hi_in``, ``en_us``...). Only the languages in
``TESTED`` have been benchmarked; every other language is reported as untested
and no accuracy figure is ever shown for it.
"""

from __future__ import annotations

import difflib

from .errors import UnknownLanguageError

LANGUAGES: dict[str, str] = {
    # Indian subcontinent
    "hi_in": "Hindi", "bn_in": "Bengali", "te_in": "Telugu", "mr_in": "Marathi", "ta_in": "Tamil",
    "ur_pk": "Urdu", "gu_in": "Gujarati", "kn_in": "Kannada", "ml_in": "Malayalam", "or_in": "Odia",
    "pa_in": "Punjabi", "as_in": "Assamese", "ne_np": "Nepali", "sd_in": "Sindhi",
    # Global
    "en_us": "English", "cmn_hans_cn": "Mandarin Chinese", "yue_hant_hk": "Cantonese",
    "es_419": "Spanish", "pt_br": "Portuguese", "fr_fr": "French", "it_it": "Italian", "de_de": "German",
    "nl_nl": "Dutch", "ru_ru": "Russian", "uk_ua": "Ukrainian", "pl_pl": "Polish", "tr_tr": "Turkish",
    "ar_eg": "Arabic", "fa_ir": "Persian", "ja_jp": "Japanese", "ko_kr": "Korean", "vi_vn": "Vietnamese",
    "th_th": "Thai", "my_mm": "Burmese", "id_id": "Indonesian", "ms_my": "Malay", "fil_ph": "Filipino",
    "sw_ke": "Swahili", "ha_ng": "Hausa", "am_et": "Amharic",
}

# Benchmarked in the EmbeddingGemma 2 Kaggle study (FLEURS test split).
TESTED: frozenset[str] = frozenset({"en_us", "hi_in", "bn_in", "ta_in", "te_in", "mr_in"})

_ALIASES: dict[str, str] = {
    "en": "en_us", "hi": "hi_in", "bn": "bn_in", "te": "te_in", "mr": "mr_in", "ta": "ta_in",
    "ur": "ur_pk", "gu": "gu_in", "kn": "kn_in", "ml": "ml_in", "or": "or_in", "od": "or_in",
    "oriya": "or_in", "pa": "pa_in", "as": "as_in", "ne": "ne_np", "sd": "sd_in",
    "zh": "cmn_hans_cn", "cmn": "cmn_hans_cn", "mandarin": "cmn_hans_cn", "chinese": "cmn_hans_cn",
    "yue": "yue_hant_hk", "es": "es_419", "pt": "pt_br", "fr": "fr_fr", "it": "it_it", "de": "de_de",
    "nl": "nl_nl", "ru": "ru_ru", "uk": "uk_ua", "pl": "pl_pl", "tr": "tr_tr", "ar": "ar_eg",
    "fa": "fa_ir", "farsi": "fa_ir", "ja": "ja_jp", "ko": "ko_kr", "vi": "vi_vn", "th": "th_th",
    "my": "my_mm", "myanmar": "my_mm", "id": "id_id", "ms": "ms_my", "fil": "fil_ph", "tl": "fil_ph",
    "tagalog": "fil_ph", "sw": "sw_ke", "ha": "ha_ng", "am": "am_et", "bangla": "bn_in",
}


def display_name(code: str) -> str:
    return LANGUAGES.get(code, code)


def _lookup_keys(available: list[str]) -> dict[str, str]:
    keys = {}
    for code in available:
        keys[code.lower()] = code
        keys[display_name(code).lower()] = code
    for alias, code in _ALIASES.items():
        if code in available:
            keys.setdefault(alias, code)
    return keys


def resolve(name: str, available: list[str], pack: str = "this pack") -> str:
    """Map a code, short code, or name ('hindi', 'hi', 'hi_in') to a code in ``available``.

    Raises UnknownLanguageError with a did-you-mean suggestion when nothing matches.
    """
    keys = _lookup_keys(available)
    key = name.strip().lower().replace("-", "_")
    if key in keys:
        return keys[key]
    close = difflib.get_close_matches(key, list(keys), n=3, cutoff=0.6)
    suggestions = sorted({keys[c] for c in close})
    hint = ""
    if suggestions:
        hint = "\n  Did you mean: " + ", ".join(f"'{c}' ({display_name(c)})" for c in suggestions) + "?"
    raise UnknownLanguageError(
        f"'{name}' is not in pack '{pack}'.{hint}\n"
        "  List languages:  voicedecide languages\n"
        f"  Add your own:    lang.add_examples({{'{key}': clips}})"
    )


def untested_message(codes: list[str]) -> str:
    """Fixed warning text. Deliberately contains no accuracy or confidence figures."""
    names = ", ".join(f"'{c}' ({display_name(c)})" for c in codes)
    verb = "has" if len(codes) == 1 else "have"
    tested = ", ".join(display_name(c) for c in sorted(TESTED, key=display_name))
    return (
        f"{names} {verb} not been tested.\n"
        f"  Tested languages: {tested}.\n"
        "  Results for untested languages may be unreliable. Check them on your own labeled clips:\n"
        f"  voicedecide eval --pack language-id --languages {','.join(codes)} your_clips.csv"
    )
