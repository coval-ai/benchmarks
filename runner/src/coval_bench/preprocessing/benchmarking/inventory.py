# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Versioned English phone inventory and source-specific normalization."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

PHONE_INVENTORY_VERSION = "coval-english-arpabet-v1"
COVAL_ENGLISH_PHONES_V1 = (
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "ER",
    "EY",
    "F",
    "G",
    "HH",
    "IH",
    "IY",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OY",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UW",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
    "UNK",
)

_INVENTORY = frozenset(COVAL_ENGLISH_PHONES_V1)
_STRESS_SUFFIX = re.compile(r"[012]$")


class PhoneSource(StrEnum):
    COVAL_ARPABET = "coval_arpabet"
    TIMIT_61 = "timit_61"
    BUCKEYE = "buckeye"
    META_ESPEAK_IPA = "meta_espeak_ipa"
    ALLOPHANT_IPA = "allophant_ipa"
    CHARSIU_CMU = "charsiu_cmu"


@dataclass(frozen=True, slots=True)
class PhoneNormalizationResult:
    symbols: tuple[str, ...]
    source_count: int
    unknown_count: int
    ignored_count: int

    @property
    def loss_rate(self) -> float:
        assessable = self.source_count - self.ignored_count
        return self.unknown_count / assessable if assessable else 0.0


_IGNORED = frozenset(
    {
        "",
        "<eps>",
        "<pad>",
        "[pad]",
        "[sil]",
        "h#",
        "pau",
        "epi",
        "sil",
        "sp",
        "bcl",
        "dcl",
        "gcl",
        "kcl",
        "pcl",
        "tcl",
    }
)

_TIMIT_MAP: dict[str, tuple[str, ...]] = {
    "ax": ("AH",),
    "ax-h": ("AH",),
    "axr": ("ER",),
    "dx": ("D",),
    "el": ("L",),
    "em": ("M",),
    "en": ("N",),
    "eng": ("NG",),
    "hv": ("HH",),
    "ix": ("IH",),
    "nx": ("N",),
    "q": ("UNK",),
    "ux": ("UW",),
}

_META_IPA_MAP: dict[str, tuple[str, ...]] = {
    "a": ("AA",),
    "aɪ": ("AY",),
    "aʊ": ("AW",),
    "b": ("B",),
    "d": ("D",),
    "dʒ": ("JH",),
    "e": ("EY",),
    "eɪ": ("EY",),
    "f": ("F",),
    "h": ("HH",),
    "i": ("IY",),
    "iː": ("IY",),
    "j": ("Y",),
    "k": ("K",),
    "l": ("L",),
    "m": ("M",),
    "n": ("N",),
    "o": ("OW",),
    "oʊ": ("OW",),
    "p": ("P",),
    "r": ("R",),
    "s": ("S",),
    "t": ("T",),
    "tʃ": ("CH",),
    "u": ("UW",),
    "uː": ("UW",),
    "v": ("V",),
    "w": ("W",),
    "z": ("Z",),
    "æ": ("AE",),
    "ð": ("DH",),
    "ŋ": ("NG",),
    "ɐ": ("AH",),
    "ɑ": ("AA",),
    "ɑː": ("AA",),
    "ɑːɹ": ("AA", "R"),
    "ɔ": ("AO",),
    "ɔː": ("AO",),
    "ə": ("AH",),
    "əl": ("AH", "L"),
    "ɚ": ("ER",),
    "ɛ": ("EH",),
    "ɜ": ("ER",),
    "ɜː": ("ER",),
    "ɡ": ("G",),
    "ɪ": ("IH",),
    "ɪɹ": ("IH", "R"),
    "ɹ": ("R",),
    "ɾ": ("D",),
    "ʃ": ("SH",),
    "ʊ": ("UH",),
    "ʌ": ("AH",),
    "ʒ": ("ZH",),
    "θ": ("TH",),
}

_ALLOPHANT_IPA_MAP: dict[str, tuple[str, ...]] = {
    **_META_IPA_MAP,
    "d̠ʒ": ("JH",),
    "eɪ̯": ("EY",),
    "ɚː": ("ER",),
    "iɪ": ("IY",),
    "ɔɪ": ("OY",),
    "t̠ʃ": ("CH",),
}


def _normalize_arpabet(symbol: str) -> tuple[str, ...] | None:
    normalized = _STRESS_SUFFIX.sub("", symbol.strip().upper())
    if normalized in _INVENTORY:
        return (normalized,)
    return None


def _normalize_one(symbol: str, source: PhoneSource) -> tuple[str, ...] | None:
    raw = symbol.strip()
    lowercase = raw.lower()
    if lowercase in _IGNORED:
        return ()
    if source in {PhoneSource.COVAL_ARPABET, PhoneSource.CHARSIU_CMU}:
        return _normalize_arpabet(raw)
    if source is PhoneSource.TIMIT_61:
        mapped = _TIMIT_MAP.get(lowercase)
        return mapped if mapped is not None else _normalize_arpabet(raw)
    if source is PhoneSource.BUCKEYE:
        base = lowercase.rstrip(";:+~")
        mapped = _TIMIT_MAP.get(base)
        return mapped if mapped is not None else _normalize_arpabet(base)
    if source is PhoneSource.ALLOPHANT_IPA:
        return _ALLOPHANT_IPA_MAP.get(raw)
    return _META_IPA_MAP.get(raw)


def normalize_phone_sequence(
    symbols: tuple[str, ...], *, source: PhoneSource
) -> PhoneNormalizationResult:
    """Normalize without silently dropping unknown speech-phone labels."""
    normalized: list[str] = []
    unknown_count = 0
    ignored_count = 0
    for symbol in symbols:
        mapped = _normalize_one(symbol, source)
        if mapped == ():
            ignored_count += 1
            continue
        if mapped is None:
            normalized.append("UNK")
            unknown_count += 1
            continue
        if mapped == ("UNK",):
            unknown_count += 1
        normalized.extend(mapped)
    return PhoneNormalizationResult(
        symbols=tuple(normalized),
        source_count=len(symbols),
        unknown_count=unknown_count,
        ignored_count=ignored_count,
    )
