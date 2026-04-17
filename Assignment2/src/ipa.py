"""
BLOCK 5: IPA Unified Representation
Task 2.1: Convert code-switched EN/HI transcript → unified IPA string.
- English: eng_to_ipa library
- Hindi (Devanagari): custom rule-based G2P mapping for Hinglish phonology
- Saves ipa_results.json
"""

import os
import json
import re
from typing import List

from tracking import wb_save, hf_upload

# ─── Hindi Devanagari → IPA rule-based G2P ────────────────────────────────────
_VOWEL_SIGNS = {
    "\u093E": "aː",
    "\u093F": "ɪ",
    "\u0940": "iː",
    "\u0941": "ʊ",
    "\u0942": "uː",
    "\u0943": "rɪ",
    "\u0947": "eː",
    "\u0948": "ɛ",
    "\u094B": "oː",
    "\u094C": "ɔ",
    "\u094D": "",
    "\u0902": "n",
    "\u0903": "h",
    "\u0901": "n",
}

_CONSONANTS = {
    "\u0915": "k",
    "\u0916": "kʰ",
    "\u0917": "ɡ",
    "\u0918": "ɡʰ",
    "\u0919": "ŋ",
    "\u091A": "tʃ",
    "\u091B": "tʃʰ",
    "\u091C": "dʒ",
    "\u091D": "dʒʰ",
    "\u091E": "ɲ",
    "\u091F": "ʈ",
    "\u0920": "ʈʰ",
    "\u0921": "ɖ",
    "\u0922": "ɖʰ",
    "\u0923": "ɳ",
    "\u0924": "t",
    "\u0925": "tʰ",
    "\u0926": "d",
    "\u0927": "dʰ",
    "\u0928": "n",
    "\u092A": "p",
    "\u092B": "pʰ",
    "\u092C": "b",
    "\u092D": "bʰ",
    "\u092E": "m",
    "\u092F": "j",
    "\u0930": "r",
    "\u0932": "l",
    "\u0935": "ʋ",
    "\u0936": "ʃ",
    "\u0937": "ʂ",
    "\u0938": "s",
    "\u0939": "ɦ",
    "\u0933": "ɭ",
    "\u0915\u093C": "q",
    "\u0916\u093C": "x",
    "\u0917\u093C": "ɣ",
    "\u091C\u093C": "z",
    "\u0921\u093C": "ɽ",
    "\u0922\u093C": "ɽʰ",
    "\u092B\u093C": "f",
}

_INDEPENDENT_VOWELS = {
    "\u0905": "ə",
    "\u0906": "aː",
    "\u0907": "ɪ",
    "\u0908": "iː",
    "\u0909": "ʊ",
    "\u090A": "uː",
    "\u090B": "rɪ",
    "\u090F": "eː",
    "\u0910": "ɛ",
    "\u0913": "oː",
    "\u0914": "ɔ",
}

_PUNCT_IPA = {",": " | ", ".": " ‖ ", "।": " ‖ ", "!": " ‖ ", "?": " ‖ "}


def _devanagari_to_ipa(text: str) -> str:
    out = []
    i = 0
    while i < len(text):
        c = text[i]
        if (i + 1 < len(text) and (c + text[i + 1]) in _CONSONANTS):
            cons = _CONSONANTS[c + text[i + 1]]
            i += 2
            if i < len(text) and text[i] in _VOWEL_SIGNS:
                v = _VOWEL_SIGNS[text[i]]
                out.append(cons + (v if v else ""))
                i += 1
            elif i < len(text) and text[i] in _CONSONANTS:
                out.append(cons + "ə")
            else:
                out.append(cons + "ə")
        elif c in _CONSONANTS:
            cons = _CONSONANTS[c]
            i += 1
            if i < len(text) and text[i] in _VOWEL_SIGNS:
                v = _VOWEL_SIGNS[text[i]]
                out.append(cons + (v if v else ""))
                i += 1
            elif i < len(text) and text[i] in _CONSONANTS:
                out.append(cons + "ə")
            else:
                out.append(cons + "ə")
        elif c in _INDEPENDENT_VOWELS:
            out.append(_INDEPENDENT_VOWELS[c])
            i += 1
        elif c in _VOWEL_SIGNS:
            out.append(_VOWEL_SIGNS[c])
            i += 1
        elif c in _PUNCT_IPA:
            out.append(_PUNCT_IPA[c])
            i += 1
        elif c == " ":
            out.append(" ")
            i += 1
        else:
            i += 1
    return "".join(out).strip()


def _english_to_ipa(text: str) -> str:
    try:
        import eng_to_ipa as ipa
        result = ipa.convert(text)
        result = re.sub(r"\*(\S+)", r"\1", result)
        return result.strip()
    except ImportError:
        pass

    text = text.lower()
    replacements = [
        (r"\bthe\b", "ðə"), (r"\ba\b", "ə"), (r"\band\b", "ænd"),
        (r"\bof\b", "ɒv"), (r"\bin\b", "ɪn"), (r"\bis\b", "ɪz"),
        (r"tion\b", "ʃən"), (r"sion\b", "ʒən"), (r"ph", "f"),
        (r"ck\b", "k"), (r"([aeiou])r\b", r"\1ɹ"),
        (r"th", "θ"), (r"sh", "ʃ"), (r"ch", "tʃ"),
        (r"ng\b", "ŋ"), (r"wh", "w"),
    ]
    for pat, rep in replacements:
        text = re.sub(pat, rep, text)
    return text.strip()


def _chunk_to_ipa(chunk: dict) -> str:
    text = chunk.get("text", "").strip()
    lang = chunk.get("lang", "EN").upper()
    if not text:
        return ""
    if lang == "HI":
        return _devanagari_to_ipa(text)
    return _english_to_ipa(text)


def ipa_convert(segs: List[dict], transcript: dict, cfg: dict) -> dict:
    print("\n" + "=" * 60)
    print("BLOCK 5  IPA Unified Representation (EN + HI G2P)")
    print("=" * 60)

    cache = cfg["ipa_json"]
    if os.path.exists(cache):
        print(f"  [cache] {cache}")
        return json.load(open(cache, encoding="utf-8"))

    try:
        import eng_to_ipa
    except ImportError:
        print("  Installing eng_to_ipa ...")
        import subprocess
        subprocess.run(["pip", "install", "eng_to_ipa", "-q"], check=False)

    all_chunks = transcript.get("segments", [])
    ipa_chunks = []
    full_ipa_parts = []

    print(f"  Converting {len(all_chunks)} transcript chunks to IPA...")
    for i, chunk in enumerate(all_chunks):
        ipa_str = _chunk_to_ipa(chunk)
        ipa_chunks.append({
            "start": chunk.get("start", 0),
            "end": chunk.get("end", 0),
            "original": chunk.get("text", ""),
            "lang": chunk.get("lang", "EN"),
            "ipa": ipa_str,
            "seg_idx": chunk.get("seg_idx", 0),
        })
        full_ipa_parts.append(ipa_str)
        if (i + 1) % 20 == 0:
            print(f"    {i + 1}/{len(all_chunks)} done")

    full_ipa = " ".join(full_ipa_parts)
    res = {
        "ipa_chunks": ipa_chunks,
        "full_ipa": full_ipa,
        "n_chunks": len(ipa_chunks),
        "sample": full_ipa[:300],
    }

    for seg in segs:
        seg["ipa"] = [c for c in ipa_chunks if c.get("seg_idx") == seg["idx"]]

    json.dump(res, open(cache, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n  {len(ipa_chunks)} IPA chunks  ({len(full_ipa)} chars)")
    print(f"  Sample: {full_ipa[:200]}")
    print(f"  Saved: {cache}")
    wb_save(cache)
    hf_upload(cache, "outputs/ipa_results.json", cfg)
    return res