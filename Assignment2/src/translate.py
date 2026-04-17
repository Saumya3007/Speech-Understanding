"""
BLOCK 6: EN/HI -> Maithili Translation  (junk-filtered)
"""
import os, json, re, time
from typing import List
from tracking import wb_log, wb_save, hf_upload

_JUNK_RE = re.compile(r'^[-mcsqtf0-9\s.]+$', re.IGNORECASE)

def _is_junk(text: str) -> bool:
    t = text.strip()
    if len(t) < 5: return True
    if _JUNK_RE.match(t): return True
    cleaned = re.sub(r'[-\s]', '', t)
    if len(cleaned) > 4:
        from collections import Counter
        top = Counter(cleaned).most_common(1)[0][1]
        if top / len(cleaned) > 0.70: return True
    if t.count('-') / max(len(t), 1) > 0.4: return True
    return False

def _translate_chunk(text, translator):
    text = text.strip()
    if not text or _is_junk(text): return ""
    for attempt in range(3):
        try:
            result = translator.translate(text)
            if result and result.strip(): return result.strip()
        except Exception as e:
            if attempt < 2: time.sleep(1.5 * (attempt + 1))
            else: print(f"    [translate] failed: {e}")
    return text

def translate(segs: List[dict], cfg: dict) -> dict:
    print("\n" + "=" * 60)
    print("BLOCK 6  Translating Transcript -> Maithili")
    print("=" * 60)

    cache = cfg["lrl_json"]
    if os.path.exists(cache):
        d = json.load(open(cache, encoding="utf-8"))
        total = sum(len(v) for v in d.get("seg_lrl", {}).values())
        if total > 0:
            print(f"  [cache] {cache}  ({total} chars)")
            for seg in segs:
                seg["lrl_text"] = d["seg_lrl"].get(seg["label"], "")
            return d
        else:
            print("  [cache] stale (0 chars) — recomputing")
            os.remove(cache)

    t_path = cfg["transcript_json"]
    if not os.path.exists(t_path):
        raise FileNotFoundError(f"Transcript not found: {t_path}")
    transcript  = json.load(open(t_path, encoding="utf-8"))
    all_chunks  = transcript.get("segments", [])

    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source="auto", target="mai")
        test = translator.translate("hello")
        print(f"  deep_translator OK  test='hello'->'{test}'")
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "deep-translator", "-q"], check=False)
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source="auto", target="mai")

    lrl_chunks = []
    seg_lrl    = {}

    for seg in segs:
        idx        = seg["idx"]
        label      = seg["label"]
        seg_chunks = [c for c in all_chunks if c.get("seg_idx") == idx]
        good, skipped = 0, 0
        texts = []

        print(f"  {label}  {len(seg_chunks)} chunks...")

        for chunk in seg_chunks:
            orig = chunk.get("text", "").strip()
            lang = chunk.get("lang", "EN")

            if _is_junk(orig):
                skipped += 1
                lrl_chunks.append({"start": chunk.get("start",0),
                    "end": chunk.get("end",0), "original": orig,
                    "lang": lang, "maithili": "", "seg_idx": idx, "skipped": True})
                continue

            mai = _translate_chunk(orig, translator)
            print(f"    [{lang}] '{orig[:50]}' -> '{mai[:50]}'")
            lrl_chunks.append({"start": chunk.get("start",0),
                "end": chunk.get("end",0), "original": orig,
                "lang": lang, "maithili": mai, "seg_idx": idx, "skipped": False})
            if mai:
                texts.append(mai)
                good += 1

        seg_lrl[label] = " ".join(texts)
        seg["lrl_text"] = seg_lrl[label]
        print(f"  {label}  good={good}  junk={skipped}  chars={len(seg_lrl[label])}")

    full_mai = " ".join(
        c["maithili"] for c in lrl_chunks
        if c.get("maithili","").strip() and not c.get("skipped", False)
    )

    res = {"lrl_chunks": lrl_chunks, "seg_lrl": seg_lrl,
           "full_maithili": full_mai,
           "n_good": sum(1 for c in lrl_chunks if not c.get("skipped")),
           "n_junk": sum(1 for c in lrl_chunks if c.get("skipped"))}

    print(f"\n  Good={res['n_good']}  Junk={res['n_junk']}  "
          f"Total Maithili chars={len(full_mai)}")
    print(f"  Sample: {full_mai[:300]}")

    json.dump(res, open(cache,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    wb_log({"n_good": res["n_good"], "n_junk": res["n_junk"]})
    wb_save(cache); hf_upload(cache, "outputs/lrl_text.json", cfg)
    return res
