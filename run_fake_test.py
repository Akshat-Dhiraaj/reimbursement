"""Throwaway harness: run the validate pipeline over the clean samples and the pytamper fakes, then
report detection recall (fakes flagged) vs false-positives (clean flagged). Uses Groq (key rotation
+ Gemini fallback). Run:  python run_fake_test.py
"""
import sys
import pathlib
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))
from slipguard.llm_validate import load_local_env, validate  # noqa: E402

load_local_env()
_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def run_folder(folder):
    rows = []
    files = sorted(p for p in pathlib.Path(folder).iterdir() if p.suffix.lower() in _EXT) \
        if pathlib.Path(folder).is_dir() else []
    for f in files:
        try:
            d = validate(str(f), provider="groq").get("decision", "?")
        except Exception as e:
            d = f"ERROR:{type(e).__name__}"
        rows.append((f.name, d))
        print(f"  {f.name[:46]:46} -> {d}", flush=True)
    return rows


print("=== CLEAN samples (a genuine receipt should mostly APPROVE) ===", flush=True)
clean = run_folder("samples")
print("\n=== FAKE / tampered receipts (should mostly REVIEW or REJECT) ===", flush=True)
fake = run_folder("fakes/pytamper")


def flagged(rows):
    return sum(1 for _, d in rows if d in ("review", "reject"))


print("\n" + "=" * 64, flush=True)
print("SUMMARY", flush=True)
for name, rows in (("CLEAN", clean), ("FAKES", fake)):
    c = Counter(d for _, d in rows)
    print(f"  {name:6} n={len(rows):3}  approve={c.get('approve', 0):3} "
          f"review={c.get('review', 0):3} reject={c.get('reject', 0):3} "
          f"err={sum(v for k, v in c.items() if k.startswith('ERROR'))}", flush=True)
for t in ("inflated_total", "future_date"):
    sub = [(n, d) for n, d in fake if t in n]
    print(f"  fakes[{t:14}]: {flagged(sub)}/{len(sub)} flagged", flush=True)
print(f"\n  RECALL (fakes caught)      : {flagged(fake)}/{len(fake)}", flush=True)
print(f"  FALSE-POS (clean flagged)  : {flagged(clean)}/{len(clean)}", flush=True)
