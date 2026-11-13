"""A10: code/config fingerprint at test-exposure time. Detects post-test-read changes (test-set tuning risk).
python -m evaluation.fingerprint "<note>"  -> appends to reports/a10/test_exposure.jsonl
python -m evaluation.fingerprint --diff     -> files changed since the FIRST fingerprint
"""
import hashlib, json, os, sys, time, glob
from evaluation.audit import ROOT, POLICY_DIRS
LOG = os.path.join(ROOT, "reports", "a10", "test_exposure.jsonl")
DIRS = POLICY_DIRS + ["src/env", "runner", "configs"]

def fp():
    out = {}
    for d in DIRS:
        for p in sorted(glob.glob(os.path.join(ROOT, d, "**", "*.py"), recursive=True)) + \
                 sorted(glob.glob(os.path.join(ROOT, d, "**", "*.json"), recursive=True)):
            out[os.path.relpath(p, ROOT)] = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    return out

def test_runs():
    """run dirs/files whose name mentions test (exposure evidence) with mtimes."""
    hits = {}
    for p in glob.glob(os.path.join(ROOT, "runs", "*", "*test*")) + glob.glob(os.path.join(ROOT, "runs", "*", "*", "*test*")):
        hits[os.path.relpath(p, ROOT)] = time.strftime("%F %T", time.localtime(os.path.getmtime(p)))
    return hits

if __name__ == "__main__":
    if sys.argv[1:] == ["--diff"]:
        rows = [json.loads(l) for l in open(LOG)]
        first, now = rows[0]["files"], fp()
        ch = {f: "changed" for f in now if f in first and now[f] != first[f]}
        ch.update({f: "new" for f in now if f not in first})
        print(json.dumps({"since": rows[0]["time"], "changes": ch}, indent=1))
    else:
        rec = {"time": time.strftime("%F %T%z"), "note": " ".join(sys.argv[1:]), "files": fp(), "test_runs": test_runs()}
        open(LOG, "a").write(json.dumps(rec) + "\n")
        print(rec["time"], len(rec["files"]), "files;", rec["test_runs"])
