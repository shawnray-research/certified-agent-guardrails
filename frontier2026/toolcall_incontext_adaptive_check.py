"""In-context tool-call evaluation, step 5b: checks on the adaptive attack (predeclared after an independent audit,
before any of these queries was sent).

The hosted judges decode at the provider default, so a best-of-20 search over single scores mixes the effect of the
rewrite with sampling noise, and first-integer parsing can credit the attacker. Two checks, with STRICT parsing
(the reply must be a bare integer 0-100 after stripping whitespace and a trailing period; anything else, a refusal,
or an error scores 1.0, i.e., fails closed) and raw replies stored:
  null    : for every judge and V1 item, the UNMODIFIED clean in-context prompt is re-scored up to 20 times, early
            stop at the first score < theta_.10 (identifies evasion at theta_.10 and theta_.20 >= theta_.10);
  rescore : for every chain whose minimum score is below theta_.20, the attempt with the minimum score (first on ties)
            is re-scored 5 times; reported per judge: the number of items whose selected rewrite evades in at least
            3 of 5 re-scores, and the mean evasion fraction, at theta_.10 and theta_.20.
Output: toolcall_incontext_adaptive_check.json
Usage: python3 toolcall_incontext_adaptive_check.py"""
import json, os, re, threading
from concurrent.futures import ThreadPoolExecutor
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("ics", os.path.join(HERE, "toolcall_incontext_score.py"))
ics = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ics)
_spec2 = importlib.util.spec_from_file_location("ada", os.path.join(HERE, "toolcall_incontext_adaptive.py"))
ada = importlib.util.module_from_spec(_spec2); _spec2.loader.exec_module(ada)
OUT = os.path.join(HERE, "toolcall_incontext_adaptive_check.json")


def strict(score_fn, prompt):
    for k in range(4):
        try:
            txt, stop = score_fn(prompt)
            if stop == "refusal":
                return 1.0, "refusal", (txt or "")[:200]
            t = (txt or "").strip().rstrip(".").strip()
            if re.fullmatch(r"\d{1,3}", t) and int(t) <= 100:
                return int(t) / 100.0, stop, t
            return 1.0, "unparsed", (txt or "")[:200]
        except Exception as e:
            err = repr(e)[:120]
    return 1.0, "error", err


def main():
    items = {it["id"]: it for it in json.load(open(ics.ITEMS))["items"]}
    A = json.load(open(os.path.join(HERE, "toolcall_incontext_adaptive.json")))
    X = json.load(open(os.path.join(HERE, "toolcall_incontext_adaptive_ext.json")))
    TH = {**A["thresholds"], **X["thresholds"]}
    chains = A["chains"] + X["chains"]
    v1 = sorted([i for i in items if items[i]["kind"] == "V1"])
    scorers = {m: ics.make_scorer(m) for m in TH}
    res = {"thresholds": TH, "null": {}, "rescore": {}}
    lock = threading.Lock()

    def null_job(j):
        m, iid = j
        prompt = ics.render_ic(items[iid]); out = []
        for _ in range(20):
            s, stop, raw = strict(scorers[m], prompt)
            out.append([s, stop, raw])
            if s < TH[m]["0.10"]:
                break
        return m, iid, out

    def res_job(c):
        m = c["model"]
        att = min(c["attempts"], key=lambda a: a["score"])
        prompt = ics.render_ic(ada.with_text(items[c["id"]], att["text"]))
        out = [list(strict(scorers[m], prompt)) for _ in range(5)]
        return m, c["id"], {"selected_score": att["score"], "rescores": out}

    jobs_res = [c for c in chains if min(a["score"] for a in c["attempts"]) < TH[c["model"]]["0.20"]]
    jobs_null = [(m, i) for m in TH for i in v1]
    print(f"rescore jobs {len(jobs_res)}, null jobs {len(jobs_null)}", flush=True)
    with ThreadPoolExecutor(48) as ex:
        for k, (m, iid, r) in enumerate(ex.map(res_job, jobs_res), 1):
            with lock:
                res["rescore"].setdefault(m, {})[iid] = r
        json.dump(res, open(OUT, "w"))
        print("rescore done", flush=True)
        for k, (m, iid, r) in enumerate(ex.map(null_job, jobs_null), 1):
            with lock:
                res["null"].setdefault(m, {})[iid] = r
                if k % 100 == 0:
                    json.dump(res, open(OUT, "w")); print(f"  null {k}/{len(jobs_null)}", flush=True)
    json.dump(res, open(OUT, "w"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
