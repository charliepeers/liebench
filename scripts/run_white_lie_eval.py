"""Run the white-lie dataset against models on OpenRouter and score the replies.

For every model, every prompt in the dataset gets a reply, and a judge model grades
every reply (neutral and pressure alike) into `label`: AFFIRMS / CANDID / EVASIVE.

Replies are cached in results/cache.jsonl, so a rerun only makes missing calls.
A call that still fails after retries is recorded as ERROR (and not cached), so
rerunning fills it in.

    export OPENROUTER_API_KEY=...
    python scripts/run_white_lie_eval.py                 # default models
    python scripts/run_white_lie_eval.py --limit 2       # first 2 scenarios only
    python scripts/run_white_lie_eval.py --models openai/gpt-6.1-terra
    python scripts/run_white_lie_eval.py --dataset data/white_lie_dataset_split.jsonl
    ->  results/<dataset name>_results.csv
"""
import argparse
import csv
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_white_lie_dataset import JUDGE_PROMPT  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "data" / "white_lie_dataset.jsonl"
RESULTS_DIR = ROOT / "results"
CACHE = RESULTS_DIR / "cache.jsonl"

DEFAULT_MODELS = [
    "openai/gpt-6.1-terra",
    "google/gemini-4-argon",
    "anthropic/claude-opus-5.5",
]
DEFAULT_JUDGE = "anthropic/claude-sonnet-5.5"

_cache_lock = threading.Lock()


def load_cache():
    cache = {}
    if CACHE.exists():
        for line in CACHE.open():
            entry = json.loads(line)
            cache[entry["key"]] = entry["text"]
    return cache


def call(model, messages, cache, max_tokens=1000):
    """messages: a chat message list, or a plain string for a single user turn."""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    # single user turns are keyed by their text, matching older cache entries
    single = len(messages) == 1 and messages[0]["role"] == "user"
    key = json.dumps([model, messages[0]["content"] if single else messages])
    if key in cache:
        return cache[key]
    body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={
            "Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"],
            "Content-Type": "application/json",
        },
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.load(resp)
            if "choices" not in data:
                raise RuntimeError(data.get("error", data))
            text = data["choices"][0]["message"]["content"] or ""
            break
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}: {e.read()[:300]!r}"
        except (urllib.error.URLError, RuntimeError, TimeoutError) as e:
            err = str(e)[:300]
        print(f"  retry {model}: {err}", file=sys.stderr)
        if attempt == 4:
            return "ERROR: " + err
        time.sleep(2 ** attempt)
    with _cache_lock:
        cache[key] = text
        with CACHE.open("a") as f:
            f.write(json.dumps({"key": key, "text": text}) + "\n")
    return text


def last_label(text, options):
    """The judge sometimes explains first, so take the last label it names."""
    found = re.findall(r"\b(" + "|".join(options) + r")\b", text)
    return found[-1] if found else "UNPARSED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--judge", default=DEFAULT_JUDGE)
    ap.add_argument("--limit", type=int, help="only the first N scenarios")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    if "OPENROUTER_API_KEY" not in os.environ:
        sys.exit("Set OPENROUTER_API_KEY first.")

    rows = [json.loads(line) for line in args.dataset.open()]
    if args.limit:
        keep = list(dict.fromkeys(r["item_id"] for r in rows))[: args.limit]
        rows = [r for r in rows if r["item_id"] in keep]
    RESULTS_DIR.mkdir(exist_ok=True)
    cache = load_cache()

    with ThreadPoolExecutor(args.workers) as pool:
        # 1. model replies
        jobs = [(m, r) for m in args.models for r in rows]
        print(f"{len(jobs)} replies...")
        replies = list(pool.map(lambda j: call(j[0], j[1]["messages"], cache), jobs))

        # 2. judge grades for every reply
        def label(job_reply):
            (_, r), reply = job_reply
            if reply.startswith("ERROR: "):
                return "ERROR"
            content = JUDGE_PROMPT.format(description=r["description"], prompt=r["prompt"], response=reply)
            verdict = call(args.judge, content, cache)
            if verdict.startswith("ERROR: "):
                return "ERROR"
            return last_label(verdict, ["AFFIRMS", "CANDID", "EVASIVE"])

        print("judging...")
        labels = list(pool.map(label, zip(jobs, replies)))

    out = RESULTS_DIR / f"{args.dataset.stem}_results.csv"
    fields = ["model", "id", "item_id", "category", "quality", "framing", "stakes",
              "system", "prompt", "response", "label"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (model, r), reply, lab in zip(jobs, replies, labels):
            w.writerow({
                "model": model, "id": r["id"], "item_id": r["item_id"], "category": r["category"],
                "quality": r["quality"], "framing": r["framing"], "stakes": r["stakes"],
                "system": r.get("system", ""), "prompt": r["prompt"], "response": reply, "label": lab,
            })
    errors = labels.count("ERROR")
    print(f"wrote {out}" + (f"  ({errors} errors; rerun to retry them)" if errors else ""))


if __name__ == "__main__":
    main()
