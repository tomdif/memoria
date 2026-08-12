"""End-to-end LongMemEval-S QA evaluation for Memoria.

Pipeline: retrieve top-5 sessions -> reader LLM answers -> official
LongMemEval model-judge protocol (per-question-type prompts, abstention
handling, `'yes' in response.lower()` parsing) verbatim from
https://github.com/xiaowu0162/LongMemEval src/evaluation/evaluate_qa.py.

Reader prompt is the official non-CoT retrieval-augmented template from
src/generation/run_generation.py (nl history format, full user+assistant
turns, ranked order, session dates, current date).

Arms:
  memoria         Memoria balanced session retriever (top 5)
  hybrid-ce-dual  commodity control: stage-1 hybrid candidates (0.70 dense +
                  0.30 BM25, pool 60) -> top-15 -> dual-pass cross-encoder,
                  pure CE ordering (top 5) — matches controls_ce.py

Reader: claude-haiku-4-5-20251001 (temperature 0).
Judge:  claude-sonnet-5, thinking disabled, max_tokens 10. NOTE: Sonnet 5
rejects non-default sampling parameters, so temperature cannot be pinned to 0;
the official protocol's temperature=0 is approximated by omitting it.

Stages (resumable):
  python longmemeval_qa.py --stage retrieve [--max N]
  python longmemeval_qa.py --stage qa --arm memoria [--max N]
  python longmemeval_qa.py --stage report
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

BENCH_DIR = Path(__file__).parent
DATA_PATH = BENCH_DIR / "data" / "longmemeval_s_cleaned.json"

READER_MODEL = "claude-haiku-4-5-20251001"
JUDGE_MODEL = "claude-sonnet-5"
TOP_K = 5

# $/MTok (input, output). Sonnet 5 billed at intro pricing through 2026-08-31.
PRICES = {
    READER_MODEL: (1.00, 5.00),
    JUDGE_MODEL: (2.00, 10.00),
}

READER_TEMPLATE = (
    "I will give you several history chats between you and a user. Please "
    "answer the question based on the relevant chat history.\n\n\n"
    "History Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nAnswer:"
)


def api_key() -> str:
    return (Path.home() / ".anthropic_api_key").read_text().strip()


# ---------------------------------------------------------------------------
# Official judge prompts (verbatim from LongMemEval evaluate_qa.py)
# ---------------------------------------------------------------------------

def get_anscheck_prompt(task, question, answer, response, abstention=False):
    if not abstention:
        if task in ['single-session-user', 'single-session-assistant', 'multi-session']:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'temporal-reasoning':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'knowledge-update':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'single-session-preference':
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        else:
            raise NotImplementedError
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
        prompt = template.format(question, answer, response)
    return prompt


# ---------------------------------------------------------------------------
# Shared data helpers
# ---------------------------------------------------------------------------

def load_data(max_n=None):
    data = json.loads(DATA_PATH.read_text())
    return data[:max_n] if max_n else data


def session_maps(q):
    """Dedup'd (ids, user_docs, all_docs) plus sid -> (date, turns)."""
    from longmemeval_final import build_session_documents

    ids, user_docs, all_docs = build_session_documents(q)
    meta = {}
    for sid, date, sess in zip(q["haystack_session_ids"], q["haystack_dates"],
                               q["haystack_sessions"]):
        if sid not in meta:
            meta[sid] = (date, sess)
    return ids, user_docs, all_docs, meta


def evidence_sids(q):
    from longmemeval_final import user_evidence_session_ids
    return user_evidence_session_ids(q)


def retrieval_path(arm):
    return BENCH_DIR / f"qa_retrieval_{arm}.json"


def results_path(arm):
    return BENCH_DIR / f"results_qa_{arm}.jsonl"


# ---------------------------------------------------------------------------
# Stage: retrieve (both arms in one pass; shares embeddings)
# ---------------------------------------------------------------------------

def stage_retrieve(max_n):
    from rank_bm25 import BM25Okapi

    from memoria.embeddings import Embedder
    from memoria.graph import KnowledgeGraph
    from memoria.retriever import EmbeddingCache, Retriever, RetrievalMode, _tokenize
    from memoria.schema import SCHEMA_SQL
    from controls_ce import K_CE, DENSE_POOL, ce_order

    data = load_data(max_n)
    embedder = Embedder()
    db = sqlite3.connect(":memory:")
    db.executescript(SCHEMA_SQL)
    retriever = Retriever(KnowledgeGraph(db), embedder)
    cache = retriever._emb_cache
    reranker = retriever.reranker  # same cross-encoder for the control

    out = {"memoria": {}, "hybrid-ce-dual": {}}
    start = time.time()
    for i, q in enumerate(data):
        qid = q["question_id"]
        ids, user_docs, all_docs, _ = session_maps(q)
        n = len(ids)
        id_to_index = {sid: j for j, sid in enumerate(ids)}

        # memoria balanced
        ranked = retriever.retrieve_sessions(
            q["question"], user_docs, ids, top_k=TOP_K,
            mode=RetrievalMode.BALANCED, all_docs=all_docs,
        )
        out["memoria"][qid] = [sid for sid, _ in ranked][:TOP_K]

        # hybrid-ce-dual control (identical to controls_ce.py)
        corpus = cache.get_batch(user_docs)
        q_emb = embedder.embed_single(q["question"])
        dense_all = np.asarray(corpus @ q_emb, dtype=float)
        pool = np.argsort(dense_all)[::-1][: min(DENSE_POOL, n)]
        dense_scores = {ids[int(j)]: max(float(dense_all[int(j)]), 0.0) for j in pool}
        dense_max = max(dense_scores.values(), default=0.0)
        dense_scale = dense_max if dense_max > 0 else 1.0
        bm25 = BM25Okapi([_tokenize(d) for d in user_docs])
        braw = np.asarray(bm25.get_scores(_tokenize(q["question"])), dtype=float)
        bmax = float(braw.max()) if braw.size else 0.0
        bnorm = braw / bmax if bmax > 0 else np.zeros_like(braw)
        stage1 = {
            sid: dense_scores.get(sid, 0.0) / dense_scale * 0.70 + float(bnorm[j]) * 0.30
            for j, sid in enumerate(ids)
        }
        cand = [sid for sid, _ in
                sorted(stage1.items(), key=lambda kv: kv[1], reverse=True)[:min(K_CE, n)]]
        ranked_ce = ce_order(reranker, q["question"], cand, id_to_index,
                             user_docs, all_docs, dual=True)
        out["hybrid-ce-dual"][qid] = ranked_ce[:TOP_K]

        if (i + 1) % 50 == 0:
            print(f"  retrieve {i+1}/{len(data)} ({(i+1)/(time.time()-start):.2f} q/s)")

    for arm, mapping in out.items():
        retrieval_path(arm).write_text(json.dumps(mapping, indent=1))
        print(f"saved {retrieval_path(arm)} ({len(mapping)} rows)")
    db.close()


# ---------------------------------------------------------------------------
# Stage: qa (reader + judge, async, resumable)
# ---------------------------------------------------------------------------

def build_reader_prompt(q, retrieved_sids, meta):
    parts = []
    for i, sid in enumerate(retrieved_sids):
        date, sess = meta[sid]
        sess_string = ""
        for turn in sess:
            sess_string += "\n\n{}: {}".format(turn["role"], turn["content"].strip())
        parts.append(
            "\n### Session {}:\nSession Date: {}\nSession Content:\n{}\n".format(
                i + 1, date, sess_string))
    history = "".join(parts)
    return READER_TEMPLATE.format(history, q["question_date"], q["question"])


async def qa_one(client, sem, q, retrieved, record_f, lock, totals):
    from retrieval_metrics import recall_all_at_k

    import anthropic

    qid = q["question_id"]
    is_abs = "_abs" in qid
    _, _, _, meta = session_maps(q)
    prompt = build_reader_prompt(q, retrieved, meta)

    async with sem:
        for attempt in range(6):
            try:
                reader_resp = await client.messages.create(
                    model=READER_MODEL, max_tokens=1024, temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                )
                break
            except (anthropic.RateLimitError, anthropic.InternalServerError,
                    anthropic.APIConnectionError):
                await asyncio.sleep(min(2 ** attempt * 2, 60))
        else:
            raise RuntimeError(f"reader failed after retries: {qid}")

        hypothesis = "".join(b.text for b in reader_resp.content if b.type == "text").strip()

        judge_prompt = get_anscheck_prompt(
            q["question_type"], q["question"], q["answer"], hypothesis,
            abstention=is_abs)
        for attempt in range(6):
            try:
                judge_resp = await client.messages.create(
                    model=JUDGE_MODEL, max_tokens=10,
                    thinking={"type": "disabled"},
                    messages=[{"role": "user", "content": judge_prompt}],
                )
                break
            except (anthropic.RateLimitError, anthropic.InternalServerError,
                    anthropic.APIConnectionError):
                await asyncio.sleep(min(2 ** attempt * 2, 60))
        else:
            raise RuntimeError(f"judge failed after retries: {qid}")

        judge_text = "".join(b.text for b in judge_resp.content if b.type == "text").strip()
        label = "yes" in judge_text.lower()

    evid = evidence_sids(q)
    recall5 = recall_all_at_k(retrieved, evid, 5) if evid else None

    record = {
        "question_id": qid,
        "question_type": q["question_type"],
        "is_abstention": is_abs,
        "retrieved": retrieved,
        "recall_all_at_5": recall5,
        "hypothesis": hypothesis,
        "judge_response": judge_text,
        "autoeval_label": label,
        "usage": {
            "reader_in": reader_resp.usage.input_tokens,
            "reader_out": reader_resp.usage.output_tokens,
            "judge_in": judge_resp.usage.input_tokens,
            "judge_out": judge_resp.usage.output_tokens,
        },
    }
    async with lock:
        record_f.write(json.dumps(record) + "\n")
        record_f.flush()
        totals["n"] += 1
        totals["reader_in"] += record["usage"]["reader_in"]
        totals["reader_out"] += record["usage"]["reader_out"]
        totals["judge_in"] += record["usage"]["judge_in"]
        totals["judge_out"] += record["usage"]["judge_out"]
        totals["correct"] += int(label)
        if totals["n"] % 25 == 0:
            print(f"  {totals['n']} done  acc so far={totals['correct']/totals['n']:.3f}  "
                  f"cost so far=${run_cost(totals):.2f}")


def run_cost(totals):
    r_in, r_out = PRICES[READER_MODEL]
    j_in, j_out = PRICES[JUDGE_MODEL]
    return (totals["reader_in"] * r_in + totals["reader_out"] * r_out
            + totals["judge_in"] * j_in + totals["judge_out"] * j_out) / 1e6


async def stage_qa(arm, max_n, concurrency):
    from anthropic import AsyncAnthropic

    data = load_data(max_n)
    retrieval = json.loads(retrieval_path(arm).read_text())

    done = set()
    rp = results_path(arm)
    if rp.exists():
        for line in rp.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["question_id"])
    todo = [q for q in data if q["question_id"] in retrieval
            and q["question_id"] not in done]
    print(f"arm={arm}: {len(done)} already done, {len(todo)} to run")
    if not todo:
        return

    client = AsyncAnthropic(api_key=api_key(), max_retries=2)
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    totals = defaultdict(int)
    start = time.time()
    with open(rp, "a") as record_f:
        await asyncio.gather(*[
            qa_one(client, sem, q, retrieval[q["question_id"]], record_f, lock, totals)
            for q in todo
        ])
    print(f"arm={arm}: ran {totals['n']} rows in {time.time()-start:.0f}s, "
          f"cost this run ${run_cost(totals):.2f}")
    print(f"  tokens: reader {totals['reader_in']}/{totals['reader_out']}, "
          f"judge {totals['judge_in']}/{totals['judge_out']}")


# ---------------------------------------------------------------------------
# Stage: report
# ---------------------------------------------------------------------------

def stage_report(arms):
    for arm in arms:
        rp = results_path(arm)
        if not rp.exists():
            print(f"[{arm}] no results file")
            continue
        rows = [json.loads(l) for l in rp.read_text().splitlines() if l.strip()]
        n = len(rows)
        overall = np.mean([r["autoeval_label"] for r in rows])
        abst = [r for r in rows if r["is_abstention"]]
        nonabst = [r for r in rows if not r["is_abstention"]]
        print(f"\n=== {arm} (n={n}) ===")
        print(f"overall QA accuracy:        {overall*100:.1f}%")
        if nonabst:
            print(f"non-abstention accuracy:    "
                  f"{np.mean([r['autoeval_label'] for r in nonabst])*100:.1f}% (n={len(nonabst)})")
        if abst:
            print(f"abstention accuracy:        "
                  f"{np.mean([r['autoeval_label'] for r in abst])*100:.1f}% (n={len(abst)})")
        by_type = defaultdict(list)
        for r in nonabst:
            by_type[r["question_type"]].append(r["autoeval_label"])
        print("by ability (non-abstention):")
        for t in sorted(by_type):
            print(f"  {t:30s} {np.mean(by_type[t])*100:5.1f}% (n={len(by_type[t])})")
        # retrieval vs QA gap (rows with evidence labels)
        with_ret = [r for r in nonabst if r["recall_all_at_5"] is not None]
        hit = [r for r in with_ret if r["recall_all_at_5"] == 1.0]
        miss = [r for r in with_ret if r["recall_all_at_5"] == 0.0]
        if hit:
            print(f"retrieval-hit rows (R-all@5=1): acc "
                  f"{np.mean([r['autoeval_label'] for r in hit])*100:.1f}% (n={len(hit)})")
        if miss:
            print(f"retrieval-miss rows:            acc "
                  f"{np.mean([r['autoeval_label'] for r in miss])*100:.1f}% (n={len(miss)})")
        tot = defaultdict(int)
        for r in rows:
            for k, v in r["usage"].items():
                tot[k] += v
        tot2 = {"reader_in": tot["reader_in"], "reader_out": tot["reader_out"],
                "judge_in": tot["judge_in"], "judge_out": tot["judge_out"]}
        print(f"tokens: {dict(tot2)}  cost ${run_cost(tot2):.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["retrieve", "qa", "report"], required=True)
    parser.add_argument("--arm", choices=["memoria", "hybrid-ce-dual"])
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=12)
    args = parser.parse_args()

    if args.stage == "retrieve":
        stage_retrieve(args.max)
    elif args.stage == "qa":
        if not args.arm:
            parser.error("--arm required for qa stage")
        asyncio.run(stage_qa(args.arm, args.max, args.concurrency))
    else:
        stage_report(["memoria", "hybrid-ce-dual"])


if __name__ == "__main__":
    main()
