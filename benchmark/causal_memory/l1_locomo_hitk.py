#!/usr/bin/env python3
# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""L1 zero-LLM evidence-retrieval evaluation on LoCoMo (causal memory stack).

Per conversation: ingest every chat message as an AO atom (one session per
conversation), then answer-side retrieval is evaluated WITHOUT any LLM:

  P (production matcher)  keywords extracted from each question are queried
     through HybridRetrievalFacade (case-insensitive substring filter, facts
     profile 0/0/12); the union of returned cards is the ranked list in
     first-seen order. This measures what the stack returns today.
  T (token-overlap reference)  a pure-Python scorer over the same AO atoms
     (token-set overlap, idf-weighted). This quantifies what the rerank
     injection point should recover next.

Metric: any-hit@{5,10} -- a question counts as hit when any gold evidence
dia_id appears in the top-k retrieved messages; plus overall union recall and
zero-hit share. Per LoCoMo category breakdown is reported.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import tempfile
from collections import Counter
from pathlib import Path

from openviking.session.ao_ledger import AOLedger
from openviking.session.influence_projection import ProjectionRegistry
from openviking.session.retrieval_facade import (
    ForkBranchAdapter,
    HybridRetrievalFacade,
    InfluenceProjectionAdapter,
    SegmentAtomAdapter,
)

LABELS = ["locomo"]
FACTS_PROFILE = {"fork_branch": 0, "projection": 0, "segment_atom": 12}
DEFAULT_DATA = "/home/zhoujie22/river2_0/evol_bench/LoCoMo/data/locomo/locomo10.json"

_STOP = set(
    "the a an to of in on at for with and or is are was were be been this that these those you your "
    "we they it its his her their i not no do does did can could would should must may might will "
    "when where why how what which who whom whose using use used needs need provide provides ensure "
    "from by as into out over under then than them there here all any each every some such only also "
    "did didn't don't about after before during between more most other another same first last next "
    "tell say says said tell me please know like just also very really much many few little".split()
)


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[0-9a-z]+", text.lower().replace("’", "'"))
    return [w for w in words if len(w) > 2 and w not in _STOP]


def _question_keywords(question: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for w in _tokens(question):
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def ingest_conversation(conv: dict, work: Path) -> tuple[str, list[dict]]:
    """One session per conversation; one AO atom per message."""
    session_id = f"locomo-{conv['sample_id']}"
    ledger = AOLedger(work / session_id, session_id)
    atoms: list[dict] = []
    sessions = sorted(
        ((k, v) for k, v in conv["conversation"].items() if re.fullmatch(r"session_\d+", k)),
        key=lambda kv: int(kv[0].split("_")[1]),
    )
    for sname, msgs in sessions:
        for m in msgs:
            rec = ledger.append(
                action={
                    "record_kind": "atom",
                    "acl_labels": LABELS,
                    "speaker": m["speaker"],
                    "text": m["text"],
                    "dia_id": m["dia_id"],
                    "locomo_session": sname,
                },
                observation={"synopsis": m["text"]},
                message_ref={"message_id": m["dia_id"]},
            )
            atoms.append({"ao_id": rec.ao_id, "dia_id": m["dia_id"], "text": m["text"], "tokens": _tokens(m["text"])})
    return session_id, atoms


def build_facade(work: Path, session_id: str) -> HybridRetrievalFacade:
    return HybridRetrievalFacade(
        fork_branch=ForkBranchAdapter(work),
        projection=InfluenceProjectionAdapter(ProjectionRegistry(work / "unused.jsonl")),
        segment_atom=SegmentAtomAdapter(work / session_id, session_id),
    )


def production_ranked_dias(facade: HybridRetrievalFacade, keywords: list[str], ao_to_dia: dict[str, str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for kw in keywords:
        res = facade.retrieve(kw, principal_labels=LABELS, profile=FACTS_PROFILE)
        for card in res.cards:
            dia = ao_to_dia.get((card.source_pointer or {}).get("ao_id"), "")
            if dia and dia not in seen:
                seen.add(dia)
                ordered.append(dia)
    return ordered


def token_ranked_dias(atoms: list[dict], q_tokens: list[str]) -> list[str]:
    if not q_tokens:
        return []
    df: Counter = Counter()
    for atom in atoms:
        df.update(set(atom["tokens"]))
    n = len(atoms)
    q_set = set(q_tokens)
    scored = []
    for atom in atoms:
        overlap = q_set & set(atom["tokens"])
        if not overlap:
            continue
        score = sum(math.log(n / df[t]) for t in overlap)
        scored.append((score, atom["sequence"] if "sequence" in atom else 0, atom["dia_id"]))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [dia for _, _, dia in scored]


def any_hit(ranked: list[str], gold: set[str], k: int) -> bool:
    return bool(gold & set(ranked[:k]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--out", default=None, help="write per-question JSON here")
    parser.add_argument("--limit-convs", type=int, default=0, help="0 = all")
    args = parser.parse_args()

    conversations = json.load(open(args.data, encoding="utf-8"))
    if args.limit_convs:
        conversations = conversations[: args.limit_convs]

    per_question = []
    overall = {"P": Counter(), "T": Counter()}
    total_q = 0
    for conv in conversations:
        work = Path(tempfile.mkdtemp(prefix=f"l1-{conv['sample_id']}-"))
        session_id, atoms = ingest_conversation(conv, work)
        facade = build_facade(work, session_id)
        ao_to_dia = {a["ao_id"]: a["dia_id"] for a in atoms}
        for q in conv["qa"]:
            gold = set(q.get("evidence") or [])
            if not gold:
                continue
            total_q += 1
            kws = _question_keywords(q["question"])
            p_ranked = production_ranked_dias(facade, kws, ao_to_dia)
            t_ranked = token_ranked_dias(atoms, _tokens(q["question"]))
            row = {
                "sample_id": conv["sample_id"],
                "category": q.get("category"),
                "question": q["question"],
                "gold": sorted(gold),
                "p_union_size": len(p_ranked),
                "P_hit5": any_hit(p_ranked, gold, 5),
                "P_hit10": any_hit(p_ranked, gold, 10),
                "T_hit5": any_hit(t_ranked, gold, 5),
                "T_hit10": any_hit(t_ranked, gold, 10),
            }
            per_question.append(row)
            for m in ("P", "T"):
                overall[m]["hit5"] += row[f"{m}_hit5"]
                overall[m]["hit10"] += row[f"{m}_hit10"]
        print(
            f"[{conv['sample_id']}] atoms={len(atoms)} qa_with_evidence="
            f"{sum(1 for r in per_question if r['sample_id'] == conv['sample_id'])}"
        )

    zero = sum(1 for r in per_question if r["p_union_size"] == 0)
    print("\n=== overall (zero-LLM evidence any-hit) ===")
    print(f"questions={total_q}  zero-union(P)={zero} ({zero / max(total_q, 1):.1%})")
    for m in ("P", "T"):
        print(
            f"{m}: hit@5={overall[m]['hit5'] / total_q:.1%}  hit@10={overall[m]['hit10'] / total_q:.1%}"
        )
    print("\n=== per category (P hit@10 / T hit@10) ===")
    by_cat: dict = {}
    for r in per_question:
        c = by_cat.setdefault(r["category"], Counter())
        c["n"] += 1
        c["p10"] += r["P_hit10"]
        c["t10"] += r["T_hit10"]
    for cat in sorted(by_cat):
        c = by_cat[cat]
        print(f"cat {cat}: n={c['n']}  P@10={c['p10'] / c['n']:.1%}  T@10={c['t10'] / c['n']:.1%}")

    if args.out:
        Path(args.out).write_text(json.dumps(per_question, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nper-question rows -> {args.out}")


if __name__ == "__main__":
    main()
