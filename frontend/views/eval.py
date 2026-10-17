"""Eval page."""

from __future__ import annotations

import html

import streamlit as st

from reminisce.settings import EVAL_PATH

def _page_eval() -> None:
    st.markdown('<p class="greeting">Eval</p>', unsafe_allow_html=True)
    if not EVAL_PATH.exists():
        st.info("No eval yet.")
        return
    import json

    report = json.loads(EVAL_PATH.read_text())
    n = report["n"]
    warm = report["prompts"]["warm"]
    strict = report["prompts"]["strict"]
    one, two, three, four = st.columns(4)
    one.metric("Embedding", f"{report['embedding_only_at_1']}/{n}")
    two.metric("Hybrid", f"{report['hybrid_at_1']}/{n}")
    three.metric("Grounded", f"{warm['served_grounded']}/{n}")
    four.metric("Warm prompt", f"{warm['raw_selection']}/{n}")
    st.markdown(
        "<p class='quiet-note'>"
        f"Cosmos descriptions {report['cosmos_descriptions']}/{n}. "
        f"Strict prompt {strict['raw_selection']}/{n}. "
        f"Warm latency {warm['mean_latency_s']}s. "
        f"W&amp;B: {html.escape(str(report.get('wandb') or 'not logged'))}."
        "</p>",
        unsafe_allow_html=True,
    )
    rows = []
    for row in warm["rows"]:
        rows.append(
            {
                "question": row["question"],
                "expected": row["expected"],
                "embedding": row["embedding_top"],
                "hybrid": row["hybrid_top"],
                "served": "yes" if row["served_ok"] else "no",
                "grounded": "yes" if row["grounded"] else "no",
                "latency_s": row["latency_s"],
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)

