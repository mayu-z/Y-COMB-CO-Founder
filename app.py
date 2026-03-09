"""
YC Co-Founder — Streamlit App
Phase 5: Three-tab interface for the YC knowledge RAG pipeline.
"""

import sys
import os
import json
import re

sys.path.insert(0, "src")

import streamlit as st

# ── Page config (must be first st call) ────────────────
st.set_page_config(
    page_title="YC Co-Founder",
    page_icon="🚀",
    layout="wide",
)

# ── Lazy-load heavy objects via session state ──────────

@st.cache_resource(show_spinner="Loading YC knowledge base…")
def load_advisor():
    from rag import YCAdvisor
    return YCAdvisor()


@st.cache_resource(show_spinner="Loading startup evaluator…")
def load_evaluator():
    from evaluator import StartupEvaluator
    return StartupEvaluator()


@st.cache_data(show_spinner=False)
def load_companies():
    """Parse all company chunks into structured rows."""
    with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    rows = []
    for c in chunks:
        if c.get("source_type") != "company":
            continue
        text = c.get("text", "")
        row = {"Name": c.get("title", "")}

        m = re.search(r"from batch (\S+)", text)
        row["Batch"] = m.group(1) if m else ""

        m = re.search(r"currently marked as (\w+)", text)
        row["Status"] = m.group(1) if m else ""

        m = re.search(r"operates in (.+?) and is associated", text)
        row["Industry"] = m.group(1) if m else ""

        m = re.search(r"one-line company description is: (.+?)\.", text)
        row["Description"] = m.group(1) if m else ""

        rows.append(row)
    return rows


# ── Sidebar ────────────────────────────────────────────

with st.sidebar:
    st.title("🚀 YC Co-Founder")
    st.caption(
        "Built on real YC knowledge — Paul Graham essays, "
        "YC partner posts, Startup School lectures, and "
        "1,494 YC-backed companies."
    )
    st.divider()
    st.markdown(
        "📚 **2,804** knowledge chunks\n\n"
        "🏢 **1,494** YC companies indexed\n\n"
        "✍️ **325** Paul Graham essay chunks\n\n"
        "🎓 **123** Startup School chunks"
    )


# ── Tabs ───────────────────────────────────────────────

tab_ask, tab_eval, tab_browse = st.tabs([
    "💬 Ask YC",
    "🔍 Evaluate My Startup",
    "🏢 Browse YC Companies",
])


# ════════════════════════════════════════════════════════
#  TAB 1 — Ask YC
# ════════════════════════════════════════════════════════

with tab_ask:
    st.header("💬 Ask YC")

    # Session state
    if "ask_query" not in st.session_state:
        st.session_state.ask_query = ""
    if "ask_result" not in st.session_state:
        st.session_state.ask_result = None

    def _run_ask(query: str):
        st.session_state.ask_query = query
        st.session_state.ask_result = None  # clear stale result to trigger rerun

    # Input
    query = st.text_input(
        "Ask anything about startups or YC…",
        value=st.session_state.ask_query,
        key="ask_input",
        label_visibility="collapsed",
        placeholder="Ask anything about startups or YC…",
    )

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        ask_clicked = st.button("Ask", type="primary", use_container_width=True)

    # Example questions
    st.markdown("**Try these:**")
    ex_cols = st.columns(4)
    examples = [
        "How do I get into YC?",
        "What is product market fit?",
        "When should I raise funding?",
        "How to talk to users?",
    ]
    for col, ex in zip(ex_cols, examples):
        with col:
            if st.button(ex, key=f"ex_{ex}"):
                _run_ask(ex)
                st.rerun()

    # Process
    should_run = ask_clicked and query.strip()
    if st.session_state.ask_query and st.session_state.ask_result is None:
        should_run = True
        query = st.session_state.ask_query

    if should_run:
        st.session_state.ask_query = query.strip()
        with st.spinner("Searching YC knowledge base…"):
            try:
                advisor = load_advisor()
                result = advisor.ask_with_sources(query.strip())
                st.session_state.ask_result = result
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.session_state.ask_result = None

    # Display
    result = st.session_state.ask_result
    if result:
        st.markdown("---")
        st.markdown(result["answer"])

        if result["sources"]:
            st.markdown("**Sources:**")
            tags = ""
            for s in result["sources"]:
                author = f" — {s['author']}" if s["author"] else ""
                tags += (
                    f'<span style="display:inline-block;background:#f0f2f6;'
                    f'border-radius:4px;padding:2px 8px;margin:2px;'
                    f'font-size:0.85em;">'
                    f'[{s["source_type"]}] {s["title"]}{author}</span> '
                )
            st.markdown(tags, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  TAB 2 — Evaluate My Startup
# ════════════════════════════════════════════════════════

with tab_eval:
    st.header("🔍 Evaluate My Startup")
    st.caption("Get a YC partner-style assessment of your idea")

    if "eval_result" not in st.session_state:
        st.session_state.eval_result = None

    left, right = st.columns([1, 1.4], gap="large")

    with left:
        desc = st.text_area(
            "Describe your startup in one sentence",
            height=80,
            placeholder="AI tool that automates legal contract review for small law firms",
        )
        industry = st.selectbox(
            "Industry",
            ["Fintech", "SaaS", "Healthcare", "EdTech", "Consumer",
             "Crypto", "DevTools", "Marketplace", "AI/ML", "Other"],
        )
        target = st.radio("Target Customer", ["B2B", "B2C"], horizontal=True)
        stage = st.selectbox(
            "Stage",
            ["Idea", "Prototype", "Live", "Revenue"],
        )
        team_size = st.number_input("Team Size", min_value=1, max_value=10, value=2)
        background = st.text_input(
            "Founder Background",
            placeholder="Brief background — ex-Google, Stanford CS, domain expert etc.",
        )

        eval_clicked = st.button("Evaluate My Startup 🚀", type="primary")

    if eval_clicked:
        if not desc.strip():
            st.warning("Please describe your startup first.")
        else:
            with right:
                with st.spinner("Analyzing against 1,494 YC companies…"):
                    try:
                        evaluator = load_evaluator()
                        result = evaluator.evaluate(
                            description=desc.strip(),
                            industry=industry.lower(),
                            target_customer=target,
                            stage=stage.lower(),
                            team_size=team_size,
                            team_background=background or "not specified",
                        )
                        st.session_state.eval_result = result
                    except Exception as e:
                        st.error(f"Something went wrong: {e}")
                        st.session_state.eval_result = None

    with right:
        result = st.session_state.eval_result
        if result:
            st.markdown("### Assessment")
            st.markdown(result["assessment"])

            st.markdown("---")
            st.markdown("### Similar YC Companies")
            for comp in result["similar_companies"]:
                sim = comp.get("similarity")
                sim_str = f" · {sim:.0%} match" if sim is not None else ""
                st.markdown(
                    f"**{comp['name']}** — {comp['industry']}  \n"
                    f"Batch {comp['batch']} · {comp['status']}{sim_str}  \n"
                    f"_{comp['description']}_"
                )

            st.markdown("---")
            st.caption(result["disclaimer"])


# ════════════════════════════════════════════════════════
#  TAB 3 — Browse YC Companies
# ════════════════════════════════════════════════════════

with tab_browse:
    st.header("🏢 Browse YC Companies")

    companies = load_companies()

    # Build filter options
    all_industries = sorted({r["Industry"] for r in companies if r["Industry"]})
    all_batches = sorted(
        {r["Batch"] for r in companies if r["Batch"]},
        reverse=True,
    )
    all_statuses = sorted({r["Status"] for r in companies if r["Status"]})

    # Filter controls
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        sel_industry = st.selectbox(
            "Industry", ["All"] + all_industries, key="browse_ind"
        )
    with f2:
        sel_batch = st.selectbox(
            "Batch", ["All"] + all_batches, key="browse_batch"
        )
    with f3:
        status_opts = ["All"] + all_statuses
        sel_status = st.selectbox("Status", status_opts, key="browse_status")
    with f4:
        search_text = st.text_input(
            "Search company name…", key="browse_search",
            label_visibility="collapsed",
            placeholder="Search company name…",
        )

    # Apply filters
    filtered = companies
    if sel_industry != "All":
        filtered = [r for r in filtered if r["Industry"] == sel_industry]
    if sel_batch != "All":
        filtered = [r for r in filtered if r["Batch"] == sel_batch]
    if sel_status != "All":
        filtered = [r for r in filtered if r["Status"] == sel_status]
    if search_text.strip():
        q = search_text.strip().lower()
        filtered = [r for r in filtered if q in r["Name"].lower()]

    # Sort newest batch first
    batch_order = {b: i for i, b in enumerate(all_batches)}
    filtered.sort(key=lambda r: batch_order.get(r["Batch"], 999))

    st.markdown(f"**Showing {len(filtered)} companies**")

    if filtered:
        st.dataframe(
            filtered,
            width="stretch",
            hide_index=True,
            column_config={
                "Name": st.column_config.TextColumn(width="medium"),
                "Industry": st.column_config.TextColumn(width="medium"),
                "Batch": st.column_config.TextColumn(width="small"),
                "Status": st.column_config.TextColumn(width="small"),
                "Description": st.column_config.TextColumn(width="large"),
            },
        )
    else:
        st.info("No companies match your filters.")
