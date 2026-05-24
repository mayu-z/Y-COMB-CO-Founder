"""
YC Co-Founder — Streamlit App
Phase 5: Three-tab interface for the YC knowledge RAG pipeline.
"""

import sys
import os
import json
import re
import html

sys.path.insert(0, "src")

import streamlit as st

from benchmark import (
    QUESTIONS_PATH,
    RESULTS_PATH,
    evaluate_question,
    load_questions,
    run_benchmark,
)

# ── Page config (must be first st call) ────────────────
st.set_page_config(
    page_title="YC Co-Founder",
    page_icon="YC",
    layout="wide",
)

st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400;1,700&family=Inter:wght@300;400;500&display=swap" rel="stylesheet">
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {
            visibility: visible !important;
            height: auto !important;
            min-height: 50px !important;
            overflow: visible !important;
        }
        [data-testid="stHeader"] {
            background: transparent !important;
            border: 0;
            height: auto !important;
            min-height: 50px !important;
            overflow: visible !important;
            pointer-events: none;
        }
        [data-testid="stToolbar"] {display: none;}

        [data-testid="collapsedControl"],
        .st-emotion-cache-vkkuhw,
        .st-emotion-cache-13veyas,
        .e9ic3ti10 {
            display: none !important;
            visibility: hidden !important;
            width: 0 !important;
            height: 0 !important;
            overflow: hidden !important;
            pointer-events: none !important;
        }

        [data-testid="collapsedControl"] svg {
            fill: #1a1a1a !important;
            color: #1a1a1a !important;
        }

        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {
            background: #F5F0E8;
            color: #1a1a1a;
        }

        body, p, li, div, span, label, input, textarea {
            font-family: "Inter", sans-serif;
            color: #1a1a1a;
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: "Playfair Display", serif;
            color: #1a1a1a;
            letter-spacing: 0.2px;
        }

        [data-testid="stSidebar"] {
            background: #EDE8DC;
            border-right: 1px solid #e0dad0;
            transition: margin-left 0.3s ease, visibility 0.3s ease;
        }

        [data-testid="stSidebar"] * {
            color: #1a1a1a;
        }

        [data-baseweb="tab-list"] {
            gap: 20px;
            border-bottom: 1px solid #d8d1c6;
        }

        [data-baseweb="tab"] {
            font-family: "Playfair Display", serif;
            font-size: 1rem;
            color: #4b4b4b;
            padding-left: 0;
            padding-right: 0;
            background: transparent;
        }

        [aria-selected="true"] {
            color: #1a1a1a !important;
            border-bottom: 2px solid #1a1a1a !important;
        }

        .stButton > button,
        .stDownloadButton > button {
            font-family: "Inter", sans-serif;
            font-weight: 500;
            border-radius: 4px;
            border: 1px solid #1a1a1a;
            transition: all 0.2s ease;
            box-shadow: none;
            opacity: 1 !important;
        }

        .stButton > button[kind="primary"],
        .stButton > button[data-testid="stBaseButton-primary"] {
            background: #1a1a1a;
            color: #ffffff !important;
            border: 1px solid #1a1a1a;
        }

        .stButton > button[kind="primary"] *,
        .stButton > button[data-testid="stBaseButton-primary"] * {
            color: #ffffff !important;
            fill: #ffffff !important;
        }

        .stButton > button[kind="primary"]:hover,
        .stButton > button[data-testid="stBaseButton-primary"]:hover {
            background: #FF6B35;
            border-color: #FF6B35;
            color: #ffffff !important;
        }

        .stButton > button[kind="secondary"],
        .stButton > button[data-testid="stBaseButton-secondary"] {
            background: transparent;
            color: #1a1a1a !important;
            border: 1px solid #1a1a1a;
        }

        .stButton > button[kind="secondary"] *,
        .stButton > button[data-testid="stBaseButton-secondary"] * {
            color: #1a1a1a !important;
            fill: #1a1a1a !important;
        }

        .stButton > button[kind="secondary"]:hover,
        .stButton > button[data-testid="stBaseButton-secondary"]:hover {
            background: #1a1a1a;
            color: #ffffff !important;
        }

        .stButton > button[kind="secondary"]:hover *,
        .stButton > button[data-testid="stBaseButton-secondary"]:hover * {
            color: #ffffff !important;
            fill: #ffffff !important;
        }

        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox [data-baseweb="select"] > div,
        .stNumberInput input {
            background: #F5F0E8 !important;
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
            border: none !important;
            border-radius: 4px;
            outline: none !important;
            box-shadow: none !important;
        }

        .stSelectbox [data-baseweb="select"] > div,
        .stSelectbox [data-baseweb="select"] input,
        .stSelectbox [data-baseweb="select"] svg {
            color: #1a1a1a !important;
            fill: #1a1a1a !important;
        }

        div[data-baseweb="popover"],
        div[data-baseweb="popover"] [role="listbox"],
        div[data-baseweb="menu"],
        div[data-baseweb="menu"] ul {
            background: #F5F0E8 !important;
        }

        div[data-baseweb="popover"] [role="option"] {
            background: #F5F0E8 !important;
            color: #1a1a1a !important;
        }

        div[data-baseweb="popover"] [role="option"][aria-selected="true"],
        div[data-baseweb="popover"] [role="option"]:hover {
            background: #EDE8DC !important;
            color: #1a1a1a !important;
        }

        .stTextInput > div > div > input::placeholder,
        .stTextArea > div > div > textarea::placeholder {
            color: #666666 !important;
            -webkit-text-fill-color: #666666 !important;
            opacity: 1;
        }

        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus,
        .stSelectbox [data-baseweb="select"] > div:focus,
        .stSelectbox [data-baseweb="select"] input:focus,
        .stNumberInput input:focus {
            border: none !important;
            outline: none !important;
            box-shadow: none !important;
        }

        .stRadio > div {
            background: #ffffff;
            border: none !important;
            border-radius: 4px;
            padding: 8px 12px;
            box-shadow: none !important;
        }

        .editorial-hero {
            text-align: center;
            margin: 56px 0 42px 0;
        }

        .editorial-line {
            font-family: "Playfair Display", serif;
            font-size: clamp(2rem, 4.4vw, 3.5rem);
            line-height: 1.1;
            color: #1a1a1a;
        }

        .editorial-italic {
            font-style: italic;
        }

        .editorial-sub {
            margin-top: 16px;
            font-family: "Inter", sans-serif;
            font-style: italic;
            color: #5d5d5d;
            font-size: 0.95rem;
        }

        .answer-card {
            background: #ffffff;
            border: 1px solid #E0DAD0;
            border-radius: 6px;
            padding: 32px;
            margin-top: 20px;
            margin-bottom: 14px;
        }

        .answer-card p {
            font-family: "Inter", sans-serif;
            line-height: 1.75;
            color: #1a1a1a;
        }

        .source-chip {
            display: inline-block;
            background: #F0EBE0;
            border: 1px solid #e2dacd;
            color: #666666;
            border-radius: 4px;
            padding: 4px 10px;
            margin: 4px 4px 0 0;
            font-size: 0.8rem;
            font-family: "Inter", sans-serif;
        }

        .section-card {
            background: #ffffff;
            border: 1px solid #E0DAD0;
            border-radius: 6px;
            padding: 24px;
        }

        .fit-score {
            font-family: "Playfair Display", serif;
            font-size: 3rem;
            color: #FF6B35;
            line-height: 1;
            margin-bottom: 12px;
        }

        .field-label {
            font-family: "Playfair Display", serif;
            font-style: italic;
            color: #1a1a1a;
            margin-top: 4px;
            margin-bottom: 4px;
        }

        .company-table-wrap {
            border: 1px solid #E0DAD0;
            border-radius: 6px;
            overflow: hidden;
            background: #ffffff;
        }

        .company-table {
            width: 100%;
            border-collapse: collapse;
            font-family: "Inter", sans-serif;
        }

        .company-table th {
            text-align: left;
            background: #F0EBE0;
            border-bottom: 1px solid #E0DAD0;
            padding: 12px;
            font-weight: 500;
        }

        .company-table td {
            padding: 12px;
            border-bottom: 1px solid #F0EBE0;
            vertical-align: top;
        }

        .company-table tr:nth-child(even) {
            background: #FAF7F2;
        }

        .company-name {
            font-family: "Playfair Display", serif;
            font-size: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Custom sidebar toggle button (JS-powered) ─────────
import streamlit.components.v1 as components
components.html(
    """
    <script>
    (function() {
        const doc = window.parent.document;
        // Remove any existing toggle button
        const old = doc.getElementById('sidebar-toggle-btn');
        if (old) old.remove();

        // Create toggle button in the parent document
        const btn = doc.createElement('div');
        btn.id = 'sidebar-toggle-btn';
        btn.title = 'Toggle sidebar';
        btn.innerHTML = '\\u276E\\u276E';
        btn.style.cssText = `
            position: fixed; top: 14px; left: 14px; z-index: 999999;
            background: #EDE8DC; border: 1px solid #E0DAD0; border-radius: 4px;
            width: 32px; height: 32px; display: flex; align-items: center;
            justify-content: center; cursor: pointer; font-size: 16px;
            color: #1a1a1a; font-family: Inter, sans-serif; user-select: none;
            transition: background 0.2s ease; padding: 0; line-height: 1;
        `;
        btn.onmouseenter = () => { btn.style.background = '#E0DAD0'; };
        btn.onmouseleave = () => { btn.style.background = '#EDE8DC'; };

        btn.onclick = function() {
            const sidebar = doc.querySelector('[data-testid="stSidebar"]');
            if (!sidebar) return;
            const isHidden = sidebar.getAttribute('aria-expanded') === 'false' ||
                             sidebar.style.marginLeft === '-21rem';
            if (isHidden) {
                sidebar.style.marginLeft = '0';
                sidebar.style.visibility = 'visible';
                sidebar.setAttribute('aria-expanded', 'true');
                btn.innerHTML = '\\u276E\\u276E';
            } else {
                sidebar.style.marginLeft = '-21rem';
                sidebar.setAttribute('aria-expanded', 'false');
                btn.innerHTML = '\\u276F\\u276F';
            }
        };

        doc.body.appendChild(btn);

        // Sync button label with sidebar state
        setInterval(() => {
            const sidebar = doc.querySelector('[data-testid="stSidebar"]');
            if (!sidebar) return;
            const hidden = sidebar.getAttribute('aria-expanded') === 'false' ||
                           sidebar.style.marginLeft === '-21rem';
            btn.innerHTML = hidden ? '\\u276F\\u276F' : '\\u276E\\u276E';
        }, 500);
    })();
    </script>
    """,
    height=0,
)


def render_company_table(rows):
    """Render a custom HTML company table with editorial styling."""
    headers = ["Name", "Industry", "Batch", "Status", "Description"]
    parts = [
        '<div class="company-table-wrap"><table class="company-table"><thead><tr>'
    ]
    for header in headers:
        parts.append(f"<th>{html.escape(header)}</th>")
    parts.append("</tr></thead><tbody>")

    for row in rows:
        parts.append("<tr>")
        parts.append(
            f'<td class="company-name">{html.escape(str(row.get("Name", "")))}</td>'
        )
        parts.append(f"<td>{html.escape(str(row.get('Industry', '')))}</td>")
        parts.append(f"<td>{html.escape(str(row.get('Batch', '')))}</td>")
        parts.append(f"<td>{html.escape(str(row.get('Status', '')))}</td>")
        parts.append(f"<td>{html.escape(str(row.get('Description', '')))}</td>")
        parts.append("</tr>")

    parts.append("</tbody></table></div>")
    st.markdown("".join(parts), unsafe_allow_html=True)

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
    st.markdown(
        """
        <div style="margin-top: 10px; margin-bottom: 10px;">
            <svg width="48" height="48" viewBox="0 0 48 48">
                <rect width="48" height="48" rx="8" fill="#FF6B35"/>
                <text x="50%" y="58%"
                      dominant-baseline="middle"
                      text-anchor="middle"
                      font-family="Playfair Display, serif"
                      font-style="italic"
                      font-size="28"
                      fill="white">y</text>
            </svg>
        </div>
        <div style="font-family: 'Playfair Display', serif; font-size: 1.7rem; font-weight: 700; color: #1a1a1a;">
            YC Co-Founder
        </div>
        <div style="font-family: 'Playfair Display', serif; font-style: italic; color: #4f4f4f; margin-top: 6px;">
            Turning builders into formidable founders
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        """
        <div style="font-family: 'Inter', sans-serif; font-weight: 300; line-height: 1.9; color: #1a1a1a;">
            2,804 knowledge chunks<br>
            1,494 YC companies indexed<br>
            325 Paul Graham essay chunks<br>
            123 Startup School chunks
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Tabs ───────────────────────────────────────────────

tab_ask, tab_eval, tab_browse, tab_benchmark = st.tabs([
    "Ask YC",
    "Evaluate My Startup",
    "Browse YC Companies",
    "Benchmark",
])


# ════════════════════════════════════════════════════════
#  TAB 1 — Ask YC
# ════════════════════════════════════════════════════════

with tab_ask:
    st.markdown(
        """
        <div class="editorial-hero">
          <div class="editorial-line">YC turns builders</div>
          <div class="editorial-line editorial-italic">into formidable founders</div>
          <div class="editorial-sub">Ask anything. Get answers backed by real YC data.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Tip: type `/benchmark` to evaluate RAG accuracy in-app.")

    # Session state
    if "ask_query" not in st.session_state:
        st.session_state.ask_query = ""
    if "ask_result" not in st.session_state:
        st.session_state.ask_result = None

    def _parse_benchmark_limit(raw_query: str):
        """Parse `/benchmark` or `/benchmark N` commands."""
        parts = raw_query.strip().split()
        if not parts or parts[0].lower() != "/benchmark":
            return None
        if len(parts) == 1:
            return None
        if len(parts) == 2 and parts[1].isdigit():
            return max(1, int(parts[1]))
        raise ValueError("Use `/benchmark` or `/benchmark <number_of_questions>`")

    def _render_benchmark_result(report: dict):
        """Render benchmark summary and lists in the Ask tab."""
        st.markdown("---")
        st.subheader("RAG Accuracy Report")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Questions", report.get("total_questions", 0))
        c2.metric("Relevance", f"{report.get('avg_relevance_score', 0.0):.2f}")
        c3.metric("Source", f"{report.get('avg_source_score', 0.0):.2f}")
        c4.metric("Avg Latency", f"{report.get('avg_latency_sec', 0.0):.2f}s")
        c5.metric("Overall", f"{report.get('overall_rag_score', 0.0):.2f} / 1.00")

        st.caption(
            f"Sources checked: all available chunks per question"
        )

        st.markdown("**Weak questions (score < 0.6):**")
        weak = report.get("weak_questions", [])
        if weak:
            for q in weak:
                st.write(f"- {q}")
        else:
            st.write("- None")

        st.markdown("**Strong questions (score > 0.85):**")
        strong = report.get("strong_questions", [])
        if strong:
            for q in strong:
                st.write(f"- {q}")
        else:
            st.write("- None")

        st.caption("Detailed output is saved to benchmark_results.json")

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
        clean_query = query.strip()
        st.session_state.ask_query = ""
        with st.spinner("Searching YC knowledge base…"):
            try:
                advisor = load_advisor()
                if clean_query.lower().startswith("/benchmark"):
                    limit = _parse_benchmark_limit(clean_query)
                    with st.spinner("Running benchmark across test questions..."):
                        report = run_benchmark(
                            advisor=advisor,
                            max_questions=limit,
                            print_progress=False,
                        )
                    result = {"type": "benchmark", "report": report}
                else:
                    result = advisor.ask_with_sources(clean_query)
                st.session_state.ask_result = result
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.session_state.ask_result = None

    # Display
    result = st.session_state.ask_result
    if result:
        if result.get("type") == "benchmark":
            _render_benchmark_result(result["report"])
        else:
            answer_html = html.escape(result["answer"]).replace("\n", "<br>")
            st.markdown(
                f'<div class="answer-card">{answer_html}</div>',
                unsafe_allow_html=True,
            )

            if result["sources"]:
                st.markdown("<div class='field-label'>Sources</div>", unsafe_allow_html=True)
                tags = ""
                for s in result["sources"]:
                    author = f" — {s['author']}" if s["author"] else ""
                    tags += (
                        f'<span class="source-chip">'
                        f'[{s["source_type"]}] {s["title"]}{author}</span> '
                    )
                st.markdown(tags, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  TAB 2 — Evaluate My Startup
# ════════════════════════════════════════════════════════

with tab_eval:
    st.header("Evaluate My Startup")
    st.caption("Get a YC partner-style assessment of your idea")

    if "eval_result" not in st.session_state:
        st.session_state.eval_result = None

    left, right = st.columns([1, 1.4], gap="large")

    with left:
        st.markdown("<div class='field-label'>Describe your startup in one sentence</div>", unsafe_allow_html=True)
        desc = st.text_area(
            "Describe your startup in one sentence",
            height=80,
            placeholder="AI tool that automates legal contract review for small law firms",
            label_visibility="collapsed",
        )
        st.markdown("<div class='field-label'>Industry</div>", unsafe_allow_html=True)
        industry = st.selectbox(
            "Industry",
            ["Fintech", "SaaS", "Healthcare", "EdTech", "Consumer",
             "Crypto", "DevTools", "Marketplace", "AI/ML", "Other"],
            label_visibility="collapsed",
        )
        st.markdown("<div class='field-label'>Target Customer</div>", unsafe_allow_html=True)
        target = st.radio(
            "Target Customer",
            ["B2B", "B2C"],
            horizontal=True,
            label_visibility="collapsed",
        )
        st.markdown("<div class='field-label'>Stage</div>", unsafe_allow_html=True)
        stage = st.selectbox(
            "Stage",
            ["Idea", "Prototype", "Live", "Revenue"],
            label_visibility="collapsed",
        )
        st.markdown("<div class='field-label'>Team Size</div>", unsafe_allow_html=True)
        team_size = st.number_input("Team Size", min_value=1, max_value=10, value=2)
        st.markdown("<div class='field-label'>Founder Background</div>", unsafe_allow_html=True)
        background = st.text_input(
            "Founder Background",
            placeholder="Brief background — ex-Google, Stanford CS, domain expert etc.",
            label_visibility="collapsed",
        )

        eval_clicked = st.button(
            "Evaluate My Startup",
            type="primary",
            use_container_width=True,
        )

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
            avg_similarity = 0.0
            similarities = [c.get("similarity") for c in result["similar_companies"] if c.get("similarity") is not None]
            if similarities:
                avg_similarity = sum(similarities) / len(similarities)

            st.markdown(
                f"""
                <div class="section-card">
                  <div class="field-label">Estimated YC Fit Score</div>
                  <div class="fit-score">{int(avg_similarity * 100)}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("### Assessment")
            st.markdown(result["assessment"])
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("### Similar YC Companies")
            for comp in result["similar_companies"]:
                sim = comp.get("similarity")
                sim_str = f" · {sim:.0%} match" if sim is not None else ""
                st.markdown(
                    f"**{comp['name']}** — {comp['industry']}  \n"
                    f"Batch {comp['batch']} · {comp['status']}{sim_str}  \n"
                    f"_{comp['description']}_"
                )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.caption(result["disclaimer"])


# ════════════════════════════════════════════════════════
#  TAB 3 — Browse YC Companies
# ════════════════════════════════════════════════════════

with tab_browse:
    st.header("Browse YC Companies")

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

    st.markdown(f"<div class='field-label'>Showing {len(filtered)} companies</div>", unsafe_allow_html=True)

    if filtered:
        render_company_table(filtered)
    else:
        st.info("No companies match your filters.")


# ════════════════════════════════════════════════════════
#  TAB 4 — Benchmark
# ════════════════════════════════════════════════════════

with tab_benchmark:
    st.header("Benchmark RAG Accuracy")
    st.caption("Run the evaluation suite with latency and extended source checks.")

    if "benchmark_report" not in st.session_state:
        st.session_state.benchmark_report = None

    all_questions = load_questions(QUESTIONS_PATH)
    total_questions = len(all_questions)

    b1, b2 = st.columns([1.5, 1])
    with b1:
        question_limit = st.number_input(
            "How many benchmark questions to run",
            min_value=1,
            max_value=total_questions,
            value=total_questions,
            step=1,
        )
    with b2:
        run_benchmark_clicked = st.button(
            "Run Benchmark",
            type="primary",
            use_container_width=True,
        )

    if run_benchmark_clicked:
        advisor = load_advisor()
        selected_questions = all_questions[:question_limit]

        progress = st.progress(0.0, text="Preparing benchmark run...")
        status = st.empty()

        results = []
        running_relevance = 0.0
        running_source = 0.0
        running_hallucination = 0.0
        latencies = []
        weak_questions = []
        strong_questions = []

        for index, item in enumerate(selected_questions, start=1):
            status.info(f"Running Q{index}/{question_limit}: {item['question']}")
            try:
                result, scores = evaluate_question(
                    advisor,
                    item,
                )
            except Exception as exc:
                if "429" in str(exc):
                    status.warning(f"Rate limit hit at Q{index}. Showing partial results ({len(results)} completed).")
                    break
                raise
            results.append(result)

            running_relevance += scores["relevance"]
            running_source += scores["source"]
            running_hallucination += scores["hallucination"]
            latencies.append(scores["total_latency_sec"])

            if scores["average"] < 0.6:
                weak_questions.append(item["question"])
            if scores["average"] > 0.85:
                strong_questions.append(item["question"])

            progress.progress(index / question_limit, text=f"Completed {index}/{question_limit}")

        question_limit = len(results)  # adjust to actual completed count

        if question_limit == 0:
            st.error("No questions completed. All API keys may have hit their rate limit.")
        else:
            avg_relevance = running_relevance / question_limit
            avg_source = running_source / question_limit
            avg_hallucination = running_hallucination / question_limit
            avg_latency = sum(latencies) / question_limit if question_limit else 0.0
            ordered = sorted(latencies) if question_limit else []
            p95_idx = max(0, min(len(ordered) - 1, int(round(0.95 * len(ordered) - 1)))) if ordered else 0
            p95_latency = ordered[p95_idx] if ordered else 0.0
            overall_score = (avg_relevance + avg_source + avg_hallucination) / 3

            report = {
                "total_questions": question_limit,
                "avg_relevance_score": round(avg_relevance, 4),
                "avg_source_score": round(avg_source, 4),
                "avg_hallucination_score": round(avg_hallucination, 4),
                "avg_latency_sec": round(avg_latency, 4),
                "p95_latency_sec": round(p95_latency, 4),
                "overall_rag_score": round(overall_score, 4),
                "weak_questions": weak_questions,
                "strong_questions": strong_questions,
                "results": results,
            }

            with open(RESULTS_PATH, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            st.session_state.benchmark_report = report
            status.success("Benchmark complete. Report saved to benchmark_results.json")

    report = st.session_state.benchmark_report
    if report:
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Questions", report["total_questions"])
        m2.metric("Relevance", f"{report['avg_relevance_score']:.2f}")
        m3.metric("Source", f"{report['avg_source_score']:.2f}")
        m4.metric("Avg Latency", f"{report.get('avg_latency_sec', 0.0):.2f}s")
        m5.metric("P95 Latency", f"{report.get('p95_latency_sec', 0.0):.2f}s")
        m6.metric("Overall", f"{report['overall_rag_score']:.2f} / 1.00")

        st.caption(
            "Sources checked: all available chunks per question"
        )

        s1, s2 = st.columns(2)
        with s1:
            st.markdown("**Weak questions (score < 0.6):**")
            if report["weak_questions"]:
                for question in report["weak_questions"]:
                    st.write(f"- {question}")
            else:
                st.write("- None")
        with s2:
            st.markdown("**Strong questions (score > 0.85):**")
            if report["strong_questions"]:
                for question in report["strong_questions"]:
                    st.write(f"- {question}")
            else:
                st.write("- None")

        st.markdown("**Per-question scores**")
        rows = []
        for item in report["results"]:
            rows.append({
                "Question": item["question"],
                "Relevance": item["scores"]["relevance"],
                "Source": item["scores"]["source"],
                "Hallucination": item["scores"]["hallucination"],
                "Sources Checked": item.get("source_count_checked", 0),
                "Latency (s)": item.get("latency", {}).get("total_latency_sec", 0.0),
                "Average": item["scores"]["average"],
            })
        st.dataframe(rows, width="stretch", hide_index=True)

        st.download_button(
            label="Download JSON Report",
            data=json.dumps(report, indent=2, ensure_ascii=False),
            file_name="benchmark_results.json",
            mime="application/json",
            use_container_width=True,
        )
    else:
        st.info("Run the benchmark to see results here.")
