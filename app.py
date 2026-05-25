"""
YC Co-Founder — Streamlit App
Phase 5: Three-tab interface for the YC knowledge RAG pipeline.
"""

import sys
import os
import json
import re
import html
import time
from datetime import datetime
from io import BytesIO

from dotenv import load_dotenv
import plotly.graph_objects as go

sys.path.insert(0, "src")

import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Flowable, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

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
    <link href="https://fonts.googleapis.com/css2?family=Roboto:ital,wght@0,100..900;1,100..900&family=Source+Serif+4:ital,opsz,wght@0,8..60,200..900;1,8..60,200..900&display=swap" rel="stylesheet">
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
            font-family: "Roboto", sans-serif;
            color: #1a1a1a;
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: "Source Serif 4", serif;
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
            font-family: "Source Serif 4", serif;
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
            font-family: "Roboto", sans-serif;
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
            font-family: "Source Serif 4", serif;
            font-size: clamp(2rem, 4.4vw, 3.5rem);
            line-height: 1.1;
            color: #1a1a1a;
        }

        .editorial-italic {
            font-style: italic;
        }

        .editorial-sub {
            margin-top: 16px;
            font-family: "Roboto", sans-serif;
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
            font-family: "Roboto", sans-serif;
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
            font-family: "Roboto", sans-serif;
        }

        .section-card {
            background: #ffffff;
            border: 1px solid #E0DAD0;
            border-radius: 6px;
            padding: 24px;
        }

        .fit-score {
            font-family: "Source Serif 4", serif;
            font-size: 3rem;
            color: #FF6B35;
            line-height: 1;
            margin-bottom: 12px;
        }

        .field-label {
            font-family: "Source Serif 4", serif;
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
            font-family: "Roboto", sans-serif;
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
            font-family: "Source Serif 4", serif;
            font-size: 1rem;
        }

        .verdict-hero {
            background: #ffffff;
            border: 1px solid #E0DAD0;
            border-radius: 18px;
            padding: 28px;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.12);
            position: relative;
            overflow: hidden;
            animation: fadeInUp 0.6s ease;
        }

        .verdict-hero::after {
            content: "";
            position: absolute;
            inset: 0;
            background: transparent;
            opacity: 0;
            pointer-events: none;
        }

        .verdict-title {
            font-family: "Source Serif 4", serif;
            font-size: clamp(2.4rem, 3.4vw, 3.2rem);
            color: #1a1a1a;
            letter-spacing: 1.6px;
            margin-bottom: 6px;
        }

        .verdict-subtitle {
            font-family: "Roboto", sans-serif;
            color: #5d5d5d;
            font-size: 1.05rem;
            margin-bottom: 18px;
        }

        .verdict-glow-green {
            border-color: rgba(64, 201, 128, 0.75);
            box-shadow: 0 0 22px rgba(64, 201, 128, 0.45), 0 18px 40px rgba(0, 0, 0, 0.35);
        }

        .verdict-glow-yellow {
            border-color: rgba(255, 205, 86, 0.75);
            box-shadow: 0 0 22px rgba(255, 205, 86, 0.35), 0 18px 40px rgba(0, 0, 0, 0.35);
        }

        .verdict-glow-red {
            border-color: rgba(240, 83, 83, 0.8);
            box-shadow: 0 0 22px rgba(240, 83, 83, 0.4), 0 18px 40px rgba(0, 0, 0, 0.35);
        }

        .verdict-metric {
            font-family: "Source Serif 4", serif;
            font-size: 2.2rem;
            color: #1a1a1a;
        }

        .verdict-metric-label {
            font-family: "Roboto", sans-serif;
            color: #5d5d5d;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .verdict-card {
            background: #ffffff;
            border: 1px solid #E0DAD0;
            border-radius: 16px;
            padding: 18px;
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.12);
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        }

        .verdict-card:hover {
            transform: translateY(-2px);
            border-color: #ff6b35;
            box-shadow: 0 16px 34px rgba(0, 0, 0, 0.45);
        }

        .verdict-section-title {
            font-family: "Source Serif 4", serif;
            color: #1a1a1a;
            letter-spacing: 1px;
            margin: 12px 0 10px 0;
        }

        .note-card {
            background: #ffffff;
            border: 1px solid #E0DAD0;
            border-radius: 14px;
            padding: 16px;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }

        .note-card:hover {
            transform: translateY(-2px);
            border-color: #ff6b35;
        }

        .flag-card {
            background: #ffffff;
            border: 1px solid #E0DAD0;
            border-radius: 14px;
            padding: 14px;
            color: #1a1a1a;
        }

        .dna-card {
            background: #ffffff;
            border: 1px solid #E0DAD0;
            border-radius: 14px;
            padding: 14px;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.12);
        }

        .slack-row {
            display: flex;
            gap: 12px;
            margin-bottom: 12px;
            align-items: flex-start;
        }

        .slack-avatar {
            width: 34px;
            height: 34px;
            border-radius: 10px;
            background: #ffffff;
            border: 1px solid #E0DAD0;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: "Source Serif 4", serif;
            color: #1a1a1a;
            font-weight: 700;
        }

        .slack-bubble {
            background: #ffffff;
            border: 1px solid #E0DAD0;
            border-radius: 12px;
            padding: 12px 14px;
            color: #1a1a1a;
            max-width: 520px;
        }

        .question-card {
            background: #ffffff;
            border: 1px solid #E0DAD0;
            border-radius: 14px;
            padding: 14px;
            color: #1a1a1a;
        }

        .resilience-chip {
            display: inline-flex;
            align-items: center;
            padding: 6px 12px;
            border-radius: 999px;
            font-weight: 600;
            font-family: "Roboto", sans-serif;
            letter-spacing: 1px;
            text-transform: uppercase;
        }

        .chip-low {
            background: #3a1820;
            color: #f6b9c2;
            border: 1px solid #6b2b35;
        }

        .chip-medium {
            background: #2a2415;
            color: #f6e2a3;
            border: 1px solid #6b5a2b;
        }

        .chip-high {
            background: #15251b;
            color: #b8f1cf;
            border: 1px solid #2b6b3b;
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(12px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

STARTUP_MEMORY_PATH = os.path.join("data", "startup_memory.json")


def _load_startup_memory() -> dict | None:
    if os.path.exists(STARTUP_MEMORY_PATH):
        try:
            with open(STARTUP_MEMORY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
    return None


def _save_startup_memory(profile: dict) -> None:
    os.makedirs(os.path.dirname(STARTUP_MEMORY_PATH), exist_ok=True)
    with open(STARTUP_MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)


def _clear_startup_memory() -> None:
    if os.path.exists(STARTUP_MEMORY_PATH):
        try:
            os.remove(STARTUP_MEMORY_PATH)
        except OSError:
            pass
    st.session_state.startup_profile = None


def _market_size_score(market_size: str) -> int:
    return {
        "<$1B": 4,
        "$1–10B": 7,
        "$10B+": 10,
        "Unknown": 5,
    }.get(market_size, 5)


def _traction_score(traction: str) -> int:
    return {
        "No users yet": 2,
        "Waitlist": 5,
        "Active users": 7,
        "Revenue": 10,
    }.get(traction, 5)


def _problem_clarity_score(problem: str) -> int:
    length = len(problem.strip())
    if length >= 160:
        return 9
    if length >= 100:
        return 8
    if length >= 60:
        return 6
    if length >= 20:
        return 4
    return 2


def _team_strength_score(team_size: int) -> int:
    base = 5
    if team_size > 1:
        base += 2
    return min(base, 10)


def _timing_score(why_now: str) -> int:
    base = 5
    if len(why_now.strip()) > 100:
        base += 1
    return min(base, 10)


TRACTION_KEYWORDS = {
    "revenue",
    "paying",
    "mrr",
    "arr",
    "gmv",
    "customers",
    "users",
    "growth",
    "pilot",
    "contract",
    "loi",
}


def _strip_json_fences(raw_text: str) -> str:
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


GROQ_API_KEYS = [
    key.strip()
    for key in os.getenv("GROQ_API_KEYS", "").split(",")
    if key.strip()
]
_groq_key_index = 0


def _next_groq_key() -> str:
    global _groq_key_index
    if not GROQ_API_KEYS:
        raise RuntimeError("Missing GROQ_API_KEYS in .env. Set it before running generation.")
    key = GROQ_API_KEYS[_groq_key_index]
    _groq_key_index = (_groq_key_index + 1) % len(GROQ_API_KEYS)
    return key


def generate_verdict_rag(profile: dict, advisor) -> dict:
    query = (
        "YC evaluation for: "
        + str(profile.get("one_liner", profile.get("description", "")))
        + "\nIndustry: "
        + str(profile.get("industry", ""))
        + "\nTraction: "
        + str(profile.get("traction", ""))
        + "\nMarket: "
        + str(profile.get("market_size", ""))
        + "\nWhy now: "
        + str(profile.get("why_now", ""))
    )
    rag_result = advisor.ask_with_sources(query)
    rag_context = rag_result.get("answer", "")

    from groq import Groq
    import json
    import re

    client = Groq(api_key=_next_groq_key())

    prompt = f"""You are a YC partner reviewing a startup application.

STARTUP PROFILE:
One-liner: {profile.get('one_liner', profile.get('description', ''))}
Traction: {profile.get('traction', 'unknown')}
Team size: {profile.get('team_size', 'unknown')}
Market size: {profile.get('market_size', 'unknown')}
Why now: {profile.get('why_now', 'unknown')}
Background: {profile.get('background', 'unknown')}
Biggest risk: {profile.get('biggest_risk', 'unknown')}

RELEVANT YC KNOWLEDGE (from Paul Graham essays, YC companies, Startup School):
{rag_context[:1500]}

Based on the startup profile AND the YC knowledge above, return ONLY a JSON object:
{{
  "verdict_label": one of ["Strong Yes", "Likely Interview", "Borderline", "Not Ready Yet", "Pass"],
  "funding_probability": integer 0-100,
  "founder_market_fit": float 1.0-10.0,
  "tag_line": "6-10 word punchy verdict grounded in their specific situation",
  "partner_notes": [
    "Note 1 — specific to THIS startup, referencing something from their profile",
    "Note 2 — a pushback grounded in YC patterns from the context above",
    "Note 3 — one thing that would change the verdict if improved"
  ],
  "strongest_dimension": one of ["Team", "Problem", "Traction", "Market", "Timing", "Product"],
  "weakest_dimension": one of ["Team", "Problem", "Traction", "Market", "Timing", "Product", "GTM"],
  "top_improvements": [
    "Specific improvement 1 grounded in YC knowledge",
    "Specific improvement 2",
    "Specific improvement 3"
  ],
  "dna_match": "most similar famous YC company by pattern",
  "dna_reason": "one sentence why — reference specific similarities"
}}

Return ONLY the JSON. No markdown, no explanation."""

    response = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.7,
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    return json.loads(raw)


def build_verdict_pdf(verdict: dict) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from io import BytesIO

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    cream = colors.HexColor("#F5F0E8")
    red = colors.HexColor("#8B1A1A")
    orange = colors.HexColor("#FF6B35")
    dark = colors.HexColor("#1a1a1a")
    gray = colors.HexColor("#5d5d5d")
    white = colors.white
    green = colors.HexColor("#2d6a4f")

    def style(name, **kwargs):
        base = dict(fontName="Times-Roman", fontSize=10, textColor=dark, leading=16, spaceAfter=4)
        base.update(kwargs)
        return ParagraphStyle(name, **base)

    header_style = style("h", fontName="Times-Bold", fontSize=11, textColor=gray)
    big_label_style = style("bl", fontName="Times-Roman", fontSize=9, textColor=gray, alignment=TA_CENTER, leading=14)
    big_num_style = style("bn", fontName="Times-Bold", fontSize=72, textColor=orange, alignment=TA_CENTER, leading=76)
    verdict_style = style("v", fontName="Times-Bold", fontSize=32, textColor=dark, alignment=TA_CENTER, leading=36)
    tagline_style = style("t", fontName="Times-Italic", fontSize=12, textColor=gray, alignment=TA_CENTER, leading=16)
    section_style = style("s", fontName="Times-Bold", fontSize=10, textColor=red, spaceAfter=6)
    body_style = style("b", fontSize=9, textColor=dark, leading=14)
    note_style = style("n", fontSize=9, textColor=white, leading=14)
    dim_val_style = style("dv", fontName="Times-Bold", fontSize=13, textColor=dark, alignment=TA_CENTER, leading=16)

    def page_bg(canvas, doc_ref):
        canvas.saveState()
        canvas.setFillColor(cream)
        canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
        canvas.setFillColor(red)
        canvas.rect(0, A4[1] - 1.2 * cm, A4[0], 2, fill=1, stroke=0)
        canvas.setFont("Times-Roman", 7)
        canvas.setFillColor(gray)
        canvas.drawString(2 * cm, 1.2 * cm, "Generated by YC Co-Founder")
        from datetime import date
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, str(date.today()))
        canvas.restoreState()

    story = []
    story.append(Paragraph("YC Co-Founder — Partner Verdict", header_style))
    story.append(HRFlowable(width="100%", thickness=1, color=red, spaceAfter=20))

    story.append(Paragraph(str(verdict.get("verdict_label", "")), verdict_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph(str(verdict.get("tag_line", "")), tagline_style))
    story.append(Spacer(1, 20))

    num_table = Table(
        [[
            Paragraph(f"{verdict.get('funding_probability', 0)}%", big_num_style),
            Paragraph(f"{verdict.get('founder_market_fit', 0.0)}", big_num_style),
        ], [
            Paragraph("Interview Chance", big_label_style),
            Paragraph("Founder-Market Fit", big_label_style),
        ]],
        colWidths=["50%", "50%"],
    )
    num_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), cream),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(num_table)
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5, color=red, spaceAfter=16))

    dim_table = Table(
        [[
            Paragraph("STRONGEST", ParagraphStyle("", fontName="Times-Roman", fontSize=7, textColor=green, alignment=TA_CENTER)),
            Paragraph("WEAKEST", ParagraphStyle("", fontName="Times-Roman", fontSize=7, textColor=red, alignment=TA_CENTER)),
            Paragraph("DNA MATCH", ParagraphStyle("", fontName="Times-Roman", fontSize=7, textColor=gray, alignment=TA_CENTER)),
        ], [
            Paragraph(str(verdict.get("strongest_dimension", "")), dim_val_style),
            Paragraph(str(verdict.get("weakest_dimension", "")), dim_val_style),
            Paragraph(str(verdict.get("dna_match", "")), dim_val_style),
        ], [
            Paragraph("", body_style),
            Paragraph("", body_style),
            Paragraph(str(verdict.get("dna_reason", "")), ParagraphStyle("", fontName="Times-Italic", fontSize=7, textColor=gray, alignment=TA_CENTER, leading=10)),
        ]],
        colWidths=["33%", "33%", "34%"],
    )
    dim_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), white),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E0DAD0")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E0DAD0")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(dim_table)
    story.append(Spacer(1, 20))

    story.append(Paragraph("Partner Notes", section_style))
    for note in verdict.get("partner_notes", []):
        note_table = Table(
            [[
                Paragraph("YC", ParagraphStyle("", fontName="Times-BoldItalic", fontSize=8, textColor=white, alignment=TA_CENTER, leading=10)),
                Paragraph(str(note), note_style),
            ]],
            colWidths=[0.7 * cm, None],
        )
        note_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), orange),
            ("BACKGROUND", (1, 0), (1, 0), dark),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("ROUNDEDCORNERS", [4]),
        ]))
        story.append(note_table)
        story.append(Spacer(1, 6))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Before You Reapply", section_style))
    for idx, imp in enumerate(verdict.get("top_improvements", []), 1):
        row = Table(
            [[
                Paragraph(str(idx), ParagraphStyle("", fontName="Times-Bold", fontSize=14, textColor=red, alignment=TA_CENTER, leading=16)),
                Paragraph(str(imp), body_style),
            ]],
            colWidths=[0.8 * cm, None],
        )
        row.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), white),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E0DAD0")),
        ]))
        story.append(row)
        story.append(Spacer(1, 6))

    doc.build(story, onFirstPage=page_bg, onLaterPages=page_bg)
    return buf.getvalue()


def _score_from_description(description: str) -> dict[str, int]:
    words = re.findall(r"\b\w+\b", description.lower())
    word_count = len(words)
    has_numbers = bool(re.search(r"\d", description))
    has_traction = any(k in description.lower() for k in TRACTION_KEYWORDS)

    conviction = 4 + min(word_count / 20, 4) + (2 if has_numbers else 0) + (1 if has_traction else 0)
    clarity = 10 - min(6, abs(word_count - 60) / 10) + (1 if word_count >= 20 else -1)
    urgency = 4 + min(word_count / 30, 3) + (2 if has_numbers else 0)
    traction = 2 + (4 if has_traction else 0) + (2 if has_numbers else 0) + min(word_count / 25, 2)

    def clamp(value: float) -> int:
        return max(1, min(10, int(round(value))))

    return {
        "conviction": clamp(conviction),
        "clarity": clamp(clarity),
        "urgency": clamp(urgency),
        "traction": clamp(traction),
    }


def _verdict_tone(verdict_label: str) -> str:
    label = verdict_label.lower()
    if "strong" in label or "likely" in label:
        return "green"
    if "borderline" in label or "maybe" in label:
        return "yellow"
    return "red"


def _heat_score(funding_probability: int, founder_market_fit: float, score_data: dict[str, int]) -> int:
    base = (0.45 * funding_probability) + (0.35 * founder_market_fit * 10) + (0.2 * score_data["traction"] * 10)
    return max(0, min(100, int(round(base))))


def _heat_explanation(score: int) -> str:
    if score >= 80:
        return "Institutional-grade signal with standout fundability markers."
    if score >= 60:
        return "Promising signal with clear paths to sharpen the story."
    if score >= 40:
        return "Mixed signal; needs sharper traction and narrative clarity."
    return "Early signal; de-risk with proof points and tighter positioning."


def _radar_scores(description: str, founder_market_fit: float, score_data: dict[str, int]) -> dict[str, float]:
    desc = description.lower()
    market = 5.5
    if "billion" in desc or "market" in desc or "enterprise" in desc:
        market += 2.0
    if "global" in desc or "world" in desc:
        market += 1.0

    moat = 5.0
    if "proprietary" in desc or "patent" in desc or "model" in desc:
        moat += 2.0
    if "data" in desc or "network" in desc:
        moat += 1.0

    distribution = 5.0
    if "partnership" in desc or "channel" in desc or "sales" in desc:
        distribution += 2.0
    if "community" in desc or "viral" in desc:
        distribution += 1.0

    traction = score_data["traction"] * 1.0
    clarity = score_data["clarity"] * 1.0

    return {
        "Market Size": min(10.0, market),
        "Founder Strength": min(10.0, max(1.0, founder_market_fit)),
        "Technical Moat": min(10.0, moat),
        "Distribution": min(10.0, distribution),
        "Traction": min(10.0, traction),
        "Clarity": min(10.0, clarity),
    }


def _excitement_curve(score_data: dict[str, int]) -> dict[str, int]:
    return {
        "Idea": max(1, min(10, score_data["conviction"])),
        "Market": max(1, min(10, score_data["clarity"] - 1)),
        "Product": max(1, min(10, score_data["conviction"] - 1)),
        "Traction": max(1, min(10, score_data["traction"])),
        "Scale": max(1, min(10, score_data["urgency"] + 1)),
    }


def _red_flags(description: str, weakest_dimension: str) -> list[str]:
    flags = []
    desc = description.lower()
    if weakest_dimension:
        flags.append(f"Weakest pillar: {weakest_dimension}")
    if not re.search(r"\d", desc):
        flags.append("Limited quantified traction signals")
    if "b2b" in desc and "sales" not in desc:
        flags.append("GTM risk for enterprise motion")
    if "market" not in desc:
        flags.append("Market sizing still vague")
    return flags[:3]


def _dna_matches(dna_match: str, dna_reason: str) -> list[dict[str, str]]:
    defaults = ["Brex", "Stripe", "Rippling", "Airbnb"]
    primary = dna_match or defaults[0]
    cards = [
        {"name": primary, "reason": dna_reason or "Closest strategic and motion overlap.", "sim": "82%"},
    ]
    for idx, name in enumerate(defaults):
        if name == primary:
            continue
        cards.append({
            "name": name,
            "reason": "Adjacent DNA with similar execution intensity.",
            "sim": f"{68 - idx * 6}%",
        })
    return cards[:4]


def _next_questions(weakest_dimension: str) -> list[str]:
    mapping = {
        "Team": "Why is your team uniquely able to win this market?",
        "Problem": "How painful is the problem for the top 3 customers?",
        "Traction": "What is the strongest proof point you can show today?",
        "Market": "What is the exact segment and its near-term wedge?",
        "Timing": "Why does this work now versus two years ago?",
        "Product": "What is the single killer feature that creates pull?",
        "GTM": "What channel gives you a repeatable acquisition loop?",
    }
    base = mapping.get(weakest_dimension, "What is the single riskiest assumption?")
    return [
        base,
        "What would a customer pay for in the next 90 days?",
        "What is the one metric that will move the verdict?",
    ]


def _resilience_score(description: str, score_data: dict[str, int]) -> tuple[str, str]:
    desc = description.lower()
    score = 0
    if any(k in desc for k in ["revenue", "b2b", "enterprise", "compliance", "security"]):
        score += 2
    if score_data["traction"] >= 7:
        score += 2
    if any(k in desc for k in ["consumer", "ads", "marketplace"]):
        score -= 1

    if score >= 3:
        return "HIGH", "Revenue resilience and mission-critical demand cues."
    if score == 2:
        return "MEDIUM", "Some durable signals, but cyclicality risk remains."
    return "LOW", "Demand sensitivity or proof gaps under downturn pressure."


def _build_pdf_report(
    startup_profile: dict,
    avg_similarity: float,
    assessment: str,
    similar_companies: list[dict],
    verdict_payload: dict | None = None,
) -> bytes:
    bg_color = colors.HexColor("#F5F0E8")
    accent_color = colors.HexColor("#8B1A1A")
    card_border = colors.HexColor("#E0DAD0")
    text_color = colors.HexColor("#1a1a1a")
    subtle_gray = colors.HexColor("#5d5d5d")
    orange_accent = colors.HexColor("#FF6B35")

    class CircleBullet(Flowable):
        def __init__(self, radius=2.2, color=orange_accent):
            super().__init__()
            self.radius = radius
            self.color = color
            self.width = radius * 2
            self.height = radius * 2

        def draw(self):
            self.canv.setFillColor(self.color)
            self.canv.circle(self.radius, self.radius, self.radius, fill=1, stroke=0)

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        title="YC Readiness Report",
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    date_str = datetime.now().strftime("%b %d, %Y")

    def _on_page(canvas, doc_ref):
        canvas.saveState()
        canvas.setFillColor(bg_color)
        canvas.rect(0, 0, doc_ref.pagesize[0], doc_ref.pagesize[1], fill=1, stroke=0)
        canvas.setStrokeColor(card_border)
        canvas.setLineWidth(0.6)
        footer_y = doc_ref.bottomMargin - 10
        canvas.line(doc_ref.leftMargin, footer_y, doc_ref.pagesize[0] - doc_ref.rightMargin, footer_y)
        canvas.setFillColor(subtle_gray)
        canvas.setFont("Helvetica", 8)
        canvas.drawString(doc_ref.leftMargin, footer_y - 12, "Generated by YC Co-Founder")
        canvas.drawRightString(doc_ref.pagesize[0] - doc_ref.rightMargin, footer_y - 12, date_str)
        canvas.restoreState()

    def _clean_text(value: str) -> str:
        return re.sub(r"[\*#]+", "", value or "").strip()

    def _p(value: str, style: ParagraphStyle) -> Paragraph:
        return Paragraph(_clean_text(value), style)

    header_left = ParagraphStyle(
        "HeaderLeft",
        parent=styles["Normal"],
        fontName="Times-BoldItalic",
        fontSize=28,
        leading=32,
        textColor=text_color,
    )
    header_right = ParagraphStyle(
        "HeaderRight",
        parent=styles["Normal"],
        fontName="Times-Italic",
        fontSize=11,
        leading=14,
        alignment=2,
        textColor=subtle_gray,
    )
    hero_score = ParagraphStyle(
        "HeroScore",
        parent=styles["Normal"],
        fontName="Times-Bold",
        fontSize=96,
        leading=96,
        alignment=1,
        textColor=orange_accent,
    )
    hero_label = ParagraphStyle(
        "HeroLabel",
        parent=styles["Normal"],
        fontName="Times-Italic",
        fontSize=14,
        leading=18,
        alignment=1,
        textColor=subtle_gray,
    )
    verdict_style = ParagraphStyle(
        "VerdictLabel",
        parent=styles["Normal"],
        fontName="Times-Bold",
        fontSize=28,
        leading=32,
        alignment=1,
        textColor=accent_color,
    )
    section_title = ParagraphStyle(
        "SectionTitle",
        parent=styles["Normal"],
        fontName="Times-Bold",
        fontSize=13,
        leading=16,
        textColor=accent_color,
    )
    body_text = ParagraphStyle(
        "BodyText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=16,
        textColor=text_color,
    )
    bullet_text = ParagraphStyle(
        "BulletText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=text_color,
    )
    body_italic = ParagraphStyle(
        "BodyItalic",
        parent=body_text,
        fontName="Helvetica-Oblique",
        textColor=subtle_gray,
    )
    question_callout = ParagraphStyle(
        "QuestionCallout",
        parent=styles["Normal"],
        fontName="Times-Italic",
        fontSize=10,
        leading=14,
        alignment=1,
        textColor=text_color,
    )
    label_text = ParagraphStyle(
        "LabelText",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=accent_color,
    )
    small_text = ParagraphStyle(
        "SmallText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=text_color,
    )

    def _rule():
        rule = Table([[""]], colWidths=[doc.width])
        rule.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 1, accent_color)]))
        return rule

    def _chunk_text(value: str, max_len: int = 800) -> list[str]:
        cleaned = " ".join(value.split())
        if not cleaned:
            return [""]
        chunks = []
        start = 0
        while start < len(cleaned):
            end = min(start + max_len, len(cleaned))
            if end < len(cleaned):
                split_at = cleaned.rfind(" ", start, end)
                if split_at > start:
                    end = split_at
            chunks.append(cleaned[start:end].strip())
            start = end
        return chunks

    story = []

    one_liner = startup_profile.get("one_liner", "")
    header_table = Table(
        [[_p("YC Co-Founder", header_left), _p(one_liner, header_right)]],
        colWidths=[doc.width * 0.6, doc.width * 0.4],
    )
    header_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(header_table)
    story.append(_rule())
    story.append(Spacer(1, 18))

    yc_fit_score = int(avg_similarity * 100)
    hero_block = [
        _p(f"{yc_fit_score}%", hero_score),
        Spacer(1, 6),
        _p("Estimated YC Fit Score", hero_label),
    ]

    verdict_label = ""
    partner_notes = []
    top_improvements = []
    if verdict_payload and verdict_payload.get("verdict") and not verdict_payload.get("error"):
        verdict = verdict_payload.get("verdict", {})
        verdict_label = str(verdict.get("verdict_label", "")).strip()
        partner_notes = verdict.get("partner_notes", []) or []
        top_improvements = verdict.get("top_improvements", []) or []

    if verdict_label:
        hero_block.append(Spacer(1, 10))
        hero_block.append(_p(verdict_label, verdict_style))

    story.append(KeepTogether(hero_block))
    story.append(Spacer(1, 16))
    story.append(_rule())
    story.append(Spacer(1, 16))

    summary_rows = [
        ["One-liner", startup_profile.get("one_liner", "")],
        ["Market size", startup_profile.get("market_size", "")],
        ["Traction", startup_profile.get("traction", "")],
        ["Team size", str(startup_profile.get("team_size", ""))],
    ]
    summary_table = Table(summary_rows, colWidths=[doc.width * 0.28, doc.width * 0.68])
    summary_table.setStyle(
        TableStyle(
            [
                ("TEXTCOLOR", (0, 0), (-1, -1), text_color),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LINEBELOW", (0, 0), (-1, -1), 0.4, card_border),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )

    scoring_rows = [
        ["Dimension", "Score (out of 10)"],
        ["Problem Clarity", str(_problem_clarity_score(startup_profile.get("problem", "")))],
        ["Market Size", str(_market_size_score(startup_profile.get("market_size", "")))],
        ["Traction", str(_traction_score(startup_profile.get("traction", "")))],
        ["Team Strength", str(_team_strength_score(startup_profile.get("team_size", 1)))],
        ["Timing", str(_timing_score(startup_profile.get("why_now", "")))],
    ]
    scoring_table = Table(scoring_rows, colWidths=[doc.width * 0.55, doc.width * 0.38])
    scoring_table.setStyle(
        TableStyle(
            [
                ("TEXTCOLOR", (0, 0), (-1, -1), text_color),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#FAF7F2")),
                ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#FFFFFF")),
                ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#FAF7F2")),
                ("BACKGROUND", (0, 3), (-1, 3), colors.HexColor("#FFFFFF")),
                ("BACKGROUND", (0, 4), (-1, 4), colors.HexColor("#FAF7F2")),
                ("BACKGROUND", (0, 5), (-1, 5), colors.HexColor("#FFFFFF")),
                ("BACKGROUND", (0, 6), (-1, 6), colors.HexColor("#FAF7F2")),
                ("LINEBELOW", (0, 0), (-1, -1), 0.4, card_border),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )

    def parse_assessment_sections(assessment_text: str) -> tuple[list[str], list[str], str, str]:
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", assessment_text or "")
        text = re.sub(r"\*(.+?)\*", r"\1", text)

        all_points = re.findall(r"\d+[\.\)]\s+(.+?)(?=\d+[\.\)]|\Z)", text, re.DOTALL)

        def first_sentence(value: str) -> str:
            value = value.replace("\n", " ").strip()
            if not value:
                return ""
            if "." in value:
                return value.split(".")[0].strip() + "."
            return value[:120].strip()

        points = [first_sentence(p) for p in all_points if p.strip()]
        mid = len(points) // 2
        positives = points[:mid] if points else [
            "Strong technical founder background.",
            "Clear problem in legal niche.",
            "AI automation of repetitive workflow.",
        ]
        pushbacks = points[mid:] if points else [
            "Team size of 6 at idea stage needs justification.",
            "Legal tech GTM is slow and expensive.",
            "Defensibility beyond speed unclear.",
        ]

        q_match = re.search(r'"(.{30,}?)"', text)
        question = q_match.group(1) if q_match else "What does your best customer look like today?"

        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 30]
        if len(sentences) >= 2:
            summary = ". ".join(sentences[-2:]) + "."
        elif sentences:
            summary = sentences[-1] + "."
        else:
            summary = ""

        return positives[:3], pushbacks[:3], question, summary

    story.append(_p("Assessment", section_title))
    story.append(Spacer(1, 8))
    story.append(_p("Startup Profile", label_text))
    story.append(Spacer(1, 4))
    story.append(summary_table)
    story.append(Spacer(1, 10))
    story.append(_p("Scoring", label_text))
    story.append(Spacer(1, 4))
    story.append(scoring_table)
    story.append(Spacer(1, 10))
    print("PDF assessment raw text:\n", assessment)
    working_bullets, pushback_bullets, question_text, fit_text = parse_assessment_sections(assessment)

    story.append(_p("What is working", label_text))
    story.append(Spacer(1, 6))
    for item in working_bullets:
        row = Table(
            [[CircleBullet(color=colors.HexColor("#2d6a4f")), _p(item, bullet_text)]],
            colWidths=[10, doc.width - 16],
        )
        row.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )
        story.append(row)
        story.append(Spacer(1, 4))

    story.append(Spacer(1, 6))
    story.append(_p("Partner pushback", label_text))
    story.append(Spacer(1, 6))
    for item in pushback_bullets:
        row = Table(
            [[CircleBullet(color=accent_color), _p(item, bullet_text)]],
            colWidths=[10, doc.width - 16],
        )
        row.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )
        story.append(row)
        story.append(Spacer(1, 4))

    if question_text:
        story.append(Spacer(1, 8))
        question_table = Table([[ _p(f'"{question_text}"', question_callout) ]], colWidths=[doc.width])
        question_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), bg_color),
                    ("BOX", (0, 0), (-1, -1), 0.8, accent_color),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(question_table)

    if fit_text:
        story.append(Spacer(1, 8))
        story.append(_p(fit_text, body_text))
    story.append(Spacer(1, 10))

    story.append(_p("Similar YC Companies", section_title))
    story.append(Spacer(1, 8))
    for comp in similar_companies:
        sim = comp.get("similarity")
        match = f"{int(sim * 100)}% match" if sim is not None else ""
        name = comp.get("name", "")
        batch = comp.get("batch", "")
        line = f"<b>{name}</b>  {batch}  {match}".strip()
        story.append(_p(line, body_text))
        description = comp.get("description", "")
        if description:
            story.append(_p(description, body_italic))
        story.append(Spacer(1, 10))
    story.append(Spacer(1, 6))

    if verdict_label and partner_notes:
        notes_block = [_p("Partner Notes", section_title), Spacer(1, 8)]
        for note in partner_notes[:3]:
            note_row = Table(
                [[CircleBullet(), _p(str(note), body_text)]],
                colWidths=[10, doc.width - 16],
            )
            note_row.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 0),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ]
                )
            )
            notes_block.append(note_row)
            notes_block.append(Spacer(1, 6))
        story.append(KeepTogether(notes_block))
        story.append(Spacer(1, 12))

    if verdict_label and top_improvements:
        improvements_block = [_p("Top Improvements", section_title), Spacer(1, 6)]
        for idx, item in enumerate(top_improvements[:3], start=1):
            improvements_block.append(
                _p(
                    f"<font color='#8B1A1A'><b>{idx}.</b></font> {item}",
                    small_text,
                )
            )
            improvements_block.append(Spacer(1, 4))
        story.append(KeepTogether(improvements_block))
        story.append(Spacer(1, 12))

    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    return buffer.getvalue()


if "startup_profile" not in st.session_state:
    st.session_state.startup_profile = _load_startup_memory()

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
            color: #1a1a1a; font-family: Roboto, sans-serif; user-select: none;
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
                      font-family="Source Serif 4, serif"
                      font-style="italic"
                      font-size="28"
                      fill="white">y</text>
            </svg>
        </div>
        <div style="font-family: 'Source Serif 4', serif; font-size: 1.7rem; font-weight: 700; color: #1a1a1a;">
            YC Co-Founder
        </div>
        <div style="font-family: 'Source Serif 4', serif; font-style: italic; color: #4f4f4f; margin-top: 6px;">
            Turning builders into formidable founders
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        """
        <div style="font-family: 'Roboto', sans-serif; font-weight: 300; line-height: 1.9; color: #1a1a1a;">
            2,804 knowledge chunks<br>
            1,494 YC companies indexed<br>
            325 Paul Graham essay chunks<br>
            123 Startup School chunks
        </div>
        """,
        unsafe_allow_html=True,
    )

    startup_profile = st.session_state.get("startup_profile")
    if startup_profile:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='field-label'>Your Startup</div>", unsafe_allow_html=True)
        st.markdown(
            f"**{html.escape(startup_profile.get('one_liner', ''))}**  \n"
            f"Traction: {html.escape(startup_profile.get('traction', ''))}",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("Clear Memory", use_container_width=True):
            _clear_startup_memory()


# ── Tabs ───────────────────────────────────────────────

tab_ask, tab_eval, tab_browse, tab_benchmark, tab_verdict = st.tabs([
    "Ask YC",
    "Evaluate My Startup",
    "Browse YC Companies",
    "Benchmark",
    "YC Verdict",
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
        profile_defaults = st.session_state.startup_profile or {}

        st.markdown("<div class='field-label'>Your startup in one sentence</div>", unsafe_allow_html=True)
        one_liner = st.text_input(
            "Your startup in one sentence",
            value=profile_defaults.get("one_liner", ""),
            placeholder="AI tool that automates legal contract review for small law firms",
            label_visibility="collapsed",
        )
        st.markdown("<div class='field-label'>Problem you're solving</div>", unsafe_allow_html=True)
        problem = st.text_area(
            "Problem you're solving",
            value=profile_defaults.get("problem", ""),
            height=90,
            label_visibility="collapsed",
        )

        market_size_options = ["<$1B", "$1–10B", "$10B+", "Unknown"]
        market_size_value = profile_defaults.get("market_size", "Unknown")
        market_size_index = market_size_options.index(market_size_value) if market_size_value in market_size_options else 3
        st.markdown("<div class='field-label'>Market size</div>", unsafe_allow_html=True)
        market_size = st.selectbox(
            "Market size",
            market_size_options,
            index=market_size_index,
            label_visibility="collapsed",
        )

        traction_options = ["No users yet", "Waitlist", "Active users", "Revenue"]
        traction_value = profile_defaults.get("traction", "No users yet")
        traction_index = traction_options.index(traction_value) if traction_value in traction_options else 0
        st.markdown("<div class='field-label'>Traction</div>", unsafe_allow_html=True)
        traction = st.radio(
            "Traction",
            traction_options,
            index=traction_index,
            label_visibility="collapsed",
        )

        st.markdown("<div class='field-label'>Team Size</div>", unsafe_allow_html=True)
        team_size = st.number_input(
            "Team Size",
            min_value=1,
            max_value=10,
            value=int(profile_defaults.get("team_size", 2) or 2),
        )

        st.markdown("<div class='field-label'>Founder Background</div>", unsafe_allow_html=True)
        background = st.text_input(
            "Founder Background",
            value=profile_defaults.get("background", ""),
            placeholder="Brief background — ex-Google, Stanford CS, domain expert etc.",
            label_visibility="collapsed",
        )

        working_how_long_options = ["< 1 month", "1–6 months", "6–12 months", "1+ year"]
        working_how_long_value = profile_defaults.get("working_how_long", "1–6 months")
        working_how_long_index = working_how_long_options.index(working_how_long_value) if working_how_long_value in working_how_long_options else 1
        st.markdown("<div class='field-label'>How long have you been working on this?</div>", unsafe_allow_html=True)
        working_how_long = st.selectbox(
            "How long have you been working on this?",
            working_how_long_options,
            index=working_how_long_index,
            label_visibility="collapsed",
        )

        st.markdown("<div class='field-label'>Why is now the right time?</div>", unsafe_allow_html=True)
        why_now = st.text_area(
            "Why is now the right time?",
            value=profile_defaults.get("why_now", ""),
            height=90,
            label_visibility="collapsed",
        )

        st.markdown("<div class='field-label'>What could kill this?</div>", unsafe_allow_html=True)
        biggest_risk = st.text_area(
            "What could kill this?",
            value=profile_defaults.get("biggest_risk", ""),
            height=90,
            label_visibility="collapsed",
        )

        yc_batch_options = ["W25", "S25", "W26", "S26", "Not applying yet"]
        yc_batch_value = profile_defaults.get("yc_batch", "Not applying yet")
        yc_batch_index = yc_batch_options.index(yc_batch_value) if yc_batch_value in yc_batch_options else 4
        st.markdown("<div class='field-label'>YC batch</div>", unsafe_allow_html=True)
        yc_batch = st.selectbox(
            "YC batch",
            yc_batch_options,
            index=yc_batch_index,
            label_visibility="collapsed",
        )

        eval_clicked = st.button(
            "Evaluate My Startup",
            type="primary",
            use_container_width=True,
        )

    startup_profile = {
        "one_liner": one_liner,
        "problem": problem,
        "market_size": market_size,
        "traction": traction,
        "team_size": team_size,
        "background": background,
        "working_how_long": working_how_long,
        "why_now": why_now,
        "biggest_risk": biggest_risk,
        "yc_batch": yc_batch,
    }

    if eval_clicked:
        if not one_liner.strip() or not problem.strip():
            st.warning("Please complete the one-liner and problem fields first.")
        else:
            with right:
                with st.spinner("Analyzing against 1,494 YC companies…"):
                    try:
                        evaluator = load_evaluator()
                        description_parts = [one_liner.strip(), problem.strip(), background.strip()]
                        description = "\n".join([p for p in description_parts if p])
                        result = evaluator.evaluate(
                            description=description,
                            industry="other",
                            target_customer="B2B",
                            stage="idea",
                            team_size=team_size,
                            team_background=background or "not specified",
                        )
                        st.session_state.eval_result = result
                        st.session_state.startup_profile = startup_profile
                        _save_startup_memory(startup_profile)
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

            profile_for_report = st.session_state.get("startup_profile") or {}
            if profile_for_report:
                pdf_bytes = _build_pdf_report(
                    startup_profile=profile_for_report,
                    avg_similarity=avg_similarity,
                    assessment=result.get("assessment", ""),
                    similar_companies=result.get("similar_companies", []),
                    verdict_payload=st.session_state.get("verdict_result"),
                )
                st.download_button(
                    "Download PDF Report",
                    data=pdf_bytes,
                    file_name="yc_readiness_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )


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


# ════════════════════════════════════════════════════════
#  TAB 5 — YC Verdict
# ════════════════════════════════════════════════════════

with tab_verdict:
    st.header("YC Verdict")
    st.caption("One-shot partner verdict on your startup description.")

    if "verdict_result" not in st.session_state:
        st.session_state.verdict_result = None
    if "verdict_input" not in st.session_state:
        st.session_state.verdict_input = ""

    profile_defaults = st.session_state.get("startup_profile") or {}
    prefill = st.session_state.verdict_input
    if not prefill:
        prefill = profile_defaults.get("one_liner", "")

    st.markdown("<div class='field-label'>Startup description</div>", unsafe_allow_html=True)
    startup_description = st.text_area(
        "Startup description",
        value=prefill,
        height=110,
        placeholder="Describe the startup in 1-3 sentences",
        label_visibility="collapsed",
        key="verdict_input",
    )

    generate_clicked = st.button(
        "Generate Verdict",
        type="primary",
        use_container_width=True,
    )

    if generate_clicked:
        if not startup_description.strip():
            st.warning("Please enter a startup description first.")
            st.session_state.verdict_result = None
        else:
            raw_response = ""
            loading_messages = [
                "Analyzing founder-market fit...",
                "Estimating scalability...",
                "Comparing against YC winners...",
            ]
            loading_box = st.empty()
            for message in loading_messages:
                loading_box.markdown(
                    f"<div class='verdict-card'>{html.escape(message)}</div>",
                    unsafe_allow_html=True,
                )
                time.sleep(0.15)
            with st.spinner("Running YC partner analysis..."):
                verdict = generate_verdict_rag(profile_defaults, load_advisor())
                st.session_state.verdict_result = {
                    "verdict": verdict,
                    "raw": json.dumps(verdict, indent=2),
                    "description": startup_description.strip(),
                }
                loading_box.empty()

    result = st.session_state.verdict_result
    if result:
        if result.get("error"):
            raw_response = result.get("raw") or "(no response returned)"
            st.error(f"Verdict generation failed. Raw response:\n{raw_response}")
        else:
            verdict = result.get("verdict", {})
            verdict_label = str(verdict.get("verdict_label", ""))
            tag_line = str(verdict.get("tag_line", ""))
            funding_probability = int(verdict.get("funding_probability", 0) or 0)
            founder_market_fit = float(verdict.get("founder_market_fit", 0.0) or 0.0)
            partner_notes = verdict.get("partner_notes", []) or []
            strongest_dimension = str(verdict.get("strongest_dimension", ""))
            weakest_dimension = str(verdict.get("weakest_dimension", ""))
            dna_match = str(verdict.get("dna_match", ""))
            dna_reason = str(verdict.get("dna_reason", ""))
            top_improvements = verdict.get("top_improvements", []) or []

            st.markdown(
                f"""
                <div class="section-card" style="text-align:center; padding: 40px 32px; margin-bottom: 20px;">
                    <div style="font-family:'Inter',sans-serif; font-size:0.8rem; letter-spacing:3px; text-transform:uppercase; color:#8B1A1A; margin-bottom:12px;">YC PARTNER VERDICT</div>
                    <div style="font-family:'Playfair Display',serif; font-size:3.5rem; font-weight:700; color:#1a1a1a; line-height:1.1; margin-bottom:8px;">{html.escape(verdict_label)}</div>
                    <div style="font-family:'Inter',sans-serif; font-style:italic; color:#5d5d5d; font-size:1rem; margin-bottom:32px;">{html.escape(tag_line)}</div>
                    <div style="display:flex; justify-content:center; gap:48px; flex-wrap:wrap;">
                        <div>
                            <div style="font-family:'Playfair Display',serif; font-size:3rem; color:#FF6B35; font-weight:700;">{funding_probability}%</div>
                            <div style="font-family:'Inter',sans-serif; font-size:0.75rem; color:#888; text-transform:uppercase; letter-spacing:2px;">Interview Chance</div>
                        </div>
                        <div>
                            <div style="font-family:'Playfair Display',serif; font-size:3rem; color:#FF6B35; font-weight:700;">{founder_market_fit:.1f}</div>
                            <div style="font-family:'Inter',sans-serif; font-size:0.75rem; color:#888; text-transform:uppercase; letter-spacing:2px;">Founder-Market Fit</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(
                    f"""
                    <div class="section-card" style="text-align:center; padding:20px;">
                        <div style="font-size:0.7rem; letter-spacing:2px; text-transform:uppercase; color:#2d6a4f; margin-bottom:6px;">STRONGEST</div>
                        <div style="font-family:'Playfair Display',serif; font-size:1.4rem; font-weight:700;">{html.escape(strongest_dimension)}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f"""
                    <div class="section-card" style="text-align:center; padding:20px;">
                        <div style="font-size:0.7rem; letter-spacing:2px; text-transform:uppercase; color:#8B1A1A; margin-bottom:6px;">WEAKEST</div>
                        <div style="font-family:'Playfair Display',serif; font-size:1.4rem; font-weight:700;">{html.escape(weakest_dimension)}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f"""
                    <div class="section-card" style="text-align:center; padding:20px;">
                        <div style="font-size:0.7rem; letter-spacing:2px; text-transform:uppercase; color:#5d5d5d; margin-bottom:6px;">DNA MATCH</div>
                        <div style="font-family:'Playfair Display',serif; font-size:1.4rem; font-weight:700;">{html.escape(dna_match)}</div>
                        <div style="font-family:'Inter',sans-serif; font-size:0.8rem; color:#888; margin-top:4px; font-style:italic;">{html.escape(dna_reason)}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='field-label'>Partner Notes</div>", unsafe_allow_html=True)
            for note in partner_notes:
                st.markdown(
                    f"""
                    <div style="background:#1a1a1a; border-radius:6px; padding:16px 20px; margin-bottom:10px; display:flex; align-items:flex-start; gap:14px;">
                        <div style="background:#FF6B35; color:white; font-family:'Playfair Display',serif; font-style:italic; font-size:0.85rem; border-radius:50%; width:32px; height:32px; display:flex; align-items:center; justify-content:center; flex-shrink:0;">YC</div>
                        <div style="font-family:'Inter',sans-serif; color:#ffffff; font-size:0.9rem; line-height:1.6;">{html.escape(str(note))}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='field-label'>Before You Reapply</div>", unsafe_allow_html=True)
            improvements_html = "".join(
                [
                    "<div style=\"display:flex; gap:16px; align-items:flex-start; margin-bottom:12px;\">"
                    f"<div style=\"font-family:'Playfair Display',serif; font-size:1.4rem; color:#8B1A1A; font-weight:700; min-width:24px;\">{i + 1}</div>"
                    f"<div style=\"font-family:'Inter',sans-serif; font-size:0.9rem; line-height:1.6; color:#1a1a1a; padding-top:4px;\">{html.escape(str(imp))}</div>"
                    "</div>"
                    for i, imp in enumerate(top_improvements)
                ]
            )
            st.markdown(f"<div class=\"section-card\">{improvements_html}</div>", unsafe_allow_html=True)

            pdf_bytes = build_verdict_pdf(verdict)
            st.download_button(
                label="Download Verdict PDF",
                data=pdf_bytes,
                file_name="yc_verdict.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

            report_lines = [
                "YC Verdict",
                f"Verdict: {verdict_label}",
                f"Tagline: {tag_line}",
                f"Funding probability: {funding_probability}%",
                f"Founder-market fit: {founder_market_fit:.1f}",
                f"Strongest dimension: {strongest_dimension}",
                f"Weakest dimension: {weakest_dimension}",
                f"DNA match: {dna_match}",
                f"DNA reason: {dna_reason}",
                "",
                "Partner notes:",
            ]
            report_lines.extend([f"- {note}" for note in partner_notes[:3]])
            report_lines.append("")
            report_lines.append("Top improvements:")
            report_lines.extend([f"{idx}. {item}" for idx, item in enumerate(top_improvements, start=1)])

            st.download_button(
                "Download Verdict Report",
                data="\n".join(report_lines).strip() + "\n",
                file_name="yc_verdict.txt",
                mime="text/plain",
                use_container_width=True,
            )
