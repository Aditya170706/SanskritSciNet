"""
ui/app.py
─────────
Module 2 — Streamlit Web Interface

Full-featured UI for the Vedic Shloka Intelligence System.
Communicates with the FastAPI backend via HTTP.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from app_config.settings import API_URL

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "Vedic Shloka Intelligence",
    page_icon  = "🕉️",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# Styles
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem; font-weight: 700;
        color: #FF6B35; text-align: center; margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.0rem; color: #888; text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        background: #1e1e2e; border-radius: 12px;
        padding: 1.2rem; margin: 0.5rem 0;
        border-left: 4px solid #FF6B35;
    }
    .label { font-size: 0.78rem; color: #aaa; text-transform: uppercase;
             letter-spacing: 0.08em; }
    .value { font-size: 1.15rem; font-weight: 600; color: #fff; }
    .math-badge-true  { background:#22c55e; color:#fff; padding:2px 10px;
                        border-radius:20px; font-size:0.8rem; }
    .math-badge-false { background:#64748b; color:#fff; padding:2px 10px;
                        border-radius:20px; font-size:0.8rem; }
    .trace-step { padding: 0.3rem 0; border-bottom: 1px solid #333;
                  font-size: 0.9rem; color: #cdd6f4; }
    .formula-box { background:#2a1a4e; border-radius:8px; padding:1rem;
                   border:1px solid #7c3aed; }
    .conf-bar-label { font-size:0.75rem; color:#aaa; }
    div[data-testid="stMetricValue"] { font-size: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _call_api(shloka: str, timeout: int = 30) -> dict:
    """Send POST /analyze to FastAPI backend."""
    resp = requests.post(
        f"{API_URL}/analyze",
        json    = {"shloka": shloka},
        timeout = timeout,
    )
    resp.raise_for_status()
    return resp.json()


def _check_health() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.json() if r.ok else None
    except Exception:
        return None


def _get_examples() -> list[dict]:
    try:
        r = requests.get(f"{API_URL}/examples", timeout=3)
        return r.json().get("examples", []) if r.ok else []
    except Exception:
        return []


def _conf_color(conf: float) -> str:
    if conf >= 0.70: return "#22c55e"
    if conf >= 0.45: return "#f59e0b"
    return "#ef4444"


def _render_confidence_bar(label: str, value: float):
    pct   = int(value * 100)
    color = _conf_color(value)
    st.markdown(
        f'<div class="conf-bar-label">{label}</div>'
        f'<div style="background:#333;border-radius:4px;height:8px;margin-bottom:6px">'
        f'<div style="background:{color};width:{pct}%;height:8px;border-radius:4px"></div>'
        f'</div>'
        f'<div style="text-align:right;font-size:0.75rem;color:{color};margin-top:-4px">'
        f'{value:.3f}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🕉️ About")
    st.markdown(
        "**Vedic Shloka Intelligence System**\n\n"
        "Analyses Sanskrit shlokas using:\n"
        "- Transfer learning (IndicBERT)\n"
        "- Hybrid retrieval engine\n"
        "- Knowledge graph queries\n"
        "- Symbolic math reasoning\n"
    )

    st.divider()
    st.markdown("### ⚡ API Status")
    health = _check_health()
    if health:
        st.success(f"API running  •  v{health.get('version','?')}")
        if health.get("pipeline_ready"):
            st.success("Pipeline ready ✓")
        else:
            st.warning("Pipeline loading…")
        st.caption(f"Uptime: {health.get('uptime_seconds',0):.0f}s")
    else:
        st.error("API offline — start the API first:\n```\npython api/main.py\n```")

    st.divider()
    st.markdown("### 📚 Example Shlokas")
    examples = _get_examples()
    selected_example = None
    for ex in examples:
        if st.button(f"🔹 {ex['label']}", use_container_width=True, key=ex['label']):
            selected_example = ex["shloka"]

    st.divider()
    st.markdown("### ⚙️ Settings")
    show_debug = st.checkbox("Show debug info", value=False)
    show_raw   = st.checkbox("Show raw JSON", value=False)


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="main-title">🕉️ Vedic Shloka Intelligence System</div>',
            unsafe_allow_html=True)
st.markdown('<div class="subtitle">Mathematical Knowledge Extraction from Ancient Sanskrit Texts</div>',
            unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Input section
# ─────────────────────────────────────────────────────────────────────────────

default_text = selected_example if selected_example else ""

col_input, col_btn = st.columns([5, 1])
with col_input:
    shloka_input = st.text_area(
        "Enter Sanskrit Shloka",
        value       = default_text,
        height      = 100,
        placeholder = "समचतुरश्रस्य क्षेत्रफलं भुजवर्गः।",
        label_visibility = "collapsed",
    )

with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("🔍 Analyze", type="primary",
                            use_container_width=True)

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────────────

if analyze_btn:
    if not shloka_input.strip():
        st.error("⚠️ Please enter a Sanskrit shloka before clicking Analyze.")
        st.stop()

    if not health or not health.get("pipeline_ready"):
        st.error("⚠️ API is offline or pipeline is not ready. "
                 "Start the API with: `python api/main.py`")
        st.stop()

    with st.spinner("Analysing shloka…"):
        t0 = time.time()
        try:
            result = _call_api(shloka_input.strip())
            elapsed = time.time() - t0
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Run: `python api/main.py`")
            st.stop()
        except requests.exceptions.Timeout:
            st.error("Request timed out. The pipeline may be loading.")
            st.stop()
        except requests.exceptions.HTTPError as e:
            detail = ""
            try:
                detail = e.response.json().get("detail", "")
            except Exception:
                pass
            st.error(f"API error: {e}  {detail}")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.stop()

    # ── Results display ────────────────────────────────────────────────────

    st.markdown("## 📊 Analysis Results")

    # Row 1 — top-level classifications
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="result-card">'
                    f'<div class="label">Veda</div>'
                    f'<div class="value">{result.get("veda","Unknown")}</div>'
                    '</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="result-card">'
                    f'<div class="label">Domain</div>'
                    f'<div class="value">{result.get("domain","Unknown")}</div>'
                    '</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="result-card">'
                    f'<div class="label">Branch</div>'
                    f'<div class="value">{result.get("branch","Unknown")}</div>'
                    '</div>', unsafe_allow_html=True)
    with c4:
        math_flag = result.get("math", False)
        badge     = "math-badge-true" if math_flag else "math-badge-false"
        badge_txt = "✓ Mathematical" if math_flag else "✗ Non-mathematical"
        st.markdown('<div class="result-card">'
                    f'<div class="label">Content Type</div>'
                    f'<span class="{badge}">{badge_txt}</span>'
                    '</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2 — formula + confidence
    col_f, col_c = st.columns([3, 2])

    with col_f:
        formula = result.get("formula")
        if formula:
            st.markdown("### 📐 Extracted Formula")
            st.markdown(
                f'<div class="formula-box">'
                f'<div class="label">Name</div>'
                f'<div class="value" style="color:#a78bfa">{formula.get("name","")}</div>'
                f'<br><div class="label">Expression</div>'
                f'<code style="font-size:1.1rem;color:#34d399">{formula.get("expr","")}</code>'
                f'<br><br><div class="label">LaTeX</div>'
                f'<code style="color:#93c5fd">{formula.get("latex","")}</code>'
                f'<br><br><div class="label">Source</div>'
                f'<div style="color:#fbbf24">{formula.get("source","")}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown("### 📐 Formula")
            st.info("No mathematical formula detected in this shloka.")

    with col_c:
        st.markdown("### 🎯 Confidence")
        overall_conf = result.get("confidence", 0.0)
        color        = _conf_color(overall_conf)
        st.markdown(
            f'<div style="font-size:3rem;font-weight:700;color:{color};'
            f'text-align:center">{overall_conf:.3f}</div>'
            f'<div style="text-align:center;color:#aaa;font-size:0.85rem">'
            f'Composite score</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        cb = result.get("confidence_breakdown", {})
        _render_confidence_bar("Model",     cb.get("model",     0))
        _render_confidence_bar("Retrieval", cb.get("retrieval", 0))
        _render_confidence_bar("Graph",     cb.get("graph",     0))

    st.divider()

    # Row 3 — reasoning trace
    st.markdown("### 🧠 Reasoning Trace")
    reasoning = result.get("reasoning", [])
    if reasoning:
        for i, step in enumerate(reasoning, 1):
            st.markdown(
                f'<div class="trace-step">'
                f'<span style="color:#FF6B35;font-weight:600">{i}.</span> {step}'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("No reasoning trace available.")

    # Row 4 — retrieved shlokas
    retrieved = result.get("top_retrieved", [])
    if retrieved:
        st.divider()
        st.markdown("### 🔍 Top Retrieved Shlokas")
        for r in retrieved:
            score = r.get("final_score", 0)
            score_color = _conf_color(score)
            st.markdown(
                f'<div style="background:#1e1e2e;border-radius:8px;padding:0.8rem;'
                f'margin:0.4rem 0;border-left:3px solid {score_color}">'
                f'<span style="color:#aaa;font-size:0.75rem">{r.get("id","")} '
                f'• {r.get("source","")} '
                f'• score=<span style="color:{score_color}">{score:.3f}</span></span><br>'
                f'<span style="color:#e2e8f0">{r.get("text","")[:80]}…</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Footer stats
    st.divider()
    fc1, fc2, fc3 = st.columns(3)
    fc1.metric("Response time", f"{elapsed*1000:.0f} ms")
    fc2.metric("Cache",    "HIT ✓" if result.get("cache_hit") else "MISS")
    fc3.metric("Pipeline latency", f"{result.get('elapsed_ms', 0):.0f} ms")

    # Debug
    if show_debug:
        with st.expander("🔧 Debug — Phase 2 Predictions"):
            p2 = result.get("phase2_predictions", {})
            for task, info in p2.items():
                st.write(f"**{task}**: {info.get('label')} (conf={info.get('conf',0):.3f})")

    if show_raw:
        with st.expander("📄 Raw JSON Response"):
            st.json(result)

else:
    # Landing state
    st.markdown("""
    <div style="text-align:center;padding:3rem;color:#666">
        <div style="font-size:4rem">🕉️</div>
        <div style="font-size:1.1rem;margin-top:1rem">
            Enter a Sanskrit shloka above and click <strong>Analyze</strong>
        </div>
        <div style="font-size:0.9rem;margin-top:0.5rem;color:#555">
            Or select an example from the sidebar
        </div>
    </div>
    """, unsafe_allow_html=True)
