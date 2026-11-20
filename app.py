from __future__ import annotations

import hashlib
import hmac
import html
import json
import time
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ───────────────────────── App configuration ─────────────────────────
st.set_page_config(
    page_title="PulseOps | Customer Intelligence",
    page_icon="assets/pulseops-mark.svg",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    BASE_DIR / "data" / "cleaned_churn.csv",
    BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    BASE_DIR / "data" / "telco_customer_churn.csv",
]
MODEL_PATH = BASE_DIR / "models" / "churn_pipeline.joblib"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"

SESSION_TIMEOUT = 30 * 60      # sign out after 30 minutes of inactivity
MAX_ATTEMPTS = 5               # failed logins before temporary lock
LOCK_SECONDS = 60
PBKDF2_ITERATIONS = 120_000

ROLE_PAGES = {
    "Admin": ["Overview", "Predict churn", "Customer explorer", "Insights & analytics", "Model lab"],
    "Analyst": ["Overview", "Predict churn", "Customer explorer", "Insights & analytics"],
    "Viewer": ["Overview", "Customer explorer", "Insights & analytics"],
}
PAGE_LABELS = {
    "Overview": "▦   Overview",
    "Predict churn": "⚠   Predict churn",
    "Customer explorer": "☰   Customer explorer",
    "Insights & analytics": "↗   Insights & analytics",
    "Model lab": "⚙   Model lab",
}

# ───────────────────────── Styles ─────────────────────────────────────
BASE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root{--bg:#f5f6f7;--line:#e5e7eb;--text:#111827;--muted:#6b7280;--blue:#1d6fe8;--green:#16a34a;--red:#dc2626}
html,body,[class*="css"]{font-family:'Inter',sans-serif}
.stApp{background:var(--bg);color:var(--text)}
[data-testid="stHeader"]{background:transparent}
.block-container{max-width:1500px;padding:1rem 2rem 2.5rem}
h1,h2,h3,h4,h5{color:var(--text);font-family:'Inter',sans-serif}
[data-testid="stSidebar"]{background:#fff;border-right:1px solid var(--line);min-width:260px}
[data-testid="stSidebar"]>div:first-child{padding:1.1rem .9rem}
[data-testid="stSidebar"] [data-testid="stRadio"]>div{gap:4px}
[data-testid="stSidebar"] [data-testid="stRadio"] label{border-radius:9px;padding:10px 13px;width:100%;transition:background .15s}
[data-testid="stSidebar"] [data-testid="stRadio"] label>div:first-child{display:none}
[data-testid="stSidebar"] [data-testid="stRadio"] label p{color:#1f2937;font-weight:500;font-size:14px}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover{background:#f3f4f6}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked){background:var(--blue)}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p{color:#fff!important}
[data-testid="stSidebar"] div.stButton>button{background:#fff;color:#1f2937;border:1px solid #d1d5db}
[data-testid="stSidebar"] div.stButton>button:hover{background:#fef2f2;border-color:#fca5a5;color:#b91c1c}
.brand{display:flex;align-items:center;gap:11px;padding:2px 4px 16px;border-bottom:1px solid var(--line);margin-bottom:16px}
.brand-mark{width:38px;height:38px;border-radius:10px;background:var(--blue);display:flex;align-items:center;justify-content:center;color:#fff}
.brand-name{color:#111827;font-size:16px;font-weight:600;line-height:1.2}
.brand-sub{font-size:12px;color:var(--muted)}
.user-box{background:#f3f4f6;border-radius:10px;padding:12px 14px;margin:14px 0 10px;display:flex;align-items:center;gap:10px}
.user-box .avatar{width:34px;height:34px;font-size:12px}
.user-box b{display:block;font-size:13px;font-weight:600;color:#111827}
.user-box span{font-size:12px;color:var(--muted)}
.status-dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:#22c55e;margin-right:6px}
.topbar{display:flex;align-items:center;justify-content:space-between;gap:16px;background:#fff;border:1px solid var(--line);border-radius:12px;padding:10px 16px;margin-bottom:26px}
.search{display:flex;align-items:center;gap:10px;background:#f3f4f6;border:1px solid var(--line);border-radius:9px;padding:9px 14px;width:min(380px,100%);color:#9ca3af;font-size:14px}
.top-right{display:flex;align-items:center;gap:14px}
.bell{position:relative;color:#374151;display:flex}
.bell i{position:absolute;top:-2px;right:-1px;width:8px;height:8px;border-radius:50%;background:#ef4444;border:2px solid #fff}
.vline{width:1px;height:32px;background:var(--line)}
.avatar{width:36px;height:36px;border-radius:50%;background:var(--blue);color:#fff;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:600;flex-shrink:0}
.uname{font-size:14px;font-weight:500;color:#111827;line-height:1.2}
.umail{font-size:12px;color:var(--muted)}
.page-title{font-size:28px;line-height:1.2;font-weight:700;color:#111827;margin:0;letter-spacing:-.4px}
.page-subtitle{color:var(--muted);font-size:15px;margin-top:6px;margin-bottom:22px}
.section-head{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin-bottom:12px}
.section-title{color:#111827;font-size:18px;font-weight:600;letter-spacing:-.2px}
.section-note{color:var(--muted);font-size:14px;margin-top:3px}
.pill{display:inline-block;border:1px solid var(--line);background:#f9fafb;color:#4b5563;border-radius:30px;padding:3px 10px;font-size:11px;font-weight:500}
.mcard{background:#fff;border:1px solid var(--line);border-radius:14px;padding:22px 24px;box-shadow:0 1px 3px rgba(17,24,39,.06);min-height:132px}
.mrow{display:flex;justify-content:space-between;align-items:flex-start}
.mlabel{font-size:14px;color:#4b5563}
.mvalue{font-size:28px;font-weight:700;color:#111827;margin-top:8px;letter-spacing:-.5px}
.micon{width:48px;height:48px;border-radius:10px;display:flex;align-items:center;justify-content:center}
.micon.green{background:#ecfdf3;color:var(--green)}.micon.red{background:#fef2f2;color:var(--red)}.micon.blue{background:#eff6ff;color:var(--blue)}
.mfoot{font-size:12px;margin-top:10px;font-weight:500}
.mfoot.green{color:var(--green)}.mfoot.red{color:var(--red)}.mfoot.blue{color:var(--blue)}
div[data-testid="stVerticalBlockBorderWrapper"]{background:#fff;border:1px solid var(--line)!important;border-radius:14px!important;padding:14px 18px 10px;box-shadow:0 1px 3px rgba(17,24,39,.06)}
.panel{background:#fff;border:1px solid var(--line);border-radius:14px;padding:18px 22px;margin-bottom:16px;box-shadow:0 1px 3px rgba(17,24,39,.06)}
.panel-sub{color:var(--muted);font-size:14px;margin-top:4px}
.qs{display:flex;align-items:center;gap:12px;background:#f3f6fa;border-radius:10px;padding:13px 15px;margin-bottom:12px}
.qs-ic{color:#2563eb;display:flex}.qs-t{flex:1}
.qs-t b{display:block;font-size:14px;font-weight:500;color:#111827}
.qs-t span{font-size:12px;color:var(--muted)}
.qs-v{font-size:22px;font-weight:700;color:#111827}
div[data-testid="stMetric"]{background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px 16px;box-shadow:0 1px 3px rgba(17,24,39,.05)}
div[data-testid="stMetric"] label{color:var(--muted)!important}
div[data-testid="stMetric"] [data-testid="stMetricValue"]{color:#111827!important;font-weight:700}
div.stButton>button,div.stFormSubmitButton>button{background:var(--blue);color:#fff;border:1px solid var(--blue);border-radius:9px;min-height:42px;font-weight:600}
div.stButton>button:hover,div.stFormSubmitButton>button:hover{background:#1558c0;border-color:#1558c0;color:#fff}
div.stDownloadButton>button{background:#fff;color:#1f2937;border:1px solid #d1d5db;border-radius:9px}
div.stDownloadButton>button:hover{border-color:var(--blue);color:var(--blue)}
[data-baseweb="select"]>div,[data-baseweb="input"]>div{background:#fff;border-color:#d1d5db;border-radius:8px}
[data-baseweb="select"] *,[data-baseweb="input"] input{color:#111827!important}
[data-baseweb="popover"],[data-baseweb="menu"],[role="listbox"],[role="option"]{background:#fff!important;color:#111827!important}
[role="option"]:hover{background:#f3f4f6!important}
.stTextInput input,.stNumberInput input{background:#fff!important;color:#111827!important}
[data-testid="stForm"]{background:#fff;border:1px solid var(--line);border-radius:14px;box-shadow:0 1px 3px rgba(17,24,39,.06)}
.stMarkdown,.stMarkdown p,.stMarkdown li{color:#1f2937}
[data-testid="stCaptionContainer"]{color:var(--muted)!important}
div[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:10px;overflow:hidden}
hr{border-color:var(--line)!important}
.result-high{display:inline-block;background:#fef2f2;color:#dc2626;padding:6px 11px;border-radius:7px;border:1px solid #fecaca;font-size:12px;font-weight:600}
.result-low{display:inline-block;background:#f0fdf4;color:#16a34a;padding:6px 11px;border-radius:7px;border:1px solid #bbf7d0;font-size:12px;font-weight:600}
.footer{border-top:1px solid var(--line);margin-top:30px;padding-top:14px;color:#9ca3af;font-size:12px}
@media(max-width:900px){.block-container{padding:1rem .9rem 2rem}.search,.uname,.umail,.vline{display:none}.mvalue{font-size:24px}.page-title{font-size:24px}}
</style>
"""

LOGIN_CSS = """
<style>
[data-testid="stSidebar"],[data-testid="collapsedControl"],[data-testid="stSidebarCollapsedControl"],[data-testid="stHeader"]{display:none!important}
.block-container{max-width:1100px;padding-top:4.5rem}
.login-hero{background:linear-gradient(150deg,#1d6fe8 0%,#1e40af 100%);border-radius:20px;padding:40px 38px;min-height:540px;color:#fff;display:flex;flex-direction:column;justify-content:space-between}
.login-hero .brand{border:0;margin:0;padding:0}
.login-hero .brand-name{color:#fff}.login-hero .brand-sub{color:#bfdbfe}
.login-hero .brand-mark{background:rgba(255,255,255,.18)}
.login-hero h2{color:#fff;font-size:32px;line-height:1.2;letter-spacing:-.8px;margin:34px 0 12px;font-weight:700}
.login-hero p{color:#dbeafe;font-size:15px;line-height:1.65;margin:0;max-width:420px}
.feat{display:flex;gap:12px;align-items:flex-start;margin-top:16px;color:#eff6ff;font-size:14px}
.feat i{flex-shrink:0;width:22px;height:22px;border-radius:50%;background:rgba(255,255,255,.2);display:flex;align-items:center;justify-content:center;font-style:normal;font-size:12px}
.login-note{font-size:12px;color:#bfdbfe}
.login-title{font-size:26px;font-weight:700;color:#111827;letter-spacing:-.5px;margin:20px 0 6px}
.login-sub{color:#6b7280;font-size:14px;margin-bottom:22px}
.demo-box{background:#fffbeb;border:1px solid #fde68a;color:#92400e;border-radius:10px;padding:10px 13px;font-size:12px;margin-top:14px;line-height:1.5}
</style>
"""

st.markdown(BASE_CSS, unsafe_allow_html=True)

# ───────────────────────── Icons & helpers ────────────────────────────
ICONS = {
    "grid": '<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/>',
    "users": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "dollar": '<line x1="12" y1="2" x2="12" y2="22"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>',
    "alert": '<path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "trend": '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
    "search": '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>',
    "bell": '<path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 0 1-3.46 0"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
}

def svg(name, size=22):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>')

BLUE, GREEN, RED = "#3b82f6", "#22c55e", "#ef4444"
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#6b7280", size=12),
    margin=dict(l=8, r=8, t=18, b=8),
    xaxis=dict(gridcolor="#eceef1", griddash="dot", zerolinecolor="#d1d5db", linecolor="#9ca3af"),
    yaxis=dict(gridcolor="#eceef1", griddash="dot", zerolinecolor="#d1d5db", linecolor="#9ca3af"),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
)

def style_fig(fig, height=300):
    fig.update_layout(**PLOT_LAYOUT, height=height)
    return fig

def section_title(title, subtitle=None, badge=None):
    badge_html = f'<span class="pill">{badge}</span>' if badge else ""
    subtitle_html = f'<div class="section-note">{subtitle}</div>' if subtitle else ""
    st.markdown(f'<div class="section-head"><div><div class="section-title">{title}</div>{subtitle_html}</div>{badge_html}</div>', unsafe_allow_html=True)

def metric_card(label, value, note, icon, tone):
    st.markdown(f"""<div class="mcard"><div class="mrow"><div><div class="mlabel">{label}</div><div class="mvalue">{value}</div></div>
    <div class="micon {tone}">{svg(icon)}</div></div><div class="mfoot {tone}">{note}</div></div>""", unsafe_allow_html=True)

def quick_stat(icon, title, sub, value):
    st.markdown(f'<div class="qs"><div class="qs-ic">{svg(icon, 20)}</div><div class="qs-t"><b>{title}</b><span>{sub}</span></div><div class="qs-v">{value}</div></div>', unsafe_allow_html=True)

def plotly_chart(fig, key=None):
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True}, key=key)

# ───────────────────────── Authentication ─────────────────────────────
def _hash_password(password: str, salt_hex: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), PBKDF2_ITERATIONS).hex()

DEMO_SALT = "a1b2c3d4e5f60718293a4b5c6d7e8f90"

@st.cache_resource(show_spinner=False)
def load_users():
    """Users come from .streamlit/secrets.toml. Falls back to a demo admin."""
    try:
        raw = st.secrets["users"]
        users = {str(k).lower(): dict(v) for k, v in raw.items()}
        if users:
            return users, False
    except Exception:
        pass
    demo = {"admin": {"name": "Admin User", "email": "admin@pulseops.com", "role": "Admin",
                      "salt": DEMO_SALT, "hash": _hash_password("admin123", DEMO_SALT)}}
    return demo, True

def authenticate(username: str, password: str):
    users, _ = load_users()
    user = users.get(username.strip().lower())
    salt = user["salt"] if user else DEMO_SALT
    candidate = _hash_password(password, salt)
    expected = user["hash"] if user else "0" * 64
    if user and hmac.compare_digest(candidate, expected):
        return user
    return None

def login_page():
    st.markdown(LOGIN_CSS, unsafe_allow_html=True)
    _, is_demo = load_users()
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.markdown(f"""
        <div class="login-hero">
          <div>
            <div class="brand"><div class="brand-mark">{svg("grid", 20)}</div>
              <div><div class="brand-name">PulseOps</div><div class="brand-sub">Churn Analytics</div></div></div>
            <h2>Spot at-risk customers before they leave.</h2>
            <p>One workspace for retention metrics, customer segments and churn predictions.</p>
            <div class="feat"><i>✓</i><span>Live retention and revenue-at-risk metrics</span></div>
            <div class="feat"><i>✓</i><span>Machine-learning churn probability per customer</span></div>
            <div class="feat"><i>✓</i><span>Role-based access for admins, analysts and viewers</span></div>
          </div>
          <div class="login-note">Secure sign-in · Sessions expire after 30 minutes of inactivity</div>
        </div>""", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="login-title">Welcome back</div><div class="login-sub">Sign in to your workspace to continue.</div>', unsafe_allow_html=True)
        locked_for = int(st.session_state.get("locked_until", 0) - time.time())
        with st.form("login_form", border=False):
            username = st.text_input("Username", placeholder="admin")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            submitted = st.form_submit_button("Sign in", use_container_width=True, disabled=locked_for > 0)
        if locked_for > 0:
            st.error(f"Too many failed attempts. Try again in {locked_for} seconds.")
        elif submitted:
            user = authenticate(username, password) if username and password else None
            if user:
                st.session_state.pop("fails", None)
                st.session_state["auth"] = {
                    "username": username.strip().lower(), "name": user.get("name", username),
                    "email": user.get("email", ""), "role": user.get("role", "Viewer").title(), "ts": time.time(),
                }
                st.rerun()
            else:
                fails = st.session_state.get("fails", 0) + 1
                st.session_state["fails"] = fails
                if fails >= MAX_ATTEMPTS:
                    st.session_state["locked_until"] = time.time() + LOCK_SECONDS
                    st.session_state["fails"] = 0
                    st.error(f"Too many failed attempts. Locked for {LOCK_SECONDS} seconds.")
                else:
                    st.error(f"Incorrect username or password. {MAX_ATTEMPTS - fails} attempts left.")
        if is_demo:
            st.markdown('<div class="demo-box"><b>Demo mode.</b> No users are configured, so the default login is <b>admin / admin123</b>. Add real users in <code>.streamlit/secrets.toml</code> before sharing this app.</div>', unsafe_allow_html=True)

def logout(message=None):
    st.session_state.pop("auth", None)
    if message:
        st.session_state["flash"] = message
    st.rerun()

if "auth" not in st.session_state:
    login_page()
    st.stop()

auth = st.session_state["auth"]
if time.time() - auth["ts"] > SESSION_TIMEOUT:
    logout()
auth["ts"] = time.time()
role = auth["role"] if auth["role"] in ROLE_PAGES else "Viewer"
initials = "".join(w[0] for w in auth["name"].split()[:2]).upper() or "U"

# ───────────────────────── Cached data layer ──────────────────────────
dataset_path = next((p for p in DATA_CANDIDATES if p.exists()), None)
if dataset_path is None:
    st.markdown('<h1 class="page-title">Your workspace is almost ready.</h1>', unsafe_allow_html=True)
    st.warning("Dataset file not found. Add the Telco Customer Churn CSV to the `data/` folder.")
    st.code("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    st.stop()

DATA_KEY = (str(dataset_path), dataset_path.stat().st_mtime)
MODEL_MTIME = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else 0.0

@st.cache_data(show_spinner="Loading data…")
def get_data(path: str, mtime: float) -> pd.DataFrame:
    data = pd.read_csv(path)
    if "Churn" not in data.columns:
        return data
    if data["Churn"].dtype == object:
        data["Churn"] = data["Churn"].astype(str).str.strip().map({"Yes": 1, "No": 0, "1": 1, "0": 0, "True": 1, "False": 0})
    data["Churn"] = pd.to_numeric(data["Churn"], errors="coerce")
    data = data.dropna(subset=["Churn"]).copy()
    data["Churn"] = data["Churn"].astype(int)
    for col in ["TotalCharges", "MonthlyCharges", "tenure"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data

@st.cache_resource(show_spinner=False)
def load_model(path: str, mtime: float):
    return joblib.load(path)

df = get_data(*DATA_KEY)
if "Churn" not in df.columns:
    st.error("Dataset must contain a `Churn` target column.")
    st.stop()
model = load_model(str(MODEL_PATH), MODEL_MTIME) if MODEL_PATH.exists() else None

def values_for_column(data, col, fallback):
    if col in data.columns:
        return sorted(data[col].dropna().unique().tolist(), key=str)
    return fallback

def clean_display_data(data):
    shown = data.copy()
    shown["Churn"] = shown["Churn"].map({0: "Retained", 1: "Churned"})
    return shown

def churn_probability(m, frame):
    classes = list(getattr(m, "classes_", []))
    proba = m.predict_proba(frame)
    idx = next((i for i, v in enumerate(classes) if str(v).lower() in ("1", "yes", "true")), proba.shape[1] - 1)
    return proba[:, idx]

@st.cache_data(show_spinner="Building dashboard…")
def overview_bundle(path: str, mtime: float, model_mtime: float):
    d = get_data(path, mtime)
    total, churned = len(d), int(d["Churn"].sum())
    has_m = "MonthlyCharges" in d.columns
    out = {
        "total": total, "churned": churned, "retained": total - churned,
        "churn_rate": churned / total * 100 if total else 0,
        "monthly_total": d["MonthlyCharges"].sum() if has_m else None,
        "risk_rev": d.loc[d["Churn"] == 1, "MonthlyCharges"].sum() if has_m else None,
        "avg_monthly": d["MonthlyCharges"].mean() if has_m else None,
        "avg_tenure": d["tenure"].mean() if "tenure" in d.columns else None,
    }
    out["retention_rate"] = 100 - out["churn_rate"]
    if "tenure" in d.columns:
        t = d.assign(**{"Tenure band": pd.cut(d["tenure"], bins=[-1, 6, 12, 24, 48, 72], labels=["0–6", "7–12", "13–24", "25–48", "49–72"])})
        band = t.groupby("Tenure band", observed=False)["Churn"].mean().mul(100).reset_index(name="Churn rate")
        fig = px.line(band, x="Tenure band", y="Churn rate", markers=True, line_shape="spline")
        fig.update_traces(line_color=BLUE, marker_color=BLUE, line_width=3, marker_size=9)
        fig.update_layout(yaxis_ticksuffix="%", xaxis_title="Tenure (months)", yaxis_title=None)
        out["trend"] = style_fig(fig, 340)
    status = pd.DataFrame({"Status": ["Retained", "Churned"], "Customers": [out["retained"], churned]})
    fig = px.pie(status, names="Status", values="Customers", color="Status", color_discrete_map={"Retained": GREEN, "Churned": RED})
    fig.update_traces(textinfo="label+percent", textposition="outside", textfont_size=14, marker=dict(line=dict(color="#ffffff", width=2)))
    fig.update_layout(showlegend=False)
    style_fig(fig, 340).update_layout(margin=dict(l=70, r=70, t=30, b=30))
    out["pie"] = fig
    if "Contract" in d.columns:
        c = d.groupby("Contract", observed=False)["Churn"].mean().mul(100).reset_index(name="Churn rate")
        c["Retention rate"] = 100 - c["Churn rate"]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=c["Contract"], y=c["Retention rate"], name="Retained", marker_color=GREEN))
        fig.add_trace(go.Bar(x=c["Contract"], y=c["Churn rate"], name="Churned", marker_color=RED))
        fig.update_layout(barmode="stack", yaxis=dict(range=[0, 100], ticksuffix="%", title=None), xaxis_title=None, legend=dict(orientation="h", y=-.2, x=0))
        out["contract"] = style_fig(fig, 340)
        top = c.sort_values("Churn rate", ascending=False).iloc[0]
        out["top_contract"] = (str(top["Contract"]), float(top["Churn rate"]))
    out["risk_table"] = None
    if model_mtime and model is not None:
        try:
            feats = [x for x in d.columns if x not in ("Churn", "customerID", "CustomerID")]
            scored = d.assign(**{"Risk score": churn_probability(model, d[feats]) * 100})
            active = scored[scored["Churn"] == 0].nlargest(8, "Risk score")
            cols = [x for x in ["customerID", "tenure", "Contract", "MonthlyCharges", "Risk score"] if x in active.columns]
            out["risk_table"] = active[cols].reset_index(drop=True)
        except Exception:
            out["risk_table"] = None
    return out

@st.cache_data(show_spinner="Building analytics…")
def analytics_bundle(path: str, mtime: float):
    d = get_data(path, mtime)
    figs = {}
    if "Contract" in d.columns:
        t = d.groupby("Contract", observed=False)["Churn"].mean().mul(100).reset_index(name="Churn rate")
        fig = px.bar(t, x="Contract", y="Churn rate", color="Churn rate", color_continuous_scale=["#93c5fd", "#2563eb"])
        fig.update_layout(coloraxis_showscale=False, yaxis_ticksuffix="%", yaxis_title=None, xaxis_title=None)
        figs["contract"] = style_fig(fig, 320)
    if "tenure" in d.columns:
        t = d.groupby("Churn")["tenure"].mean().rename(index={0: "Retained", 1: "Churned"}).rename_axis("Status").reset_index(name="Months")
        fig = px.bar(t, x="Status", y="Months", color="Status", color_discrete_map={"Retained": GREEN, "Churned": RED})
        fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title=None)
        figs["tenure"] = style_fig(fig, 320)
        tb = d.assign(**{"Tenure band": pd.cut(d["tenure"], bins=[-1, 6, 12, 24, 48, 72], labels=["0–6", "7–12", "13–24", "25–48", "49–72"])})
        band = tb.groupby("Tenure band", observed=False)["Churn"].mean().mul(100).reset_index(name="Churn rate")
        fig = px.line(band, x="Tenure band", y="Churn rate", markers=True, line_shape="spline")
        fig.update_traces(line_color=BLUE, marker_color=BLUE, line_width=3, marker_size=9)
        fig.update_layout(yaxis_ticksuffix="%", xaxis_title="Tenure (months)", yaxis_title=None)
        figs["bands"] = style_fig(fig, 320)
    if "MonthlyCharges" in d.columns:
        fig = px.histogram(d, x="MonthlyCharges", color=d["Churn"].map({0: "Retained", 1: "Churned"}), nbins=28, barmode="overlay",
                           opacity=.75, color_discrete_map={"Retained": GREEN, "Churned": RED})
        fig.update_layout(xaxis_title="Monthly charges ($)", yaxis_title="Customers", legend_title=None)
        figs["charges"] = style_fig(fig, 320)
    return figs

# ───────────────────────── Sidebar ────────────────────────────────────
allowed_pages = ROLE_PAGES[role]
with st.sidebar:
    st.markdown(f"""<div class="brand"><div class="brand-mark">{svg("grid", 20)}</div>
      <div><div class="brand-name">PulseOps</div><div class="brand-sub">Churn Analytics</div></div></div>""", unsafe_allow_html=True)
    page = st.radio("Navigation", allowed_pages, format_func=PAGE_LABELS.get, label_visibility="collapsed")
    st.write("")
    status_line = '<span class="status-dot"></span>Prediction model loaded' if model is not None else "○ Model not trained yet"
    st.markdown(f'<div class="user-box"><div class="avatar">{html.escape(initials)}</div><div><b>{html.escape(auth["name"])}</b><span>{role} · {status_line}</span></div></div>', unsafe_allow_html=True)
    if st.button("Sign out", use_container_width=True):
        logout()

# ───────────────────────── Top bar + title ────────────────────────────
st.markdown(f"""
<div class="topbar">
  <div class="search">{svg("search", 17)}<span>Search...</span></div>
  <div class="top-right">
    <div class="bell">{svg("bell", 20)}<i></i></div><div class="vline"></div>
    <div class="avatar">{html.escape(initials)}</div>
    <div><div class="uname">{html.escape(auth["name"])}</div><div class="umail">{html.escape(auth["email"] or role)}</div></div>
  </div>
</div>""", unsafe_allow_html=True)

titles = {
    "Overview": ("Dashboard Overview", "Monitor key metrics and customer insights"),
    "Predict churn": ("Churn Risk", "Estimate churn probability from a customer's account and service profile."),
    "Customer explorer": ("Customers", "Filter, inspect and export the customer records behind your retention metrics."),
    "Insights & analytics": ("Insights & Analytics", "Find the customer segments and service patterns associated with churn."),
    "Model lab": ("Model Lab", "Review the saved model, evaluation metrics, and training status."),
}
title, subtitle = titles[page]
st.markdown(f'<h1 class="page-title">{title}</h1><div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)

# ───────────────────────── Overview ───────────────────────────────────
if page == "Overview":
    b = overview_bundle(*DATA_KEY, MODEL_MTIME)
    money = lambda v: f"${v:,.0f}" if v is not None else "—"
    cards = [
        ("Total Customers", f"{b['total']:,}", f"{b['retained']:,} retained customers", "users", "green"),
        ("Monthly Revenue", money(b["monthly_total"]), "Recurring charges, all accounts", "dollar", "green"),
        ("Churned Customers", f"{b['churned']:,}", f"{b['churn_rate']:.1f}% churn rate · {money(b['risk_rev'])}/mo at risk", "alert", "red"),
        ("Avg Monthly Charge", f"${b['avg_monthly']:,.2f}" if b["avg_monthly"] is not None else "—", f"Retention rate {b['retention_rate']:.1f}%", "trend", "green"),
    ]
    for col, item in zip(st.columns(4, gap="medium"), cards):
        with col:
            metric_card(*item)
    st.write("")
    left, right = st.columns(2, gap="medium")
    with left:
        with st.container(border=True):
            section_title("Churn Trend", "Churn rate by customer lifetime")
            if "trend" in b: plotly_chart(b["trend"], "ov-trend")
            else: st.info("Tenure field isn't available in this dataset.")
    with right:
        with st.container(border=True):
            section_title("Customer Status", "Retained vs churned customers")
            plotly_chart(b["pie"], "ov-pie")
    st.write("")
    left, right = st.columns(2, gap="medium")
    with left:
        with st.container(border=True):
            section_title("Churn by Contract", "Retained vs churned share per contract type")
            if "contract" in b: plotly_chart(b["contract"], "ov-contract")
            else: st.info("Contract field isn't available in this dataset.")
    with right:
        with st.container(border=True):
            section_title("Quick Stats", "Key performance indicators")
            quick_stat("shield", "Retention Rate", "Customers who stayed", f"{b['retention_rate']:.1f}%")
            if b["avg_tenure"] is not None:
                quick_stat("clock", "Average Tenure", "Months with the provider", f"{b['avg_tenure']:.1f}")
            if b["risk_rev"] is not None:
                quick_stat("alert", "Revenue at Risk", "Monthly charges, churned accounts", money(b["risk_rev"]))
            if "top_contract" in b:
                quick_stat("trend", f"Highest churn: {html.escape(b['top_contract'][0])}", "Contract type", f"{b['top_contract'][1]:.1f}%")
    if b["risk_table"] is not None:
        st.write("")
        with st.container(border=True):
            section_title("Customers to contact first", "Active customers with the highest predicted churn risk", "Model output")
            st.dataframe(b["risk_table"], use_container_width=True, hide_index=True,
                         column_config={"Risk score": st.column_config.ProgressColumn("Churn risk", min_value=0, max_value=100, format="%.0f%%"),
                                        "MonthlyCharges": st.column_config.NumberColumn("Monthly ($)", format="$%.2f"),
                                        "tenure": st.column_config.NumberColumn("Tenure (mo)")})

# ───────────────────────── Prediction workspace ───────────────────────
elif page == "Predict churn":
    if model is None:
        st.error("Trained model not found. Run `python src/train.py` after adding the dataset.")
        st.stop()
    st.markdown('<div class="panel"><div class="section-title">Build a customer profile</div><div class="panel-sub">Select customer attributes. The saved pipeline will return a model-based estimate.</div></div>', unsafe_allow_html=True)
    values = lambda col, fb: values_for_column(df, col, fb)
    with st.form("churn_prediction_form", border=True):
        st.markdown("#### Customer profile")
        a, b_, c = st.columns(3)
        with a:
            gender = st.selectbox("Gender", values("gender", ["Female", "Male"]))
            senior = st.selectbox("Senior citizen", values("SeniorCitizen", [0, 1]), format_func=lambda x: "Yes" if str(x) in ("1", "1.0") else "No")
            partner = st.selectbox("Partner", values("Partner", ["Yes", "No"]))
            dependents = st.selectbox("Dependents", values("Dependents", ["Yes", "No"]))
        with b_:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            phone = st.selectbox("Phone service", values("PhoneService", ["Yes", "No"]))
            multiple = st.selectbox("Multiple lines", values("MultipleLines", ["No", "Yes", "No phone service"]))
            internet = st.selectbox("Internet service", values("InternetService", ["DSL", "Fiber optic", "No"]))
        with c:
            contract = st.selectbox("Contract", values("Contract", ["Month-to-month", "One year", "Two year"]))
            paperless = st.selectbox("Paperless billing", values("PaperlessBilling", ["Yes", "No"]))
            payment = st.selectbox("Payment method", values("PaymentMethod", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]))
        st.markdown("#### Service add-ons")
        a, b_, c = st.columns(3)
        addon = ["Yes", "No", "No internet service"]
        with a:
            online_security = st.selectbox("Online security", values("OnlineSecurity", addon))
            online_backup = st.selectbox("Online backup", values("OnlineBackup", addon))
        with b_:
            device_protection = st.selectbox("Device protection", values("DeviceProtection", addon))
            tech_support = st.selectbox("Tech support", values("TechSupport", addon))
        with c:
            streaming_tv = st.selectbox("Streaming TV", values("StreamingTV", addon))
            streaming_movies = st.selectbox("Streaming movies", values("StreamingMovies", addon))
        st.markdown("#### Billing profile")
        a, b_, c = st.columns(3)
        with a:
            monthly = st.number_input("Monthly charges ($)", min_value=0.0, max_value=10000.0, value=70.0, step=5.0)
        with b_:
            total = st.number_input("Total charges ($)", min_value=0.0, max_value=1000000.0, value=840.0, step=25.0)
        with c:
            st.markdown('<div style="padding:12px 14px;border:1px solid #e5e7eb;border-radius:10px;background:#f3f6fa;margin-top:4px"><div style="color:#6b7280;font-size:12px">Prediction engine</div><div style="font-size:15px;font-weight:600;color:#1d6fe8;margin-top:6px">Pipeline inference</div><div style="font-size:12px;color:#6b7280;margin-top:3px">Preprocessing + classifier</div></div>', unsafe_allow_html=True)
        submitted = st.form_submit_button("Estimate churn risk", use_container_width=True)

    if submitted:
        customer = {
            "gender": gender, "SeniorCitizen": int(senior), "Partner": partner, "Dependents": dependents,
            "tenure": int(tenure), "PhoneService": phone, "MultipleLines": multiple, "InternetService": internet,
            "OnlineSecurity": online_security, "OnlineBackup": online_backup, "DeviceProtection": device_protection,
            "TechSupport": tech_support, "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
            "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": float(monthly), "TotalCharges": float(total),
        }
        feature_cols = [x for x in df.columns if x not in ("Churn", "customerID", "CustomerID")]
        input_df = pd.DataFrame([customer]).reindex(columns=feature_cols)
        try:
            pred = model.predict(input_df)[0]
            probability = float(churn_probability(model, input_df)[0]) if hasattr(model, "predict_proba") else None
            churned = str(pred).strip().lower() in ("1", "yes", "true")
            st.write("")
            with st.container(border=True):
                section_title("Prediction result", "Interpret this as a risk estimate, not a guarantee.", "Model output")
                r1, r2 = st.columns([1.3, 1])
                with r1:
                    if churned:
                        st.markdown('<span class="result-high">Elevated churn risk</span>', unsafe_allow_html=True)
                        st.markdown("### This customer may leave")
                        st.write("Consider proactive outreach, reviewing contract options, and checking the support experience.")
                    else:
                        st.markdown('<span class="result-low">Lower churn risk</span>', unsafe_allow_html=True)
                        st.markdown("### This customer is predicted to stay")
                        st.write("Maintain service quality and monitor engagement over time.")
                with r2:
                    if probability is not None:
                        st.metric("Estimated churn probability", f"{probability*100:.1f}%")
                        st.progress(max(0.0, min(1.0, probability)))
                        st.caption("Model-estimated probability.")
                    else:
                        st.info("This model does not expose class probabilities.")
                st.download_button("Download prediction record", pd.DataFrame([{**customer, "predicted_churn": int(churned), "churn_probability": probability}]).to_csv(index=False).encode("utf-8"), "churn_prediction.csv", "text/csv")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.info("Check that the model was trained with the same dataset schema as the app.")

# ───────────────────────── Customer explorer ──────────────────────────
elif page == "Customer explorer":
    fragment = getattr(st, "fragment", None) or (lambda f: f)   # reruns only this section when filters change

    @fragment
    def explorer():
        section_title("Filter customer records", "Use filters to isolate a segment, then export your selection.")
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            status_filter = st.selectbox("Customer status", ["All customers", "Retained", "Churned"])
        with f2:
            contract_filter = st.selectbox("Contract type", ["All contracts"] + values_for_column(df, "Contract", []))
        with f3:
            internet_filter = st.selectbox("Internet service", ["All services"] + values_for_column(df, "InternetService", []))
        with f4:
            tmax = int(df["tenure"].max()) if "tenure" in df.columns and df["tenure"].notna().any() else 72
            trange = st.slider("Tenure range (months)", 0, max(1, tmax), (0, max(1, tmax)))
        mask = pd.Series(True, index=df.index)
        if status_filter == "Retained": mask &= df["Churn"] == 0
        if status_filter == "Churned": mask &= df["Churn"] == 1
        if contract_filter != "All contracts" and "Contract" in df: mask &= df["Contract"] == contract_filter
        if internet_filter != "All services" and "InternetService" in df: mask &= df["InternetService"] == internet_filter
        if "tenure" in df: mask &= df["tenure"].between(*trange)
        filtered = df[mask]
        st.write("")
        m1, m2, m3 = st.columns(3)
        m1.metric("Matching customers", f"{len(filtered):,}")
        m2.metric("Churn rate in selection", f"{filtered['Churn'].mean()*100:.1f}%" if len(filtered) else "—")
        m3.metric("Average tenure", f"{filtered['tenure'].mean():.1f} months" if len(filtered) and "tenure" in filtered else "—")
        visible = [x for x in ["customerID", "gender", "tenure", "Contract", "InternetService", "MonthlyCharges", "TotalCharges", "Churn"] if x in filtered.columns]
        st.dataframe(clean_display_data(filtered[visible].head(1000)), use_container_width=True, hide_index=True, height=390)
        if len(filtered) > 1000:
            st.caption(f"Showing the first 1,000 of {len(filtered):,} rows. The export includes all matching rows.")
        if role in ("Admin", "Analyst"):
            st.download_button("Export filtered customers (CSV)", clean_display_data(filtered).to_csv(index=False).encode("utf-8"), "filtered_customers.csv", "text/csv")
        else:
            st.caption("Export is available to Admin and Analyst roles.")
    explorer()

# ───────────────────────── Analytics ──────────────────────────────────
elif page == "Insights & analytics":
    figs = analytics_bundle(*DATA_KEY)
    layout = [("contract", "Churn by contract", "Share of customers who left"), ("tenure", "Average tenure by outcome", "Months with the provider"),
              ("charges", "Monthly charges distribution", "Compare retained vs churned"), ("bands", "Tenure bands", "Churn rate by customer lifetime")]
    for row in (layout[:2], layout[2:]):
        for col, (key, head, note) in zip(st.columns(2, gap="medium"), row):
            with col:
                with st.container(border=True):
                    section_title(head, note)
                    if key in figs: plotly_chart(figs[key], f"an-{key}")
                    else: st.info("Not available in this dataset.")
        st.write("")

# ───────────────────────── Model lab ──────────────────────────────────
elif page == "Model lab":
    section_title("Model status", "Your Streamlit app uses a saved scikit-learn pipeline.")
    if model is None:
        st.error("Model file missing. Add the dataset, then run `python src/train.py`.")
    else:
        st.success(f"Pipeline loaded from `{MODEL_PATH.relative_to(BASE_DIR)}`")
        st.write("**Model type:**", type(model).__name__)
        if hasattr(model, "named_steps"):
            st.write("**Pipeline stages:**", " → ".join(model.named_steps.keys()))
    st.write("")
    if METRICS_PATH.exists():
        try:
            metrics = json.loads(METRICS_PATH.read_text())
            section_title("Held-out test results", "Metrics generated by src/train.py, not hard-coded values.")
            for col, key, label in zip(st.columns(4), ["accuracy", "precision", "recall", "f1"], ["Accuracy", "Precision", "Recall", "F1 score"]):
                col.metric(label, f"{metrics.get(key, 0)*100:.1f}%")
            if metrics.get("roc_auc") is not None:
                st.metric("ROC-AUC", f"{metrics['roc_auc']*100:.1f}%")
            st.caption(f"Evaluated on {metrics.get('test_rows', '—')} held-out rows. Split seed: {metrics.get('random_state', 42)}.")
        except Exception:
            st.warning("Metrics file exists but couldn't be read. Re-run training.")
    else:
        st.info("No evaluation report found yet. Run `python src/train.py` to train and evaluate the model.")
    st.write("")
    with st.container(border=True):
        section_title("Project assets", "Useful files in this workspace")
        st.markdown("- `data/` — source dataset\n- `models/churn_pipeline.joblib` — trained preprocessing + model pipeline\n- `models/metrics.json` — test-set evaluation metrics\n- `src/train.py` — reproducible training script")
        st.download_button("Download current dataset", df.to_csv(index=False).encode("utf-8"), "customer_churn_dataset.csv", "text/csv")

st.markdown('<div class="footer">PulseOps Customer Intelligence <span style="float:right">Python · scikit-learn · Streamlit</span></div>', unsafe_allow_html=True)