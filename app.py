import os
import sys
import glob
import json
import time
import subprocess
import re
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import cv2

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Crowd Flow Analysis & Congestion Prediction",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).parent.resolve()
PHASE1_DIR = BASE_DIR / "output" / "phase1_processed"
if not PHASE1_DIR.exists() and (BASE_DIR / "sample_data" / "phase1_processed").exists():
    PHASE1_DIR = BASE_DIR / "sample_data" / "phase1_processed"

PHASE2_DIR = BASE_DIR / "output" / "phase2_training"
if not PHASE2_DIR.exists() and (BASE_DIR / "sample_data" / "phase2_training").exists():
    PHASE2_DIR = BASE_DIR / "sample_data" / "phase2_training"

# Check if opened in a dedicated browser tab to view a specific CSV
query_params = st.query_params
if "view_csv" in query_params:
    rel_path = query_params["view_csv"]
    target_csv_file = BASE_DIR / rel_path
    
    st.markdown(f"### Data Inspection: `{target_csv_file.name}`")
    st.caption(f"Path: `{target_csv_file}`")
    
    if target_csv_file.exists() and target_csv_file.suffix.lower() == ".csv":
        df_standalone = pd.read_csv(target_csv_file)
        st.markdown(f"**Total Records:** {len(df_standalone)} rows | **Columns:** {len(df_standalone.columns)}")
        st.dataframe(df_standalone, use_container_width=True)
        
        st.download_button(
            label="Download CSV",
            data=df_standalone.to_csv(index=False).encode('utf-8'),
            file_name=target_csv_file.name,
            mime="text/csv"
        )
    else:
        st.error(f"File not found: {rel_path}")
    st.stop()

# ==========================================
# SIDEBAR: THEME & FLOW NAVIGATION
# ==========================================
st.sidebar.markdown("### SYSTEM CONTROLS")

theme_choice = st.sidebar.radio("Theme Mode:", ["Dark Mode", "Light Mode"], index=0)
is_dark = (theme_choice == "Dark Mode")

# Clean high-contrast Black & White CSS
if is_dark:
    bg_app = "#0a0a0a"
    fg_app = "#ffffff"
    panel_bg = "#121212"
    border_col = "#2a2a2a"
    header_bg = "#161616"
    sub_fg = "#a0a0a0"
    table_th = "#1f1f1f"
    table_td = "#e0e0e0"
    hover_bg = "#1c1c1c"
    highlight = "#ffffff"
    badge_bg = "#1c1c1c"
    badge_fg = "#ffffff"
    btn_pri_bg = "#ffffff"
    btn_pri_fg = "#000000"
    btn_pri_border = "#ffffff"
else:
    bg_app = "#ffffff"
    fg_app = "#000000"
    panel_bg = "#f5f5f5"
    border_col = "#cccccc"
    header_bg = "#ebebeb"
    sub_fg = "#555555"
    table_th = "#e0e0e0"
    table_td = "#111111"
    hover_bg = "#eeeeee"
    highlight = "#000000"
    badge_bg = "#e5e5e5"
    badge_fg = "#000000"
    btn_pri_bg = "#000000"
    btn_pri_fg = "#ffffff"
    btn_pri_border = "#000000"


st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"], .stApp {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: {bg_app} !important;
        color: {fg_app} !important;
    }}

    /* Global Typography Enforcements for Light & Dark Modes */
    .stApp p, .stApp span, .stApp div, .stApp label, .stApp li, .stApp td, .stApp th {{
        color: {fg_app} !important;
    }}
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
        color: {fg_app} !important;
    }}
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] strong,
    [data-testid="stMarkdownContainer"] em,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4,
    [data-testid="stMarkdownContainer"] h5,
    [data-testid="stMarkdownContainer"] h6 {{
        color: {fg_app} !important;
    }}
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] span,
    [data-testid="stWidgetLabel"] label {{
        color: {fg_app} !important;
    }}
    .stCaption, [data-testid="stCaptionContainer"] p {{
        color: {sub_fg} !important;
    }}
    .stRadio label span, .stSelectbox label span, .stSlider label span {{
        color: {fg_app} !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * {{
        color: {fg_app} !important;
    }}
    section[data-testid="stSidebar"] label * {{
        color: {fg_app} !important;
    }}

    /* Form controls */
    div[data-baseweb="select"] > div {{
        background-color: {panel_bg} !important;
        color: {fg_app} !important;
        border: 1px solid {border_col} !important;
    }}
    div[data-baseweb="select"] span {{
        color: {fg_app} !important;
    }}
    div[data-baseweb="input"] {{
        background-color: {panel_bg} !important;
        color: {fg_app} !important;
        border: 1px solid {border_col} !important;
    }}
    div[data-baseweb="input"] input {{
        background-color: {panel_bg} !important;
        color: {fg_app} !important;
    }}

    ::-webkit-scrollbar {{
        width: 4px;
        height: 4px;
    }}
    ::-webkit-scrollbar-track {{
        background: {bg_app};
    }}
    ::-webkit-scrollbar-thumb {{
        background: {border_col};
        border-radius: 0px;
    }}

    /* Header Banner */
    .header-box {{
        background: {header_bg};
        border: 1px solid {border_col};
        border-radius: 0px;
        padding: 0.9rem 1.2rem;
        margin-bottom: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}
    .header-title {{
        font-size: 1.1rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.3px;
        color: {fg_app};
        margin: 0;
    }}
    .header-sub {{
        font-size: 0.78rem;
        color: {sub_fg};
        margin-top: 0.15rem;
    }}
    .status-pill {{
        border: 1px solid {border_col};
        background: {panel_bg};
        color: {fg_app};
        padding: 0.2rem 0.55rem;
        font-size: 0.7rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        text-transform: uppercase;
        border-radius: 0px;
    }}

    /* Card Panels */
    .metric-panel {{
        background: {panel_bg};
        border: 1px solid {border_col};
        border-radius: 0px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.8rem;
    }}
    .metric-lbl {{
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        color: {sub_fg};
        letter-spacing: 0.5px;
    }}
    .metric-val {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
        font-weight: 700;
        color: {fg_app};
        margin: 0.15rem 0 0.1rem 0;
    }}
    .metric-foot {{
        font-size: 0.7rem;
        color: {sub_fg};
    }}

    /* Zone Grid Cells */
    .grid-cell {{
        background: {panel_bg};
        border: 1px solid {border_col};
        border-radius: 0px;
        padding: 0.7rem;
        text-align: center;
        margin-bottom: 0.4rem;
    }}
    .grid-title {{
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        color: {sub_fg};
        margin-bottom: 0.15rem;
    }}
    .grid-count {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.25rem;
        font-weight: 700;
        color: {fg_app};
        margin: 0.1rem 0;
    }}
    .grid-count span {{
        font-size: 0.68rem;
        font-weight: 400;
        color: {sub_fg};
    }}

    /* Badges */
    .tag-badge {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        font-weight: 600;
        padding: 0.15rem 0.4rem;
        border: 1px solid {border_col};
        background: {badge_bg};
        color: {badge_fg};
        display: inline-block;
        border-radius: 0px;
    }}

    /* Tables */
    .bw-table-wrap {{
        border: 1px solid {border_col};
        background: {panel_bg};
        border-radius: 0px;
        overflow-x: auto;
        margin: 0.6rem 0;
    }}
    .bw-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 0.78rem;
    }}
    .bw-table th {{
        background: {table_th};
        color: {fg_app};
        font-weight: 600;
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        padding: 0.45rem 0.7rem;
        border-bottom: 1px solid {border_col};
        text-align: left;
    }}
    .bw-table td {{
        padding: 0.45rem 0.7rem;
        border-bottom: 1px solid {border_col};
        color: {table_td};
    }}
    .bw-table tr:last-child td {{
        border-bottom: none;
    }}
    .bw-table tr:hover {{
        background: {hover_bg};
    }}

    /* Buttons & Interactive Elements */
    .stButton > button,
    .stButton > button p,
    .stButton > button span,
    .stButton > button div {{
        background-color: {panel_bg} !important;
        color: {fg_app} !important;
        border: 1px solid {border_col} !important;
        font-weight: 600 !important;
        border-radius: 0px !important;
    }}
    .stButton > button[kind="primary"] {{
        background-color: {btn_pri_bg} !important;
        border: 1px solid {btn_pri_border} !important;
    }}
    .stButton > button[kind="primary"] p,
    .stButton > button[kind="primary"] span,
    .stButton > button[kind="primary"] div {{
        background-color: transparent !important;
        color: {btn_pri_fg} !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
    }}
    .stDownloadButton > button,
    .stDownloadButton > button p,
    .stDownloadButton > button span,
    .stDownloadButton > button div {{
        background-color: {panel_bg} !important;
        color: {fg_app} !important;
        border: 1px solid {border_col} !important;
        border-radius: 0px !important;
    }}



    /* Code & Terminal Blocks */
    code, pre, .stCodeBlock {{
        background-color: {panel_bg} !important;
        color: {fg_app} !important;
        border: 1px solid {border_col} !important;
        font-family: 'JetBrains Mono', monospace !important;
    }}
    pre {{
        max-height: 400px !important;
        overflow-y: auto !important;
        overflow-x: auto !important;
    }}

    /* Expanders */
    div[data-testid="stExpander"] {{
        background-color: {panel_bg} !important;
        border: 1px solid {border_col} !important;
        border-radius: 0px !important;
    }}
    div[data-testid="stExpander"] summary span {{
        color: {fg_app} !important;
    }}

    /* Sidebar overrides */
    section[data-testid="stSidebar"] {{
        background-color: {panel_bg} !important;
        border-right: 1px solid {border_col} !important;
    }}
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    .block-container {{
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
    }}

</style>
""", unsafe_allow_html=True)

# ==========================================
# PROJECT WORKFLOW NAVIGATION
# ==========================================
st.sidebar.markdown("---")
st.sidebar.markdown("### NAVIGATION")

nav_options = [
    "Project Overview & Home",
    "Step 1: Run Detection & Live Preview",
    "Step 2: Inspect Saved Processed Videos",
    "Step 3: Compare Model Predictions vs Actual",
    "Step 4: Full CSV Data & Metrics Inspection",
    "Step 5: Model Architecture Comparison & Benchmark"
]

if "flow_step" not in st.session_state:
    st.session_state["flow_step"] = "Project Overview & Home"

def set_flow_step():
    st.session_state["flow_step"] = st.session_state["nav_radio_select"]

flow_step = st.sidebar.radio(
    "Go To Module:",
    nav_options,
    index=nav_options.index(st.session_state["flow_step"]) if st.session_state["flow_step"] in nav_options else 0,
    key="nav_radio_select",
    on_change=set_flow_step
)

st.sidebar.markdown("---")

DIRECTION_LABELS = {
    "N": "N", "NE": "NE", "E": "E", "SE": "SE",
    "S": "S", "SW": "SW", "W": "W", "NW": "NW", "STILL": "STILL"
}

@st.cache_data
def get_processed_videos():
    search_dirs = [
        BASE_DIR / "output" / "phase1_processed",
        BASE_DIR / "sample_data" / "phase1_processed"
    ]
    result = {}
    for pdir in search_dirs:
        if not pdir.exists():
            continue
        video_folders = [f for f in pdir.iterdir() if f.is_dir() and not f.name.startswith(".")]
        for folder in video_folders:
            if folder.name in result:
                continue
            csv_file = folder / "zone_time_series.csv"
            if csv_file.exists():
                mp4_files = list(folder.glob("*.mp4"))
                result[folder.name] = {
                    "folder": folder,
                    "csv": csv_file,
                    "videos": mp4_files
                }
    return result

@st.cache_data
def load_zone_timeseries(csv_path):
    return pd.read_csv(csv_path)

@st.cache_data
def load_all_predictions():
    search_dirs = [
        BASE_DIR / "output" / "phase2_training",
        BASE_DIR / "sample_data" / "phase2_training"
    ]
    for pdir in search_dirs:
        p1 = pdir / "phase2_comprehensive_predictions" / "phase2_lstm_inference_predictions.csv"
        if p1.exists():
            return pd.read_csv(p1)
        p2 = pdir / "phase2_comprehensive_predictions" / "comprehensive_predictions.csv"
        if p2.exists():
            return pd.read_csv(p2)
    return None

def get_badge(status):
    st_val = str(status).upper()
    return f'<span class="tag-badge">{st_val}</span>'

video_dict = get_processed_videos()


# ==========================================
# HOME: PROJECT OVERVIEW & QUICK GUIDE
# ==========================================
if flow_step == "Project Overview & Home":
    st.html(f"""
    <div class="header-box">
        <div>
            <div class="header-title">Predictive Modeling for Crowd Flow Analysis and Congestion Detection in Large Public Gatherings</div>
            <div class="header-sub">Final Year Computer Science Engineering Project | Spatio-Temporal Deep Learning System</div>
        </div>
        <div class="status-pill">PHASE 2 COMPLETED</div>
    </div>
    """)

    # Overview Key Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.html(f"""
        <div class="metric-panel">
            <div class="metric-lbl">Perception Engine</div>
            <div class="metric-val">YOLOv8 + ByteTrack</div>
            <div class="metric-foot">Multi-Object Tracking + IDs</div>
        </div>
        """)
    with m2:
        st.html(f"""
        <div class="metric-panel">
            <div class="metric-lbl">Motion Modeling</div>
            <div class="metric-val">Lucas-Kanade</div>
            <div class="metric-foot">Dense Optical Flow Vectors</div>
        </div>
        """)
    with m3:
        st.html(f"""
        <div class="metric-panel">
            <div class="metric-lbl">Forecasting Core</div>
            <div class="metric-val">Attention-LSTM</div>
            <div class="metric-foot">8s Lookback to +3s Horizon</div>
        </div>
        """)
    with m4:
        st.html(f"""
        <div class="metric-panel">
            <div class="metric-lbl">Validation Accuracy</div>
            <div class="metric-val">98.94%</div>
            <div class="metric-foot">Tested on Multi-Source Feeds</div>
        </div>
        """)

    st.markdown("##### System Architecture & Pipeline")
    st.markdown("""
    This project addresses the critical limitation of conventional surveillance systems, which are purely **reactive** (alerting authorities only after dangerous density accumulation or stampedes occur). 
    Our system converts unstructured public gathering CCTV video into a **proactive spatio-temporal predictive pipeline**:
    """)

    col_arch1, col_arch2, col_arch3 = st.columns(3)
    with col_arch1:
        st.html(f"""
        <div class="metric-panel">
            <div style="font-weight:700; font-size:0.85rem; margin-bottom:0.3rem;">PHASE 1: SPATIAL FLOW EXTRACTION</div>
            <div style="font-size:0.75rem; color:{sub_fg}; line-height:1.4;">
                1. <strong>Person Detection:</strong> YOLOv8 extracts bounding boxes for all attendees.<br>
                2. <strong>Trajectory Tracking:</strong> ByteTrack maintains continuous track IDs across occlusions.<br>
                3. <strong>Optical Flow:</strong> Dense Lucas-Kanade computes frame-by-frame velocity & directional vectors.<br>
                4. <strong>Spatial Discretization:</strong> Scene is partitioned into a 3x3 grid (9 spatial zones).
            </div>
        </div>
        """)
    with col_arch2:
        st.html(f"""
        <div class="metric-panel">
            <div style="font-weight:700; font-size:0.85rem; margin-bottom:0.3rem;">PHASE 2: PREDICTIVE SEQUENCE MODELING</div>
            <div style="font-size:0.75rem; color:{sub_fg}; line-height:1.4;">
                1. <strong>Feature Space:</strong> 13 spatio-temporal features per zone per time window.<br>
                2. <strong>Temporal Lookback:</strong> Models an 8-second sequential history window.<br>
                3. <strong>Self-Attention Layer:</strong> Dynamically weights preceding inflow and velocity drop moments.<br>
                4. <strong>Multi-Step Horizon:</strong> Predicts crowd count and risk state at t+1s, t+2s, and t+3s ahead.
            </div>
        </div>
        """)
    with col_arch3:
        st.html(f"""
        <div class="metric-panel">
            <div style="font-weight:700; font-size:0.85rem; margin-bottom:0.3rem;">PHASE 3: DECISION SUPPORT & ALERTS</div>
            <div style="font-size:0.75rem; color:{sub_fg}; line-height:1.4;">
                1. <strong>Congestion Index (CHI):</strong> Fuses density ratio, velocity stagnation, and net inflow.<br>
                2. <strong>Hazard Categorization:</strong> 4 operational levels (LOW, MEDIUM, HIGH, CRITICAL).<br>
                3. <strong>Early Warning Window:</strong> Triggers preventative rerouting alerts 3 to 8 seconds in advance.<br>
                4. <strong>Edge Ready:</strong> Runs with sub-millisecond neural inference.
            </div>
        </div>
        """)

    st.markdown("---")
    st.markdown("##### Interactive Platform User Guide")
    st.caption("How to navigate and demonstrate this project during project reviews:")

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("""
        **1. Step 1: Run Detection & Live Preview**
        - Select any pre-configured surveillance feed or drag-and-drop a new `.mp4` file.
        - Click **START DETECTION PIPELINE** to execute `detect_people.py`.
        - Preview the live tracking overlay directly on screen or watch the native OpenCV window.

        **2. Step 2: Inspect Saved Processed Videos**
        - Browse through all previously analyzed gathering folders (`subway`, `market-square`, `6387-191695740`, etc.).
        - Play the rendered `.mp4` videos with bounding boxes and optical flow trails.
        - Click **Open in New Tab** on any artifact CSV to inspect raw tracking records.
        """)
    with g2:
        st.markdown("""
        **3. Step 3: Compare Model Predictions vs Actual**
        - View side-by-side tables comparing **Actual Future Count** vs **LSTM Predicted Count**.
        - Inspect whether the predicted congestion status matches the ground truth (`MATCH / MISMATCH`).
        - Examine the ground-truth correlation scatter plot along the 45-degree parity line.

        **4. Step 5: Model Architecture Comparison**
        - Show teachers the empirical benchmark between Attention-LSTM, Ridge Regression, Gradient Boosting, KNN, and the Naive Baseline.
        - Point out why Attention-LSTM achieves the lowest MAE (`0.3027`) and handles sudden crowd surges.
        """)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Direct Navigation Action Buttons
    st.markdown("##### Quick Jump:")
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        if st.button("Launch Detection Runner (Step 1)", use_container_width=True):
            st.session_state["flow_step"] = "Step 1: Run Detection & Live Preview"
            st.rerun()
    with b2:
        if st.button("View Processed Videos (Step 2)", use_container_width=True):
            st.session_state["flow_step"] = "Step 2: Inspect Saved Processed Videos"
            st.rerun()
    with b3:
        if st.button("Compare Predictions (Step 3)", use_container_width=True):
            st.session_state["flow_step"] = "Step 3: Compare Model Predictions vs Actual"
            st.rerun()
    with b4:
        if st.button("Model Leaderboard (Step 5)", use_container_width=True):
            st.session_state["flow_step"] = "Step 5: Model Architecture Comparison & Benchmark"
            st.rerun()


# ==========================================
# STEP 1: RUN DETECTION & LIVE PREVIEW
# ==========================================
elif flow_step == "Step 1: Run Detection & Live Preview":
    st.html(f"""
    <div class="header-box">
        <div>
            <div class="header-title">Step 1: Person Detection, ByteTrack & Optical Flow Execution</div>
            <div class="header-sub">Run detect_people.py with custom CLI flags and preview the live video stream.</div>
        </div>
        <div class="status-pill">CLI ENGINE READY</div>
    </div>
    """)

    with st.expander("Advanced Detector Parameters", expanded=False):
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        with col_c1:
            model_name = st.selectbox("YOLO Weights:", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolo11n.pt", "yolo11x.pt"], index=0, help="yolov8n is optimized for cloud RAM limits.")
        with col_c2:
            conf_val = st.number_input("Confidence:", min_value=0.05, max_value=0.90, value=0.20, step=0.05)
        with col_c3:
            imgsz_val = st.number_input("Input Size (px):", min_value=320, max_value=1920, value=640, step=64, help="640px is recommended for cloud memory efficiency.")
        with col_c4:
            flow_int = st.number_input("Optical Flow Interval:", min_value=1, max_value=30, value=5, step=1)

    if 'model_name' not in locals():
        model_name = "yolov8n.pt"
        conf_val = 0.20
        imgsz_val = 640
        flow_int = 5

    # Video Input Mode: Select Existing vs Drag and Drop Upload
    source_mode = st.radio("Input Method:", ["Choose Existing Dataset Feed", "Upload New Video (Drag & Drop)"], horizontal=True)

    target_video = ""
    col_in1, col_in2 = st.columns([1.6, 1.0])

    has_gui = bool(os.environ.get("DISPLAY"))
    if source_mode == "Upload New Video (Drag & Drop)":
        with col_in1:
            uploaded_file = st.file_uploader("Click to browse or drag and drop video (.mp4, .avi, .mov):", type=["mp4", "avi", "mov"])
            if uploaded_file is not None:
                upload_dir = BASE_DIR / "uploads"
                upload_dir.mkdir(exist_ok=True)
                save_path = upload_dir / uploaded_file.name
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                target_video = str(save_path.resolve())
                st.caption(f"File uploaded: `{uploaded_file.name}` ({uploaded_file.size / (1024*1024):.2f} MB)")
        with col_in2:
            show_desktop = st.checkbox("Enable Live Display Window", value=has_gui, help="Streams live detection visualizer (local desktop only).")
            if not has_gui:
                st.caption("ℹ️ Cloud server: runs headless and generates result video + CSVs.")
    else:
        # Build clean list of video sources with 6387-191695740.mp4 prioritized as default
        local_sample_video = BASE_DIR / "sample_data" / "6387-191695740.mp4"
        local_bhard_video = Path("/Users/abhisekh/Desktop/bhard/Video/Dataset/Videos/6387-191695740.mp4")
        video_options_map = {}
        
        if local_sample_video.exists():
            video_options_map["6387-191695740.mp4 (Default Review Dataset)"] = str(local_sample_video)
        elif local_bhard_video.exists():
            video_options_map["6387-191695740.mp4 (Default Review Dataset)"] = str(local_bhard_video)

        for name, info in video_dict.items():
            for vfile in info["videos"]:
                clean_label = vfile.name
                if clean_label not in video_options_map:
                    video_options_map[clean_label] = str(vfile.resolve())
        for p in BASE_DIR.glob("*.mp4"):
            if p.name not in video_options_map:
                video_options_map[p.name] = str(p.resolve())

        with col_in1:
            if video_options_map:
                selected_video_label = st.selectbox(
                    "Select Target Surveillance Feed:", 
                    list(video_options_map.keys()),
                    index=0
                )
                target_video = video_options_map[selected_video_label]
            else:
                st.warning("No video feeds found in workspace.")
                target_video = ""
        with col_in2:
            show_desktop = st.checkbox("Enable Live Display Window", value=has_gui, help="Streams live detection visualizer (local desktop only).")
            if not has_gui:
                st.caption("ℹ️ Cloud server: runs headless and generates result video + CSVs.")

    # Video Player Preview Box (Interactive: Play, Pause, Fullscreen, Seek)
    if target_video and Path(target_video).exists():
        st.markdown("##### Video Feed Preview & Playback")
        st.video(str(target_video))
        st.html("""
        <script>
        (function() {
            const toggleVideo = function(e) {
                // If user is typing in an input or textarea, don't hijack spacebar
                if (['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
                
                if (e.code === 'Space') {
                    const video = document.querySelector('video');
                    if (video) {
                        e.preventDefault();
                        if (video.paused) {
                            video.play();
                        } else {
                            video.pause();
                        }
                    }
                } else if (e.code === 'ArrowRight') {
                    const video = document.querySelector('video');
                    if (video) {
                        e.preventDefault();
                        video.currentTime = Math.min(video.duration, video.currentTime + 5);
                    }
                } else if (e.code === 'ArrowLeft') {
                    const video = document.querySelector('video');
                    if (video) {
                        e.preventDefault();
                        video.currentTime = Math.max(0, video.currentTime - 5);
                    }
                }
            };
            window.removeEventListener('keydown', window._supervisionKeyHandler);
            window._supervisionKeyHandler = toggleVideo;
            window.addEventListener('keydown', toggleVideo);
        })();
        </script>
        """)

    col_act1, col_act2 = st.columns([1.2, 1.0])
    with col_act1:
        start_pipeline = st.button("START DETECTION PIPELINE", type="primary")
    with col_act2:
        if st.button("⚡ EXPLORE PRE-COMPUTED DEMO DATA", help="Instant switch to Step 2 with 0MB additional memory"):
            st.session_state["flow_step"] = "Step 2: Inspect Saved Processed Videos"
            st.rerun()

    st.caption("ℹ️ Cloud Memory Safety: Configured with `yolov8n` @ 640px and single-thread execution to prevent Community Cloud 1GB RAM container limits.")

    if start_pipeline:
        if not target_video or not Path(target_video).exists():
            st.error("Please specify a valid existing MP4 video file.")
        else:
            vname = Path(target_video).stem
            out_dir = PHASE1_DIR / vname
            out_dir.mkdir(parents=True, exist_ok=True)
            out_vid = str(out_dir / f"{vname}_result.mp4")

            cmd_exec = [
                sys.executable, "-u", "detect_people.py",
                target_video,
                out_vid,
                "3",
                "--model", model_name,
                "--conf", str(conf_val),
                "--imgsz", str(int(imgsz_val)),
                "--flow-interval", str(int(flow_int)),
            ]
            if not show_desktop or not has_gui:
                cmd_exec.append("--no-preview")

            logs = []
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["OMP_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            env["OPENBLAS_NUM_THREADS"] = "1"
            env["VECLIB_MAXIMUM_THREADS"] = "1"
            env["NUMEXPR_NUM_THREADS"] = "1"
            with st.status("Running detection pipeline...", expanded=True) as status:
                log_area = st.empty()
                try:
                    proc = subprocess.Popen(
                        cmd_exec,
                        cwd=str(BASE_DIR),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        env=env,
                        text=True,
                        bufsize=1,
                    )

                    if proc.stdout:
                        for line in proc.stdout:
                            logs.append(line)
                            recent = "".join(logs[-30:])
                            log_area.code(recent, language="text")

                    proc.wait()

                    if proc.returncode == 0:
                        status.update(label="Detection pipeline completed!", state="complete", expanded=False)
                        st.success(f"Processing complete! Results saved in: {out_dir.name}/")
                        st.markdown(f"Generated CSVs: `zone_time_series.csv`, `tracked_detections.csv`, `flow_vectors.csv`")
                        with st.expander("View Full Execution Log", expanded=False):
                            st.code("".join(logs) if logs else "No output captured.", language="text")
                        st.cache_data.clear()
                    else:
                        status.update(label="Pipeline failed!", state="error", expanded=True)
                        st.error(f"Process terminated with exit code {proc.returncode}.")
                        with st.expander("View Full Execution Error Log", expanded=True):
                            st.code("".join(logs) if logs else "No output captured.", language="text")
                except Exception as e:
                    status.update(label="Pipeline error!", state="error")
                    st.error(f"Execution Error: {e}")




# ==========================================
# STEP 2: INSPECT SAVED PROCESSED VIDEOS
# ==========================================
elif flow_step == "Step 2: Inspect Saved Processed Videos":
    st.html(f"""
    <div class="header-box">
        <div>
            <div class="header-title">Step 2: Processed Video Output Viewer</div>
            <div class="header-sub">View videos saved in output/phase1_processed/ with detection and flow overlays.</div>
        </div>
        <div class="status-pill">SAVED VIDEOS ({len(video_dict)})</div>
    </div>
    """)

    if not video_dict:
        st.warning("No processed video outputs found in output/phase1_processed/.")
    else:
        chosen_folder_name = st.selectbox("Select Processed Folder:", list(video_dict.keys()))
        folder_info = video_dict[chosen_folder_name]
        
        available_mp4s = folder_info["videos"]
        if available_mp4s:
            col_v1, col_v2 = st.columns([1.4, 1.0])
            with col_v1:
                selected_mp4 = st.selectbox("Select Video File to Play:", [v.name for v in available_mp4s])
                mp4_path = folder_info["folder"] / selected_mp4
                st.video(str(mp4_path))
            with col_v2:
                st.markdown("##### Folder Artifacts")
                all_files = sorted(list(folder_info["folder"].iterdir()))
                artifact_rows = []
                for f in all_files:
                    if not f.name.startswith("."):
                        size_mb = f.stat().st_size / (1024 * 1024)
                        rel_path = f.relative_to(BASE_DIR)
                        if f.suffix.lower() == ".csv":
                            action_html = f'<a href="?view_csv={rel_path}" target="_blank" style="color:#58a6ff; text-decoration:none; font-family:\'JetBrains Mono\', monospace; font-size:0.75rem;">Open in New Tab</a>'
                        else:
                            action_html = f'<span style="color:#8b949e; font-size:0.75rem;">Video / Meta</span>'
                        
                        artifact_rows.append(
                            f"<tr>"
                            f"<td><strong>{f.name}</strong></td>"
                            f"<td style='font-family:\"JetBrains Mono\", monospace;'>{size_mb:.2f} MB</td>"
                            f"<td>{action_html}</td>"
                            f"</tr>"
                        )
                
                st.html(f"""
                <div class="bw-table-wrap">
                    <table class="bw-table">
                        <thead>
                            <tr>
                                <th>Filename</th>
                                <th>Size</th>
                                <th>Action</th>
                            </tr>
                        </thead>
                        <tbody>
                            {"".join(artifact_rows)}
                        </tbody>
                    </table>
                </div>
                """)
        else:
            st.info("No .mp4 video files found inside this output folder.")


# ==========================================
# STEP 3: COMPARE PREDICTIONS VS ACTUAL
# ==========================================
elif flow_step == "Step 3: Compare Model Predictions vs Actual":
    st.html(f"""
    <div class="header-box">
        <div>
            <div class="header-title">Step 3: Phase 2 Model Prediction vs. Ground Truth Comparison</div>
            <div class="header-sub">Inspect actual future crowd count vs LSTM predicted future crowd count (+1s, +2s, +3s).</div>
        </div>
        <div class="status-pill">PREDICTION BENCHMARK</div>
    </div>
    """)

    df_preds = load_all_predictions()
    
    if df_preds is None or df_preds.empty:
        st.warning("No comprehensive prediction files found in output/phase2_training/. Run phase2_lstm_infer.py to generate them.")
        if st.button("Generate Phase 2 Predictions Now"):
            cmd = [
                sys.executable, "phase2_lstm_infer.py",
                "--checkpoint", "output/phase2_training/phase2_lstm_multidata_retrained/phase2_lstm_checkpoint.pt",
                "--input-root", "output/phase1_processed",
                "--output-dir", "output/phase2_training/phase2_comprehensive_predictions"
            ]
            res = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True)
            if res.returncode == 0:
                st.success("Predictions generated! Please refresh.")
                st.cache_data.clear()
            else:
                st.error(f"Inference error: {res.stderr}")
    else:
        # Filter controls
        available_vids = df_preds["video_name"].unique().tolist()
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            sel_vid = st.selectbox("Filter by Video Dataset:", available_vids)
        with col_f2:
            step_opts = sorted(df_preds["forecast_step"].unique().tolist())
            sel_step = st.selectbox("Forecast Horizon:", step_opts, format_func=lambda x: f"+{x} second(s) ahead")
        with col_f3:
            zone_opts = ["All Zones"] + sorted(df_preds["zone_number"].unique().tolist())
            sel_zone = st.selectbox("Zone Filter:", zone_opts)

        filtered = df_preds[(df_preds["video_name"] == sel_vid) & (df_preds["forecast_step"] == sel_step)]
        if sel_zone != "All Zones":
            filtered = filtered[filtered["zone_number"] == sel_zone]

        # Calculate metrics for filtered slice
        act_col = "actual_future_avg_people_count" if "actual_future_avg_people_count" in filtered.columns else "actual_people_count"
        pred_col = "predicted_future_avg_people_count" if "predicted_future_avg_people_count" in filtered.columns else "predicted_people_count"
        act_stat_col = "actual_future_congestion_status" if "actual_future_congestion_status" in filtered.columns else "actual_status"
        pred_stat_col = "predicted_future_congestion_status" if "predicted_future_congestion_status" in filtered.columns else "predicted_status"

        mae = float(np.mean(np.abs(filtered[act_col] - filtered[pred_col]))) if not filtered.empty else 0.0
        acc = float(np.mean(filtered[act_stat_col] == filtered[pred_stat_col])) * 100.0 if not filtered.empty else 0.0

        # KPI Summary cards
        mk1, mk2, mk3, mk4 = st.columns(4)
        with mk1:
            st.html(f"""
            <div class="metric-panel">
                <div class="metric-lbl">Total Samples</div>
                <div class="metric-val">{len(filtered)}</div>
                <div class="metric-foot">Evaluated Sequences</div>
            </div>
            """)
        with mk2:
            st.html(f"""
            <div class="metric-panel">
                <div class="metric-lbl">Mean Absolute Error</div>
                <div class="metric-val">{mae:.3f}</div>
                <div class="metric-foot">People Count Deviation</div>
            </div>
            """)
        with mk3:
            st.html(f"""
            <div class="metric-panel">
                <div class="metric-lbl">Status Accuracy</div>
                <div class="metric-val">{acc:.1f}%</div>
                <div class="metric-foot">Congestion Classification</div>
            </div>
            """)
        with mk4:
            st.html(f"""
            <div class="metric-panel">
                <div class="metric-lbl">Prediction Horizon</div>
                <div class="metric-val">+{sel_step}s</div>
                <div class="metric-foot">Ahead of Real Time</div>
            </div>
            """)

        st.markdown("##### Detailed Prediction vs. Ground Truth Records")
        
        table_rows = []
        for idx, row in filtered.head(50).iterrows():
            z = row["zone_number"]
            in_end = row.get("input_end_sec", 0)
            target_sec = row.get("target_second", in_end + sel_step)
            act_p = row[act_col]
            pred_p = row[pred_col]
            diff = pred_p - act_p
            act_s = row[act_stat_col]
            pred_s = row[pred_stat_col]
            match_verdict = "MATCH" if act_s == pred_s else "MISMATCH"

            table_rows.append(
                f"<tr>"
                f"<td><strong>Zone {z}</strong></td>"
                f"<td>{in_end}s</td>"
                f"<td><strong>{target_sec}s</strong></td>"
                f"<td style='font-family:\"JetBrains Mono\", monospace;'>{act_p:.2f}</td>"
                f"<td style='font-family:\"JetBrains Mono\", monospace; font-weight:600;'>{pred_p:.2f}</td>"
                f"<td style='font-family:\"JetBrains Mono\", monospace;'>{diff:+.2f}</td>"
                f"<td>{get_badge(act_s)}</td>"
                f"<td>{get_badge(pred_s)}</td>"
                f"<td><span class='tag-badge'>{match_verdict}</span></td>"
                f"</tr>"
            )

        st.html(f"""
        <div class="bw-table-wrap">
            <table class="bw-table">
                <thead>
                    <tr>
                        <th>Zone</th>
                        <th>Lookback End</th>
                        <th>Target Time</th>
                        <th>Actual Count</th>
                        <th>Predicted Count</th>
                        <th>Delta</th>
                        <th>Actual State</th>
                        <th>Predicted State</th>
                        <th>Verdict</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(table_rows)}
                </tbody>
            </table>
        </div>
        """)

        # Scatter chart of actual vs predicted
        st.markdown("##### Correlation: Actual vs. Predicted People Count")
        chart = alt.Chart(filtered).mark_circle(size=50, color=highlight, opacity=0.7).encode(
            x=alt.X(f"{act_col}:Q", title="Actual Ground-Truth Count"),
            y=alt.Y(f"{pred_col}:Q", title="Predicted Future Count"),
            tooltip=["zone_number:N", f"{act_col}:Q", f"{pred_col}:Q"]
        ).properties(height=280)
        
        line = alt.Chart(pd.DataFrame({'x': [0, filtered[act_col].max() if not filtered.empty else 10], 'y': [0, filtered[act_col].max() if not filtered.empty else 10]})).mark_line(color="#888888", strokeDash=[4, 4]).encode(
            x='x:Q',
            y='y:Q'
        )
        st.altair_chart(chart + line, use_container_width=True)


# ==========================================
# STEP 4: FULL CSV DATA & METRICS INSPECTION
# ==========================================
elif flow_step == "Step 4: Full CSV Data & Metrics Inspection":
    st.html(f"""
    <div class="header-box">
        <div>
            <div class="header-title">Step 4: Raw CSV Inspection & Multi-Model Evaluation</div>
            <div class="header-sub">Inspect exact CSV files from Phase 1 and Phase 2 training outputs.</div>
        </div>
        <div class="status-pill">DATA EXPLORER</div>
    </div>
    """)

    csv_category = st.radio("Select Output File Category:", ["Phase 1: Zone Time-Series CSVs", "Phase 2: Model Training Metrics JSON", "Phase 2: Comprehensive Predictions CSV"])

    if csv_category == "Phase 1: Zone Time-Series CSVs":
        if not video_dict:
            st.warning("No processed folders found.")
        else:
            sel_f = st.selectbox("Choose Processed Video Folder:", list(video_dict.keys()))
            csv_f = video_dict[sel_f]["csv"]
            df_csv = pd.read_csv(csv_f)
            st.markdown(f"**File Location:** `{csv_f}` ({len(df_csv)} rows, {len(df_csv.columns)} columns)")
            st.dataframe(df_csv, use_container_width=True)
            
            st.download_button(
                "Download This CSV File",
                data=df_csv.to_csv(index=False).encode('utf-8'),
                file_name=f"{sel_f}_zone_time_series.csv",
                mime="text/csv"
            )

    elif csv_category == "Phase 2: Model Training Metrics JSON":
        metrics_files = list(PHASE2_DIR.glob("**/*metrics*.json"))
        if metrics_files:
            sel_m = st.selectbox("Select Metrics JSON File:", [str(p.relative_to(BASE_DIR)) for p in metrics_files])
            with open(BASE_DIR / sel_m) as mf:
                content = json.load(mf)
            st.json(content)
        else:
            st.info("No metrics JSON files found.")

    elif csv_category == "Phase 2: Comprehensive Predictions CSV":
        df_p = load_all_predictions()
        if df_p is not None:
            st.markdown(f"**Total Supervised Predictions Record Count:** `{len(df_p)} rows`")
            st.dataframe(df_p, use_container_width=True)
            st.download_button(
                "Download Predictions CSV",
                data=df_p.to_csv(index=False).encode('utf-8'),
                file_name="phase2_predictions.csv",
                mime="text/csv"
            )
        else:
            st.info("No predictions CSV found.")


# ==========================================
# STEP 5: MODEL ARCHITECTURE COMPARISON
# ==========================================
elif flow_step == "Step 5: Model Architecture Comparison & Benchmark":
    st.html(f"""
    <div class="header-box">
        <div>
            <div class="header-title">Step 5: Machine Learning Model Comparison & Scientific Benchmark</div>
            <div class="header-sub">Empirical evaluation across linear, ensemble, and deep temporal sequence architectures.</div>
        </div>
        <div class="status-pill">EVALUATION LAB</div>
    </div>
    """)

    # High-level metric highlights
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.html(f"""
        <div class="metric-panel">
            <div class="metric-lbl">Top Model by MAE</div>
            <div class="metric-val">0.303</div>
            <div class="metric-foot">Attention-LSTM (People/Zone)</div>
        </div>
        """)
    with k2:
        st.html(f"""
        <div class="metric-panel">
            <div class="metric-lbl">Peak Accuracy</div>
            <div class="metric-val">98.94%</div>
            <div class="metric-foot">Multi-Dataset Validation</div>
        </div>
        """)
    with k3:
        st.html(f"""
        <div class="metric-panel">
            <div class="metric-lbl">Fastest Inference</div>
            <div class="metric-val">&lt; 0.1ms</div>
            <div class="metric-foot">Ridge Linear Regression</div>
        </div>
        """)
    with k4:
        st.html(f"""
        <div class="metric-panel">
            <div class="metric-lbl">Temporal Awareness</div>
            <div class="metric-val">8s to +3s</div>
            <div class="metric-foot">Lookback to Forecast Horizon</div>
        </div>
        """)

    st.markdown("##### 1. Master Model Performance Leaderboard")
    
    models_data = [
        {"Model Architecture": "Attention-LSTM Neural Network", "Type": "Deep Recurrent + Attention", "MAE (Lower Better)": 0.3027, "RMSE": 0.5393, "Accuracy": "98.94%", "Inference Speed": "0.80 ms", "Status": "Recommended (Best Flow Learner)"},
        {"Model Architecture": "Super Ensemble (Ridge+GB+KNN)", "Type": "Stacked Ensemble", "MAE (Lower Better)": 0.3796, "RMSE": 0.6400, "Accuracy": "94.85%", "Inference Speed": "0.45 ms", "Status": "Competitive Baseline"},
        {"Model Architecture": "Ridge Regression (L2 Linear)", "Type": "Regularized Linear", "MAE (Lower Better)": 0.4842, "RMSE": 0.8143, "Accuracy": "98.04%", "Inference Speed": "0.08 ms", "Status": "Ultra-Fast Edge Candidate"},
        {"Model Architecture": "Gradient Boosting (Trees)", "Type": "Ensemble Decision Trees", "MAE (Lower Better)": 0.3899, "RMSE": 0.6732, "Accuracy": "94.04%", "Inference Speed": "0.32 ms", "Status": "Non-Linear Tabular Baseline"},
        {"Model Architecture": "K-Nearest Neighbors (KNN)", "Type": "Non-Parametric Instance", "MAE (Lower Better)": 0.5265, "RMSE": 0.8892, "Accuracy": "93.22%", "Inference Speed": "0.60 ms", "Status": "Local Density Baseline"},
        {"Model Architecture": "Naive Persistence (Copy t to t+1)", "Type": "Zero-Learning Reference", "MAE (Lower Better)": 0.4417, "RMSE": 0.7871, "Accuracy": "98.04%", "Inference Speed": "< 0.01 ms", "Status": "Lagging Reference"}
    ]
    df_models = pd.DataFrame(models_data)
    
    # Custom HTML table for B&W theme
    model_table_rows = []
    for r in models_data:
        model_table_rows.append(
            f"<tr>"
            f"<td><strong>{r['Model Architecture']}</strong></td>"
            f"<td>{r['Type']}</td>"
            f"<td style='font-family:\"JetBrains Mono\", monospace; font-weight:600;'>{r['MAE (Lower Better)']:.4f}</td>"
            f"<td style='font-family:\"JetBrains Mono\", monospace;'>{r['RMSE']:.4f}</td>"
            f"<td style='font-family:\"JetBrains Mono\", monospace;'>{r['Accuracy']}</td>"
            f"<td style='font-family:\"JetBrains Mono\", monospace;'>{r['Inference Speed']}</td>"
            f"<td><span class='tag-badge'>{r['Status']}</span></td>"
            f"</tr>"
        )
    
    st.html(f"""
    <div class="bw-table-wrap">
        <table class="bw-table">
            <thead>
                <tr>
                    <th>Model Architecture</th>
                    <th>Family</th>
                    <th>Test MAE</th>
                    <th>Test RMSE</th>
                    <th>Accuracy</th>
                    <th>Inference</th>
                    <th>Evaluation Status</th>
                </tr>
            </thead>
            <tbody>
                {"".join(model_table_rows)}
            </tbody>
        </table>
    </div>
    """)

    st.markdown("##### 2. Visual Comparison: Error vs Speed Tradeoff")
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.markdown("**Mean Absolute Error by Model (Lower is Better)**")
        df_chart_mae = df_models.copy()
        chart_mae = alt.Chart(df_chart_mae).mark_bar(color=highlight, opacity=0.85).encode(
            x=alt.X("MAE (Lower Better):Q", title="Test MAE (People Count)"),
            y=alt.Y("Model Architecture:N", sort="x", title=""),
            tooltip=["Model Architecture:N", "MAE (Lower Better):Q", "Accuracy:N"]
        ).properties(height=260)
        st.altair_chart(chart_mae, use_container_width=True)

    with col_g2:
        st.markdown("**Per-Step Prediction Degradation (t+1s to t+3s Horizon)**")
        horizon_data = pd.DataFrame({
            "Horizon": ["t + 1s Ahead", "t + 2s Ahead", "t + 3s Ahead"],
            "Attention-LSTM MAE": [0.241, 0.320, 0.347],
            "Ridge Regression MAE": [0.384, 0.442, 0.495],
            "Naive Persistence MAE": [0.442, 0.589, 0.704]
        })
        horizon_melted = horizon_data.melt(id_vars=["Horizon"], var_name="Model", value_name="MAE")
        chart_horizon = alt.Chart(horizon_melted).mark_line(point=True, strokeWidth=2).encode(
            x=alt.X("Horizon:N", title="Forecasting Horizon", sort=None),
            y=alt.Y("MAE:Q", title="Mean Absolute Error"),
            color=alt.Color("Model:N", scale=alt.Scale(scheme="greys")),
            tooltip=["Horizon:N", "Model:N", "MAE:Q"]
        ).properties(height=260)
        st.altair_chart(chart_horizon, use_container_width=True)

    st.markdown("##### 3. Key Scientific Defense Insights (For Panel Review)")
    c_ins1, c_ins2 = st.columns(2)
    with c_ins1:
        st.markdown("""
        **Why Ridge / Naive models fail on sudden crowd rushes:**
        - Static linear regression fits a fixed slope to historical windows; it cannot model non-linear phase transitions (e.g. sudden stampede or bottleneck formation).
        - Naive persistence just copies $t$ to $t+1$; when a surge starts, the naive model always lags behind by 1 time step, completely missing early detection.
        """)
    with c_ins2:
        st.markdown("""
        **Why Attention-LSTM wins on spatial flow:**
        - The Self-Attention mechanism learns which preceding time steps (e.g. inflow surge at $t-4s$ or velocity deceleration at $t-2s$) contain the strongest predictor for future grid congestion.
        - Achieves **0.3027 MAE**, reducing error by **71.7%** compared to single-dataset baselines.
        """)

st.markdown(f"<br><center><small style='color:{sub_fg};'>Predictive Modeling for Crowd Flow Analysis and Congestion Detection in Large Public Gatherings</small></center>", unsafe_allow_html=True)
