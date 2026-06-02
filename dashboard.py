"""
LIVE ML DEPLOYMENT DASHBOARD FOR DR. DANISH
===========================================
Apple-themed, mobile-friendly, interactive Streamlit app.
Pure machine learning - no econometrics.

Pages:
  1. Overview      - Dataset summary + live metrics
  2. Model Arena   - Live training, comparison, accuracy plots
  3. Predictor     - What-if simulator with real-time predictions
  4. Future Forecast - Multi-year iterative projections
  5. Spatial Explorer - Neighbor spillover visualization
  6. Explainability - Live SHAP breakdowns
  7. Terminal      - Transparent training logs

Run: streamlit run live_ml_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO, StringIO
import sys
import warnings
import datetime
warnings.filterwarnings('ignore')

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="ML Live Lab | Dr. Danish",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# APPLE WEBSITE THEME CSS
# ================================================================
APPLE_CSS = """
<style>
    /* ===== FORCE LIGHT MODE ===== */
    .stApp {
        background: linear-gradient(180deg, #f5f5f7 0%, #fafafc 100%) !important;
        color: #1d1d1f !important;
    }
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'SF Pro Text', 'Inter', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        color: #1d1d1f !important;
        -webkit-font-smoothing: antialiased;
    }
    
    /* Force all main content text to dark */
    .stApp .main .block-container,
    .stApp .main .block-container p,
    .stApp .main .block-container span,
    .stApp .main .block-container li,
    .stApp .main .block-container div,
    .stApp .main .block-container label {
        color: #1d1d1f !important;
    }
    
    /* Headings */
    h1 {
        color: #1d1d1f !important;
        font-weight: 700 !important;
        font-size: 2.2rem !important;
        letter-spacing: -0.02em !important;
        margin-bottom: 0.3rem !important;
    }
    h2 {
        color: #1d1d1f !important;
        font-weight: 600 !important;
        font-size: 1.5rem !important;
        letter-spacing: -0.01em !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.6rem !important;
    }
    h3 {
        color: #1d1d1f !important;
        font-weight: 600 !important;
        font-size: 1.15rem !important;
        margin-top: 1rem !important;
    }

    /* Streamlit widget labels */
    .stSlider label, .stSelectbox label, .stNumberInput label, .stRadio label,
    [data-testid="stWidgetLabel"] label,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] span {
        color: #1d1d1f !important;
        font-weight: 500 !important;
    }

    /* Selectbox / dropdown: ensure text is visible */
    .stSelectbox [data-baseweb="select"],
    .stSelectbox [data-baseweb="select"] * {
        background-color: #ffffff !important;
        color: #1d1d1f !important;
    }
    .stSelectbox [data-baseweb="select"] [data-baseweb="tag"],
    .stSelectbox [data-baseweb="select"] .css-1dimb5e-singleValue,
    .stSelectbox div[data-baseweb="select"] span {
        color: #1d1d1f !important;
    }
    /* Dropdown menu items */
    [data-baseweb="popover"],
    [data-baseweb="popover"] li,
    [data-baseweb="menu"],
    [data-baseweb="menu"] li,
    [role="listbox"],
    [role="listbox"] [role="option"],
    [role="listbox"] [role="option"] * {
        background-color: #ffffff !important;
        color: #1d1d1f !important;
    }
    [role="listbox"] [role="option"]:hover,
    [role="listbox"] [aria-selected="true"] {
        background-color: #f0f0f5 !important;
    }

    /* Number input */
    .stNumberInput input {
        background-color: #ffffff !important;
        color: #1d1d1f !important;
        border: 1px solid #d2d2d7 !important;
    }
    .stNumberInput button {
        color: #1d1d1f !important;
        background: #f5f5f7 !important;
    }

    /* Slider thumb and value text */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: #0071e3 !important;
    }
    .stSlider [data-testid="stThumbValue"],
    .stSlider [data-baseweb="slider"] div[role="slider"] div {
        color: #ffffff !important;
    }
    .stSlider div[data-testid="stTickBarMin"],
    .stSlider div[data-testid="stTickBarMax"] {
        color: #86868b !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #1a1a2e !important;
        border-right: 1px solid rgba(255,255,255,0.06) !important;
    }
    section[data-testid="stSidebar"] * {
        color: #e0e0e6 !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        color: #e0e0e6 !important;
        padding: 8px 12px !important;
        border-radius: 10px !important;
        transition: all 0.2s ease;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255,255,255,0.06) !important;
    }
    section[data-testid="stSidebar"] .stButton > button[kind="secondary"],
    section[data-testid="stSidebar"] .stButton > button[data-testid="baseButton-secondary"] {
        background: rgba(255,255,255,0.08) !important;
        color: #e0e0e6 !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        box-shadow: none !important;
    }
    section[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover,
    section[data-testid="stSidebar"] .stButton > button[data-testid="baseButton-secondary"]:hover {
        background: rgba(255,255,255,0.14) !important;
    }
    
    /* Apple-style Metric Cards */
    .apple-card {
        background: #ffffff;
        border-radius: 18px;
        padding: 24px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.04);
        border: 1px solid rgba(0,0,0,0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .apple-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.08);
    }
    .apple-card * {
        color: #1d1d1f;
    }
    .apple-metric-value {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1d1d1f !important;
        letter-spacing: -0.03em;
        line-height: 1.1;
    }
    .apple-metric-label {
        font-size: 0.75rem;
        color: #86868b !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
        margin-top: 6px;
    }
    .apple-metric-delta {
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 4px;
    }
    .delta-positive { color: #34c759 !important; }
    .delta-negative { color: #ff3b30 !important; }
    
    /* Section cards */
    .section-card {
        background: #ffffff;
        border-radius: 20px;
        padding: 28px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.04);
        border: 1px solid rgba(0,0,0,0.04);
        margin-bottom: 20px;
    }
    .section-card * {
        color: #1d1d1f;
    }
    
    /* Buttons */
    .stApp .main .stButton > button {
        background: #0071e3 !important;
        color: #ffffff !important;
        border-radius: 980px !important;
        padding: 12px 28px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        border: none !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 16px rgba(0,113,227,0.3) !important;
    }
    .stApp .main .stButton > button:hover {
        background: #0077ed !important;
        box-shadow: 0 6px 20px rgba(0,113,227,0.4) !important;
        transform: translateY(-1px);
    }
    .stApp .main .stButton > button:active {
        transform: scale(0.98);
    }
    
    /* Secondary button (main area) */
    .stApp .main .stButton > button[kind="secondary"],
    .stApp .main .stButton > button[data-testid="baseButton-secondary"] {
        background: #f5f5f7 !important;
        color: #1d1d1f !important;
        box-shadow: none !important;
        border: 1px solid #d2d2d7 !important;
    }
    .stApp .main .stButton > button[kind="secondary"]:hover,
    .stApp .main .stButton > button[data-testid="baseButton-secondary"]:hover {
        background: #e8e8ed !important;
    }
    
    /* Sliders */
    .stSlider > div > div > div {
        background: #0071e3 !important;
    }
    
    /* Terminal */
    .terminal-window {
        background: #0d0d0d;
        border-radius: 14px;
        padding: 20px;
        font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
        font-size: 0.9rem;
        color: #00ff41;
        overflow-x: auto;
        box-shadow: inset 0 2px 8px rgba(0,0,0,0.5);
        border: 1px solid #2c2c2e;
    }
    .terminal-window * {
        color: #00ff41 !important;
    }
    .terminal-window .terminal-header * {
        color: #86868b !important;
    }
    .terminal-header {
        display: flex;
        gap: 8px;
        margin-bottom: 12px;
        padding-bottom: 12px;
        border-bottom: 1px solid #2c2c2e;
    }
    .terminal-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
    }
    .dot-red { background: #ff5f57 !important; }
    .dot-yellow { background: #febc2e !important; }
    .dot-green { background: #28c840 !important; }
    
    /* Tables */
    div[data-testid="stDataFrame"] td {
        color: #1d1d1f !important;
        font-size: 0.88rem !important;
    }
    div[data-testid="stDataFrame"] th {
        background: #f5f5f7 !important;
        color: #1d1d1f !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 500;
        color: #86868b !important;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background: #ffffff !important;
        color: #0071e3 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        font-weight: 600;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        color: #1d1d1f !important;
        background: #ffffff;
        border-radius: 12px;
    }
    details summary span {
        color: #1d1d1f !important;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #d2d2d7, transparent);
        margin: 2rem 0;
    }
    
    /* Footer */
    .footer-text {
        text-align: center;
        color: #86868b !important;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-bottom: 2rem;
    }
    
    /* Radio buttons in main area */
    .stRadio [role="radiogroup"] label {
        color: #1d1d1f !important;
    }
    
    /* Warnings/info boxes */
    .stAlert {
        color: #1d1d1f !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        color: #1d1d1f !important;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: #d2d2d7;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #86868b;
    }
</style>
"""
st.markdown(APPLE_CSS, unsafe_allow_html=True)

# ================================================================
# CONSTANTS
# ================================================================
ADJACENCY = {
    'Anhui': ['Jiangsu', 'Zhejiang', 'Jiangxi', 'Hubei', 'Henan', 'Shandong'],
    'Beijing': ['Hebei', 'Tianjin'],
    'Chongqing': ['Shaanxi', 'Hubei', 'Hunan', 'Guizhou', 'Sichuan'],
    'Fujian': ['Guangdong', 'Jiangxi', 'Zhejiang'],
    'Gansu': ['Inner Mongolia Autonomous Region', 'Ningxia Hui Autonomous Region', 'Shaanxi', 'Sichuan', 'Qinghai', 'Xinjiang Uygur Autonomous Region'],
    'Guangdong': ['Guangxi Zhuang Autonomous Region', 'Hunan', 'Jiangxi', 'Fujian'],
    'Guangxi Zhuang Autonomous Region': ['Guizhou', 'Yunnan', 'Hunan', 'Guangdong'],
    'Guizhou': ['Sichuan', 'Chongqing', 'Hunan', 'Guangxi Zhuang Autonomous Region', 'Yunnan'],
    'Hainan': ['Guangdong'],
    'Hebei': ['Beijing', 'Tianjin', 'Shanxi', 'Inner Mongolia Autonomous Region', 'Liaoning', 'Henan', 'Shandong'],
    'Heilongjiang': ['Inner Mongolia Autonomous Region', 'Jilin'],
    'Henan': ['Hebei', 'Shanxi', 'Shaanxi', 'Hubei', 'Anhui', 'Shandong'],
    'Hubei': ['Henan', 'Shaanxi', 'Chongqing', 'Hunan', 'Jiangxi', 'Anhui'],
    'Hunan': ['Hubei', 'Chongqing', 'Guizhou', 'Guangxi Zhuang Autonomous Region', 'Guangdong', 'Jiangxi'],
    'Inner Mongolia Autonomous Region': ['Heilongjiang', 'Jilin', 'Liaoning', 'Hebei', 'Shanxi', 'Shaanxi', 'Ningxia Hui Autonomous Region', 'Gansu'],
    'Jiangsu': ['Zhejiang', 'Anhui', 'Shandong', 'Shanghai'],
    'Jiangxi': ['Fujian', 'Guangdong', 'Hunan', 'Hubei', 'Anhui', 'Zhejiang'],
    'Jilin': ['Inner Mongolia Autonomous Region', 'Liaoning', 'Heilongjiang'],
    'Liaoning': ['Inner Mongolia Autonomous Region', 'Jilin', 'Hebei'],
    'Ningxia Hui Autonomous Region': ['Inner Mongolia Autonomous Region', 'Gansu', 'Shaanxi'],
    'Qinghai': ['Gansu', 'Sichuan', 'Xinjiang Uygur Autonomous Region', 'Tibet'],
    'Shaanxi': ['Inner Mongolia Autonomous Region', 'Shanxi', 'Henan', 'Hubei', 'Chongqing', 'Sichuan', 'Gansu', 'Ningxia Hui Autonomous Region'],
    'Shandong': ['Hebei', 'Henan', 'Anhui', 'Jiangsu'],
    'Shanghai': ['Jiangsu', 'Zhejiang'],
    'Shanxi': ['Hebei', 'Inner Mongolia Autonomous Region', 'Shaanxi', 'Henan'],
    'Sichuan': ['Qinghai', 'Gansu', 'Shaanxi', 'Chongqing', 'Guizhou', 'Yunnan', 'Tibet'],
    'Tianjin': ['Beijing', 'Hebei'],
    'Tibet': ['Xinjiang Uygur Autonomous Region', 'Qinghai', 'Sichuan', 'Yunnan'],
    'Xinjiang Uygur Autonomous Region': ['Gansu', 'Qinghai', 'Tibet'],
    'Yunnan': ['Tibet', 'Sichuan', 'Guizhou', 'Guangxi Zhuang Autonomous Region'],
    'Zhejiang': ['Fujian', 'Jiangxi', 'Anhui', 'Jiangsu', 'Shanghai']
}

MODEL_CONFIG = {
    'XGBoost': {
        'model': xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                   min_child_weight=5, reg_alpha=0.5, reg_lambda=2.0,
                                   subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1),
        'color': '#0071e3'
    },
    'LightGBM': {
        'model': LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                min_child_samples=5, reg_alpha=0.5, reg_lambda=2.0,
                                subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1),
        'color': '#34c759'
    },
    'Random Forest': {
        'model': RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1),
        'color': '#ff9500'
    },
    'Gradient Boosting': {
        'model': HistGradientBoostingRegressor(max_iter=200, max_depth=4, learning_rate=0.05,
                                                min_samples_leaf=5, l2_regularization=2.0, random_state=42),
        'color': '#af52de'
    },
    'Ridge Regression': {
        'model': Ridge(alpha=1.0, random_state=42),
        'color': '#ff3b30'
    },
    'Neural Network': {
        'model': MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
                               alpha=0.5, max_iter=2000, early_stopping=True,
                               validation_fraction=0.1, random_state=42),
        'color': '#5856d6'
    }
}

FEATURE_COLS = ['Arrivals', 'TourismGDP', 'GreenTech', 'Governance', 'Urbanization', 'Year',
                'Energy_lag1', 'CO2_lag1', 'Energy_neighbor_avg', 'CO2_neighbor_avg',
                'Arrivals_neighbor_avg', 'TourismGDP_neighbor_avg']

POLICY_COLS = ['Arrivals', 'TourismGDP', 'GreenTech', 'Governance', 'Urbanization']

# ================================================================
# DATA LOADING & FEATURE ENGINEERING
# ================================================================
@st.cache_data(show_spinner=False)
def load_and_engineer_data():
    df = pd.read_excel('Tourism carbon emissions data.xlsx', sheet_name='Sheet1')
    df.rename(columns={
        'Tourism carbon emissions (million tonnes)': 'CO2',
        'Tourism industry energy consumption (TJ)': 'Energy',
        'Number of Tourism arrival': 'Arrivals',
        'Tourism GDP': 'TourismGDP',
        'Green techn.': 'GreenTech',
        'Intensity of environmental regulation (Governance)': 'Governance',
        'UrbanizationRate': 'Urbanization'
    }, inplace=True)
    df = df.sort_values(['Province', 'Year'])
    df['Is_Coastal'] = df['Province'].isin(['Beijing', 'Tianjin', 'Liaoning', 'Shanghai', 'Jiangsu',
                                             'Zhejiang', 'Fujian', 'Shandong', 'Guangdong', 'Hainan'])
    
    # Lag features
    df['Energy_lag1'] = df.groupby('Province')['Energy'].shift(1)
    df['CO2_lag1'] = df.groupby('Province')['CO2'].shift(1)
    
    # Spatial neighbor features
    neighbor_vars = ['Energy', 'CO2', 'Arrivals', 'TourismGDP']
    for var in neighbor_vars:
        vals = []
        for _, row in df.iterrows():
            prov, year = row['Province'], row['Year']
            neighbors = ADJACENCY.get(prov, [])
            nd = df[(df['Province'].isin(neighbors)) & (df['Year'] == year)][var]
            vals.append(nd.mean() if len(nd) > 0 else np.nan)
        df[f'{var}_neighbor_avg'] = vals
    
    for var in neighbor_vars:
        df[f'{var}_neighbor_avg'] = df.groupby('Year')[f'{var}_neighbor_avg'].transform(lambda x: x.fillna(x.mean()))
    
    return df


# ================================================================
# TERMINAL LOGGER
# ================================================================
class TerminalLogger:
    def __init__(self):
        self.buffer = StringIO()
        self.lines = []
    
    def log(self, text, emoji=""):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {emoji} {text}"
        self.lines.append(line)
        self.buffer.write(line + "\n")
    
    def get_text(self):
        return "\n".join(self.lines)
    
    def clear(self):
        self.buffer = StringIO()
        self.lines = []


# ================================================================
# MODEL TRAINING
# ================================================================
def train_all_models(df, logger=None):
    df_model = df.dropna().copy()
    train_mask = df_model['Year'] <= 2019
    test_mask = df_model['Year'] >= 2020
    
    df_train = df_model[train_mask].copy()
    df_test = df_model[test_mask].copy()
    
    X_train = df_train[FEATURE_COLS]
    X_test = df_test[FEATURE_COLS]
    
    results = {'Energy': {}, 'CO2': {}}
    trained_models = {'Energy': {}, 'CO2': {}}
    
    for target in ['Energy', 'CO2']:
        y_train = df_train[target]
        y_test = df_test[target]
        
        if logger:
            logger.log(f"Training models for {target}...", "")
        
        for name, cfg in MODEL_CONFIG.items():
            model = cfg['model']
            try:
                if name == 'Neural Network':
                    scaler = StandardScaler()
                    X_tr_s = scaler.fit_transform(X_train)
                    X_te_s = scaler.transform(X_test)
                    model.fit(X_tr_s, y_train)
                    pred_train = model.predict(X_tr_s)
                    pred_test = model.predict(X_te_s)
                    trained_models[target][name] = {'model': model, 'scaler': scaler}
                else:
                    model.fit(X_train, y_train)
                    pred_train = model.predict(X_train)
                    pred_test = model.predict(X_test)
                    trained_models[target][name] = {'model': model}
                
                results[target][name] = {
                    'Train_R2': r2_score(y_train, pred_train),
                    'Test_R2': r2_score(y_test, pred_test),
                    'Train_RMSE': np.sqrt(mean_squared_error(y_train, pred_train)),
                    'Test_RMSE': np.sqrt(mean_squared_error(y_test, pred_test)),
                    'Train_MAE': mean_absolute_error(y_train, pred_train),
                    'Test_MAE': mean_absolute_error(y_test, pred_test),
                    'color': cfg['color']
                }
                if logger:
                    logger.log(f"  {name}: Test R2 = {results[target][name]['Test_R2']:.3f}", "  OK")
            except Exception as e:
                if logger:
                    logger.log(f"  {name}: FAILED - {str(e)}", "  FAIL")
                results[target][name] = {
                    'Train_R2': np.nan, 'Test_R2': np.nan,
                    'Train_RMSE': np.nan, 'Test_RMSE': np.nan,
                    'Train_MAE': np.nan, 'Test_MAE': np.nan,
                    'color': cfg['color'], 'error': str(e)
                }
    
    return results, trained_models, df_train, df_test


# ================================================================
# HELPER: MATPLOTLIB TO BASE64
# ================================================================
def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ================================================================
# INIT SESSION STATE
# ================================================================
def init_session():
    defaults = {
        'page': 'Overview',
        'df': None,
        'models_trained': False,
        'results': None,
        'trained_models': None,
        'df_train': None,
        'df_test': None,
        'logger': TerminalLogger(),
        'shap_explainer': None,
        'shap_values_energy': None,
        'shap_values_co2': None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v



# ================================================================
# PAGE 1: OVERVIEW
# ================================================================
def page_overview(df, results):
    st.markdown("<h1 style='margin-bottom:4px;'>ML Live Lab</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#86868b; font-size:1.05rem; margin-bottom:28px;'>Interactive machine learning deployment for China Tourism Energy & CO2 Research</p>", unsafe_allow_html=True)
    
    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        (f"{len(df):,}", "Observations", ""),
        (f"{df['Province'].nunique()}", "Provinces", ""),
        (f"{df['Year'].min():.0f}-{df['Year'].max():.0f}", "Time Span", ""),
    ]
    if results:
        energy_r2 = max([v['Test_R2'] for v in results['Energy'].values() if not np.isnan(v['Test_R2'])] + [0])
        co2_r2 = max([v['Test_R2'] for v in results['CO2'].values() if not np.isnan(v['Test_R2'])] + [0])
        metrics.append((f"{max(energy_r2, co2_r2):.3f}", "Best Test R2", f"CO2 model" if co2_r2 > energy_r2 else "Energy model"))
    else:
        metrics.append(("--", "Best Test R2", "Train models"))
    
    for col, (val, label, sub) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f"""
            <div class='apple-card'>
                <div class='apple-metric-value'>{val}</div>
                <div class='apple-metric-label'>{label}</div>
                {f"<div class='apple-metric-delta delta-positive'>{sub}</div>" if sub else ""}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Two columns: dataset preview + quick stats
    left, right = st.columns([3, 2])
    with left:
        st.markdown("<h2>Dataset Preview</h2>", unsafe_allow_html=True)
        preview = df[['Province', 'Year', 'Energy', 'CO2', 'Arrivals', 'TourismGDP', 'GreenTech', 'Governance', 'Urbanization']].head(8)
        st.dataframe(preview, use_container_width=True, hide_index=True)
    
    with right:
        st.markdown("<h2>Quick Stats</h2>", unsafe_allow_html=True)
        stats_df = df[['Energy', 'CO2', 'Arrivals', 'TourismGDP']].agg(['mean', 'std', 'min', 'max']).round(2)
        stats_df.index = ['Mean', 'Std Dev', 'Min', 'Max']
        st.dataframe(stats_df, use_container_width=True)
        
        st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='apple-card' style='padding:16px;'>
            <div style='font-size:0.85rem; color:#86868b; margin-bottom:6px;'>TEMPORAL SPLIT</div>
            <div style='font-size:1.1rem; font-weight:600; color:#1d1d1f;'>Train: 2008-2019</div>
            <div style='font-size:1.1rem; font-weight:600; color:#1d1d1f;'>Test: 2020-2023</div>
            <div style='font-size:0.8rem; color:#86868b; margin-top:6px;'>Honest evaluation. No data leakage.</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # National trends chart
    st.markdown("<h2>National Trends</h2>", unsafe_allow_html=True)
    national = df.groupby('Year')[['Energy', 'CO2', 'Arrivals', 'TourismGDP']].mean().reset_index()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=national['Year'], y=national['Energy'], mode='lines+markers', name='Energy (TJ)', line=dict(color='#0071e3', width=3), marker=dict(size=8)), secondary_y=False)
    fig.add_trace(go.Scatter(x=national['Year'], y=national['CO2'], mode='lines+markers', name='CO2 (Mt)', line=dict(color='#34c759', width=3), marker=dict(size=8)), secondary_y=False)
    fig.add_trace(go.Scatter(x=national['Year'], y=national['Arrivals'], mode='lines+markers', name='Arrivals', line=dict(color='#ff9500', width=2, dash='dot'), marker=dict(size=6)), secondary_y=True)
    fig.add_vline(x=2019.5, line_dash="dash", line_color="#ff3b30", opacity=0.5, annotation_text="COVID", annotation_position="top")
    fig.update_layout(
        template='plotly_white', height=420,
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        font=dict(family="-apple-system, BlinkMacSystemFont, SF Pro Display, Inter, sans-serif", color="#1d1d1f"),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_xaxes(showgrid=False, linecolor='#d2d2d7', linewidth=1)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.04)', linecolor='#d2d2d7', linewidth=1)
    st.plotly_chart(fig, use_container_width=True, key="overview_national_trends")


# ================================================================
# PAGE 2: MODEL ARENA
# ================================================================
def page_model_arena(df, results, trained_models, df_train, df_test, logger):
    st.markdown("<h1>Model Arena</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#86868b; font-size:1.05rem; margin-bottom:24px;'>Train, compare, and diagnose all 6 ML algorithms in real-time</p>", unsafe_allow_html=True)
    
    # Train button
    col_btn1, col_btn2, _ = st.columns([1, 1, 3])
    with col_btn1:
        if st.button("Train All Models", use_container_width=True):
            logger.clear()
            logger.log("Starting model training pipeline...", "")
            with st.spinner("Training 12 models (6 for Energy, 6 for CO2)..."):
                new_results, new_models, new_train, new_test = train_all_models(df, logger)
            st.session_state.results = new_results
            st.session_state.trained_models = new_models
            st.session_state.df_train = new_train
            st.session_state.df_test = new_test
            st.session_state.models_trained = True
            logger.log("All models trained successfully!", "")
            st.rerun()
    
    with col_btn2:
        if st.button("Retrain", use_container_width=True):
            st.session_state.models_trained = False
            st.session_state.results = None
            st.session_state.trained_models = None
            logger.clear()
            st.rerun()
    
    if not results:
        st.markdown("""
        <div class='section-card' style='text-align:center; padding:60px 20px;'>
            <div style='font-size:3rem; margin-bottom:16px;'></div>
            <h3 style='color:#86868b; font-weight:500;'>No models trained yet</h3>
            <p style='color:#86868b;'>Click "Train All Models" to begin the live training pipeline.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Terminal output
    with st.expander("  Live Training Terminal", expanded=True):
        st.markdown(f"""
        <div class='terminal-window'>
            <div class='terminal-header'>
                <div class='terminal-dot dot-red'></div>
                <div class='terminal-dot dot-yellow'></div>
                <div class='terminal-dot dot-green'></div>
                <span style='color:#86868b; margin-left:8px; font-size:0.75rem;'>ml-live-lab -- training.log</span>
            </div>
            <pre style='margin:0; color:#00ff41; line-height:1.6; font-weight:600;'>{logger.get_text()}</pre>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Copy Logs", key="copy_logs"):
            st.code(logger.get_text(), language="text")
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Results tabs
    tab1, tab2 = st.tabs(["  Energy Consumption", "  CO2 Emissions"])
    
    for target, tab in zip(['Energy', 'CO2'], [tab1, tab2]):
        with tab:
            target_results = results[target]
            
            # Winner banner
            valid = {k: v for k, v in target_results.items() if 'error' not in v}
            if valid:
                winner = max(valid.items(), key=lambda x: x[1]['Test_R2'])
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, {winner[1]["color"]}15, {winner[1]["color"]}05); border-radius:16px; padding:20px 24px; margin-bottom:20px; border: 1px solid {winner[1]["color"]}30;'>
                    <div style='font-size:0.8rem; color:#86868b; text-transform:uppercase; letter-spacing:0.08em; font-weight:600;'>Best Model</div>
                    <div style='font-size:1.8rem; font-weight:700; color:{winner[1]["color"]}; margin:4px 0;'>{winner[0]}</div>
                    <div style='font-size:0.95rem; color:#1d1d1f;'>Test R2 = <b>{winner[1]['Test_R2']:.3f}</b> &nbsp;|&nbsp; RMSE = <b>{winner[1]['Test_RMSE']:.1f}</b> &nbsp;|&nbsp; MAE = <b>{winner[1]['Test_MAE']:.1f}</b></div>
                </div>
                """, unsafe_allow_html=True)
            
            # Comparison table
            table_data = []
            for name, metrics in target_results.items():
                table_data.append({
                    'Model': name,
                    'Train R2': f"{metrics['Train_R2']:.3f}" if not np.isnan(metrics['Train_R2']) else "N/A",
                    'Test R2': f"{metrics['Test_R2']:.3f}" if not np.isnan(metrics['Test_R2']) else "N/A",
                    'Train RMSE': f"{metrics['Train_RMSE']:.1f}" if not np.isnan(metrics['Train_RMSE']) else "N/A",
                    'Test RMSE': f"{metrics['Test_RMSE']:.1f}" if not np.isnan(metrics['Test_RMSE']) else "N/A",
                    'Train MAE': f"{metrics['Train_MAE']:.1f}" if not np.isnan(metrics['Train_MAE']) else "N/A",
                    'Test MAE': f"{metrics['Test_MAE']:.1f}" if not np.isnan(metrics['Test_MAE']) else "N/A",
                })
            st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
            
            # Charts
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("<h3>Test R2 Comparison</h3>", unsafe_allow_html=True)
                chart_df = pd.DataFrame([
                    {'Model': name, 'Test R2': m['Test_R2'], 'Color': m['color']}
                    for name, m in target_results.items() if not np.isnan(m['Test_R2'])
                ])
                fig = px.bar(chart_df, x='Test R2', y='Model', orientation='h', color='Model',
                             color_discrete_map={row['Model']: row['Color'] for _, row in chart_df.iterrows()})
                fig.update_layout(template='plotly_white', height=320, showlegend=False,
                                  margin=dict(l=10, r=10, t=10, b=10),
                                  plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                  font=dict(family="Inter, sans-serif", color="#1d1d1f"))
                fig.update_xaxes(showgrid=True, gridcolor='rgba(0,0,0,0.04)', range=[0, 1], tickfont=dict(color='#1d1d1f'))
                fig.update_yaxes(showgrid=False, tickfont=dict(color='#1d1d1f', size=12))
                st.plotly_chart(fig, use_container_width=True, key=f"arena_r2_{target}")
            
            with c2:
                st.markdown("<h3>Test RMSE Comparison</h3>", unsafe_allow_html=True)
                chart_df2 = pd.DataFrame([
                    {'Model': name, 'Test RMSE': m['Test_RMSE'], 'Color': m['color']}
                    for name, m in target_results.items() if not np.isnan(m['Test_RMSE'])
                ])
                fig2 = px.bar(chart_df2, x='Test RMSE', y='Model', orientation='h', color='Model',
                              color_discrete_map={row['Model']: row['Color'] for _, row in chart_df2.iterrows()})
                fig2.update_layout(template='plotly_white', height=320, showlegend=False,
                                   margin=dict(l=10, r=10, t=10, b=10),
                                   plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                   font=dict(family="Inter, sans-serif", color="#1d1d1f"))
                fig2.update_xaxes(showgrid=True, gridcolor='rgba(0,0,0,0.04)', tickfont=dict(color='#1d1d1f'))
                fig2.update_yaxes(showgrid=False, tickfont=dict(color='#1d1d1f', size=12))
                st.plotly_chart(fig2, use_container_width=True, key=f"arena_rmse_{target}")


# ================================================================
# PAGE 3: PREDICTOR (What-If Simulator)
# ================================================================
def page_predictor(df, trained_models):
    st.markdown("<h1>What-If Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#86868b; font-size:1.05rem; margin-bottom:24px;'>Adjust policy variables and see predicted Energy & CO2 in real-time</p>", unsafe_allow_html=True)
    
    if not trained_models:
        st.warning("Train models in the **Model Arena** first.")
        return
    
    # Province and Year selection
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        province = st.selectbox("Province", sorted(df['Province'].unique()), key="pred_province")
    with col_sel2:
        year = st.selectbox("Year", sorted(df['Year'].unique()), key="pred_year")
    
    # Get historical data for this province-year
    row = df[(df['Province'] == province) & (df['Year'] == year)]
    if len(row) == 0:
        st.error("No data found for this province-year combination.")
        return
    row = row.iloc[0]
    
    # Sidebar-like card for sliders
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<h3>Policy Levers</h3>", unsafe_allow_html=True)
    
    sliders = {}
    cols = st.columns(3)
    for i, col_name in enumerate(POLICY_COLS):
        with cols[i % 3]:
            min_val = float(df[col_name].min())
            max_val = float(df[col_name].max())
            current = float(row[col_name])
            sliders[col_name] = st.slider(
                col_name, min_value=min_val, max_value=max_val,
                value=current, step=(max_val - min_val) / 100,
                help=f"Historical {col_name} for {province} in {year}: {current:.2f}"
            )
    
    # Lag values (user can override)
    st.markdown("<h3>Lag Values (Auto-filled from history)</h3>", unsafe_allow_html=True)
    lag_cols = st.columns(2)
    with lag_cols[0]:
        energy_lag = st.number_input("Energy_lag1", value=float(row['Energy_lag1']) if not pd.isna(row['Energy_lag1']) else 0.0)
    with lag_cols[1]:
        co2_lag = st.number_input("CO2_lag1", value=float(row['CO2_lag1']) if not pd.isna(row['CO2_lag1']) else 0.0)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Compute neighbor averages based on historical neighbor data for this year
    neighbors = ADJACENCY.get(province, [])
    neighbor_data = df[(df['Province'].isin(neighbors)) & (df['Year'] == year)]
    
    neighbor_energy = neighbor_data['Energy'].mean() if len(neighbor_data) > 0 else df[df['Year'] == year]['Energy'].mean()
    neighbor_co2 = neighbor_data['CO2'].mean() if len(neighbor_data) > 0 else df[df['Year'] == year]['CO2'].mean()
    neighbor_arrivals = neighbor_data['Arrivals'].mean() if len(neighbor_data) > 0 else df[df['Year'] == year]['Arrivals'].mean()
    neighbor_tourismgdp = neighbor_data['TourismGDP'].mean() if len(neighbor_data) > 0 else df[df['Year'] == year]['TourismGDP'].mean()
    
    # Build feature vector
    X_pred = pd.DataFrame([{
        'Arrivals': sliders['Arrivals'],
        'TourismGDP': sliders['TourismGDP'],
        'GreenTech': sliders['GreenTech'],
        'Governance': sliders['Governance'],
        'Urbanization': sliders['Urbanization'],
        'Year': year,
        'Energy_lag1': energy_lag,
        'CO2_lag1': co2_lag,
        'Energy_neighbor_avg': neighbor_energy,
        'CO2_neighbor_avg': neighbor_co2,
        'Arrivals_neighbor_avg': neighbor_arrivals,
        'TourismGDP_neighbor_avg': neighbor_tourismgdp,
    }])
    
    # Predictions
    preds = {'Energy': {}, 'CO2': {}}
    for target in ['Energy', 'CO2']:
        for name, cfg in trained_models[target].items():
            model = cfg['model']
            if 'scaler' in cfg:
                X_s = cfg['scaler'].transform(X_pred)
                preds[target][name] = model.predict(X_s)[0]
            else:
                preds[target][name] = model.predict(X_pred)[0]
    
    # Display predictions in big cards
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h2>Live Predictions</h2>", unsafe_allow_html=True)
    
    pcol1, pcol2 = st.columns(2)
    
    with pcol1:
        energy_winner = max(preds['Energy'].items(), key=lambda x: x[1] if not np.isnan(x[1]) else -999)[0]
        energy_val = np.mean(list(preds['Energy'].values()))  # ensemble average
        energy_baseline = row['Energy']
        energy_change = ((energy_val - energy_baseline) / energy_baseline * 100) if energy_baseline != 0 else 0
        
        st.markdown(f"""
        <div class='apple-card' style='border-top: 4px solid #0071e3;'>
            <div style='font-size:0.8rem; color:#86868b; text-transform:uppercase; letter-spacing:0.08em; font-weight:600; margin-bottom:8px;'>Predicted Energy</div>
            <div class='apple-metric-value' style='font-size:3rem;'>{energy_val:,.0f}</div>
            <div style='font-size:0.9rem; color:#86868b; margin-top:4px;'>TJ (terajoules)</div>
            <div style='margin-top:12px; padding-top:12px; border-top:1px solid #f0f0f0;'>
                <div style='font-size:0.85rem;'><span style='color:#86868b;'>Baseline:</span> <b>{energy_baseline:,.0f}</b> &nbsp;|&nbsp; <span class="{'delta-positive' if energy_change < 0 else 'delta-negative'}">{energy_change:+.1f}%</span></div>
                <div style='font-size:0.8rem; color:#86868b; margin-top:4px;'>Ensemble of {len(preds['Energy'])} models</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Energy model breakdown
        energy_df = pd.DataFrame([{'Model': k, 'Prediction': v} for k, v in preds['Energy'].items()])
        fig_e = px.bar(energy_df, x='Prediction', y='Model', orientation='h', color='Model',
                       color_discrete_map={n: MODEL_CONFIG[n]['color'] for n in energy_df['Model']})
        fig_e.update_layout(template='plotly_white', height=260, showlegend=False,
                            margin=dict(l=10, r=10, t=10, b=10),
                            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(family="Inter, sans-serif", color="#1d1d1f"))
        fig_e.update_xaxes(showgrid=True, gridcolor='rgba(0,0,0,0.04)', tickfont=dict(color='#1d1d1f'))
        fig_e.update_yaxes(showgrid=False, tickfont=dict(color='#1d1d1f', size=12))
        st.plotly_chart(fig_e, use_container_width=True, key="pred_energy_bar")
    
    with pcol2:
        co2_val = np.mean(list(preds['CO2'].values()))
        co2_baseline = row['CO2']
        co2_change = ((co2_val - co2_baseline) / co2_baseline * 100) if co2_baseline != 0 else 0
        
        st.markdown(f"""
        <div class='apple-card' style='border-top: 4px solid #34c759;'>
            <div style='font-size:0.8rem; color:#86868b; text-transform:uppercase; letter-spacing:0.08em; font-weight:600; margin-bottom:8px;'>Predicted CO2</div>
            <div class='apple-metric-value' style='font-size:3rem;'>{co2_val:.2f}</div>
            <div style='font-size:0.9rem; color:#86868b; margin-top:4px;'>Million tonnes</div>
            <div style='margin-top:12px; padding-top:12px; border-top:1px solid #f0f0f0;'>
                <div style='font-size:0.85rem;'><span style='color:#86868b;'>Baseline:</span> <b>{co2_baseline:.2f}</b> &nbsp;|&nbsp; <span class="{'delta-positive' if co2_change < 0 else 'delta-negative'}">{co2_change:+.1f}%</span></div>
                <div style='font-size:0.8rem; color:#86868b; margin-top:4px;'>Ensemble of {len(preds['CO2'])} models</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        co2_df = pd.DataFrame([{'Model': k, 'Prediction': v} for k, v in preds['CO2'].items()])
        fig_c = px.bar(co2_df, x='Prediction', y='Model', orientation='h', color='Model',
                       color_discrete_map={n: MODEL_CONFIG[n]['color'] for n in co2_df['Model']})
        fig_c.update_layout(template='plotly_white', height=260, showlegend=False,
                            margin=dict(l=10, r=10, t=10, b=10),
                            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(family="Inter, sans-serif", color="#1d1d1f"))
        fig_c.update_xaxes(showgrid=True, gridcolor='rgba(0,0,0,0.04)', tickfont=dict(color='#1d1d1f'))
        fig_c.update_yaxes(showgrid=False, tickfont=dict(color='#1d1d1f', size=12))
        st.plotly_chart(fig_c, use_container_width=True, key="pred_co2_bar")
    
    # Policy impact analysis
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h2>Policy Impact Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#86868b; font-size:0.9rem;'>How much does each lever change the prediction vs. the historical baseline?</p>", unsafe_allow_html=True)
    
    impact_data = []
    x_base = pd.DataFrame([{
        'Arrivals': row['Arrivals'], 'TourismGDP': row['TourismGDP'], 'GreenTech': row['GreenTech'],
        'Governance': row['Governance'], 'Urbanization': row['Urbanization'], 'Year': year,
        'Energy_lag1': energy_lag, 'CO2_lag1': co2_lag,
        'Energy_neighbor_avg': neighbor_energy, 'CO2_neighbor_avg': neighbor_co2,
        'Arrivals_neighbor_avg': neighbor_arrivals, 'TourismGDP_neighbor_avg': neighbor_tourismgdp,
    }])
    
    xgb_e = trained_models['Energy']['XGBoost']['model']
    xgb_c = trained_models['CO2']['XGBoost']['model']
    base_e = xgb_e.predict(x_base)[0]
    base_c = xgb_c.predict(x_base)[0]
    
    for col in POLICY_COLS:
        x_mod = x_base.copy()
        x_mod[col] = sliders[col]
        impact_e = xgb_e.predict(x_mod)[0] - base_e
        impact_c = xgb_c.predict(x_mod)[0] - base_c
        impact_data.append({
            'Variable': col, 'Energy Impact': impact_e, 'CO2 Impact': impact_c,
            'Changed': sliders[col] != row[col]
        })
    
    impact_df = pd.DataFrame(impact_data)
    
    fig_impact = go.Figure()
    fig_impact.add_trace(go.Bar(name='Energy Impact', x=impact_df['Variable'], y=impact_df['Energy Impact'],
                                 marker_color=['#0071e3' if c else '#d2d2d7' for c in impact_df['Changed']]))
    fig_impact.add_trace(go.Bar(name='CO2 Impact', x=impact_df['Variable'], y=impact_df['CO2 Impact'],
                                 marker_color=['#34c759' if c else '#e8e8ed' for c in impact_df['Changed']]))
    fig_impact.update_layout(template='plotly_white', barmode='group', height=380,
                              margin=dict(l=10, r=10, t=10, b=10),
                              plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                              legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                              font=dict(family="Inter, sans-serif", color="#1d1d1f"))
    fig_impact.update_xaxes(showgrid=False, linecolor='#d2d2d7')
    fig_impact.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.04)', linecolor='#d2d2d7')
    st.plotly_chart(fig_impact, use_container_width=True, key="pred_impact")



# ================================================================
# PAGE 4: FUTURE FORECAST
# ================================================================
def page_forecast(df, trained_models):
    st.markdown("<h1>Future Forecast</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#86868b; font-size:1.05rem; margin-bottom:24px;'>Project Energy & CO2 into the future using iterative ML forecasting</p>", unsafe_allow_html=True)
    
    if not trained_models:
        st.warning("Train models in the **Model Arena** first.")
        return
    
    province = st.selectbox("Select Province", sorted(df['Province'].unique()), key="fc_province")
    target_year = st.slider("Target Year", min_value=2024, max_value=2030, value=2026, key="fc_year")
    
    st.markdown("<h3>Annual Growth Assumptions (% per year)</h3>", unsafe_allow_html=True)
    growth_cols = st.columns(5)
    growth_rates = {}
    for i, col in enumerate(POLICY_COLS):
        with growth_cols[i]:
            growth_rates[col] = st.slider(f"{col} growth", min_value=-10.0, max_value=20.0, value=3.0, step=0.5, key=f"gr_{col}") / 100
    
    # Get 2023 actuals as starting point
    start_row = df[(df['Province'] == province) & (df['Year'] == 2023)]
    if len(start_row) == 0:
        st.error("No 2023 data available for this province.")
        return
    start_row = start_row.iloc[0]
    
    # Iterative forecasting
    years = list(range(2024, target_year + 1))
    forecast = []
    
    prev_energy = start_row['Energy']
    prev_co2 = start_row['CO2']
    current_values = {col: start_row[col] for col in POLICY_COLS}
    
    for yr in years:
        # Project policy variables
        for col in POLICY_COLS:
            current_values[col] *= (1 + growth_rates[col])
        
        # Compute neighbor values (assume neighbors grow at national average rate)
        neighbors = ADJACENCY.get(province, [])
        neighbor_2023 = df[(df['Province'].isin(neighbors)) & (df['Year'] == 2023)]
        if len(neighbor_2023) > 0:
            neighbor_energy = neighbor_2023['Energy'].mean() * (1 + growth_rates['Arrivals']) ** (yr - 2023)
            neighbor_co2 = neighbor_2023['CO2'].mean() * (1 + growth_rates['Arrivals']) ** (yr - 2023)
            neighbor_arrivals = neighbor_2023['Arrivals'].mean() * (1 + growth_rates['Arrivals']) ** (yr - 2023)
            neighbor_tourismgdp = neighbor_2023['TourismGDP'].mean() * (1 + growth_rates['TourismGDP']) ** (yr - 2023)
        else:
            national_2023 = df[df['Year'] == 2023]
            neighbor_energy = national_2023['Energy'].mean() * (1 + growth_rates['Arrivals']) ** (yr - 2023)
            neighbor_co2 = national_2023['CO2'].mean() * (1 + growth_rates['Arrivals']) ** (yr - 2023)
            neighbor_arrivals = national_2023['Arrivals'].mean() * (1 + growth_rates['Arrivals']) ** (yr - 2023)
            neighbor_tourismgdp = national_2023['TourismGDP'].mean() * (1 + growth_rates['TourismGDP']) ** (yr - 2023)
        
        X_fc = pd.DataFrame([{
            'Arrivals': current_values['Arrivals'],
            'TourismGDP': current_values['TourismGDP'],
            'GreenTech': current_values['GreenTech'],
            'Governance': current_values['Governance'],
            'Urbanization': current_values['Urbanization'],
            'Year': yr,
            'Energy_lag1': prev_energy,
            'CO2_lag1': prev_co2,
            'Energy_neighbor_avg': neighbor_energy,
            'CO2_neighbor_avg': neighbor_co2,
            'Arrivals_neighbor_avg': neighbor_arrivals,
            'TourismGDP_neighbor_avg': neighbor_tourismgdp,
        }])
        
        pred_e = trained_models['Energy']['XGBoost']['model'].predict(X_fc)[0]
        pred_c = trained_models['CO2']['XGBoost']['model'].predict(X_fc)[0]
        
        forecast.append({
            'Year': yr, 'Energy': pred_e, 'CO2': pred_c,
            **{f'{k}': v for k, v in current_values.items()}
        })
        
        prev_energy = pred_e
        prev_co2 = pred_c
    
    forecast_df = pd.DataFrame(forecast)
    
    # Historical data for this province
    hist = df[df['Province'] == province][['Year', 'Energy', 'CO2']].copy()
    hist['Type'] = 'Historical'
    forecast_df['Type'] = 'Projected'
    
    combined = pd.concat([
        hist[['Year', 'Energy', 'CO2', 'Type']],
        forecast_df[['Year', 'Energy', 'CO2', 'Type']]
    ], ignore_index=True)
    
    # Plot
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f'{province} - Energy Forecast', f'{province} - CO2 Forecast'))
    
    hist_e = hist[hist['Year'] <= 2023]
    proj_e = forecast_df
    fig.add_trace(go.Scatter(x=hist_e['Year'], y=hist_e['Energy'], mode='lines+markers', name='Historical Energy',
                              line=dict(color='#0071e3', width=3), marker=dict(size=8), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=proj_e['Year'], y=proj_e['Energy'], mode='lines+markers', name='Projected Energy',
                              line=dict(color='#0071e3', width=3, dash='dash'), marker=dict(size=8, symbol='diamond'), showlegend=False), row=1, col=1)
    
    hist_c = hist[hist['Year'] <= 2023]
    proj_c = forecast_df
    fig.add_trace(go.Scatter(x=hist_c['Year'], y=hist_c['CO2'], mode='lines+markers', name='Historical CO2',
                              line=dict(color='#34c759', width=3), marker=dict(size=8), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=proj_c['Year'], y=proj_c['CO2'], mode='lines+markers', name='Projected CO2',
                              line=dict(color='#34c759', width=3, dash='dash'), marker=dict(size=8, symbol='diamond'), showlegend=False), row=1, col=2)
    
    fig.add_vline(x=2023.5, line_dash="dot", line_color="#ff3b30", opacity=0.5, row=1, col=1)
    fig.add_vline(x=2023.5, line_dash="dot", line_color="#ff3b30", opacity=0.5, row=1, col=2)
    
    fig.update_layout(
        template='plotly_white', height=450,
        margin=dict(l=40, r=40, t=60, b=40),
        font=dict(family="Inter, sans-serif", color="#1d1d1f"),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        showlegend=True, legend=dict(orientation='h', yanchor='bottom', y=-0.18, xanchor='center', x=0.5)
    )
    fig.update_xaxes(showgrid=False, linecolor='#d2d2d7')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.04)', linecolor='#d2d2d7')
    st.plotly_chart(fig, use_container_width=True, key="forecast_chart")
    
    # Projection table
    st.markdown("<h3>Projection Details</h3>", unsafe_allow_html=True)
    display_df = forecast_df[['Year', 'Energy', 'CO2', 'Arrivals', 'TourismGDP', 'GreenTech', 'Governance', 'Urbanization']].copy()
    for col in ['Energy', 'CO2', 'Arrivals', 'TourismGDP']:
        display_df[col] = display_df[col].round(2 if col == 'CO2' else 0)
    for col in ['GreenTech', 'Governance', 'Urbanization']:
        display_df[col] = display_df[col].round(3)
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ================================================================
# PAGE 5: SPATIAL EXPLORER
# ================================================================
def page_spatial(df):
    st.markdown("<h1>Spatial Explorer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#86868b; font-size:1.05rem; margin-bottom:24px;'>Explore geographic spillovers between adjacent provinces</p>", unsafe_allow_html=True)
    
    province = st.selectbox("Select Province", sorted(df['Province'].unique()), key="spat_province")
    year = st.selectbox("Select Year", sorted(df['Year'].unique()), key="spat_year")
    
    neighbors = ADJACENCY.get(province, [])
    
    st.markdown(f"""
    <div class='apple-card' style='margin-bottom:20px;'>
        <div style='font-size:0.8rem; color:#86868b; text-transform:uppercase; letter-spacing:0.08em; font-weight:600;'>Geographic Neighbors</div>
        <div style='font-size:1.3rem; font-weight:600; color:#1d1d1f; margin-top:6px;'>{', '.join(neighbors) if neighbors else 'No land neighbors (island/border)'}</div>
        <div style='font-size:0.85rem; color:#86868b; margin-top:4px;'>{len(neighbors)} adjacent province{'s' if len(neighbors) != 1 else ''}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Selected province vs neighbor average
    prov_row = df[(df['Province'] == province) & (df['Year'] == year)]
    neighbor_rows = df[(df['Province'].isin(neighbors)) & (df['Year'] == year)]
    
    if len(prov_row) == 0 or len(neighbor_rows) == 0:
        st.warning("Insufficient data for this selection.")
        return
    
    prov_row = prov_row.iloc[0]
    
    metrics = ['Energy', 'CO2', 'Arrivals', 'TourismGDP', 'GreenTech', 'Governance', 'Urbanization']
    prov_vals = [prov_row[m] for m in metrics]
    neighbor_vals = [neighbor_rows[m].mean() for m in metrics]
    
    # Radar chart
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=prov_vals + [prov_vals[0]],
        theta=metrics + [metrics[0]],
        fill='toself',
        name=province,
        line=dict(color='#0071e3', width=3),
        fillcolor='rgba(0,113,227,0.15)'
    ))
    fig.add_trace(go.Scatterpolar(
        r=neighbor_vals + [neighbor_vals[0]],
        theta=metrics + [metrics[0]],
        fill='toself',
        name=f'Neighbor Average ({len(neighbor_rows)} provinces)',
        line=dict(color='#34c759', width=3),
        fillcolor='rgba(52,199,89,0.15)'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(max(prov_vals), max(neighbor_vals)) * 1.1], linecolor='#d2d2d7', gridcolor='rgba(0,0,0,0.04)')),
        template='plotly_white', height=500,
        font=dict(family="Inter, sans-serif", color="#1d1d1f"),
        legend=dict(orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5),
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True, key="spatial_radar")
    
    # Neighbor detail table
    st.markdown("<h3>Neighbor Breakdown</h3>", unsafe_allow_html=True)
    neighbor_detail = neighbor_rows[['Province', 'Energy', 'CO2', 'Arrivals', 'TourismGDP']].copy()
    neighbor_detail = neighbor_detail.round({'Energy': 0, 'CO2': 2, 'Arrivals': 0, 'TourismGDP': 0})
    st.dataframe(neighbor_detail, use_container_width=True, hide_index=True)
    
    # Bar chart: province vs neighbors
    comparison = pd.DataFrame({
        'Metric': metrics,
        province: prov_vals,
        'Neighbor Average': neighbor_vals
    })
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(name=province, x=metrics, y=prov_vals, marker_color='#0071e3'))
    fig_bar.add_trace(go.Bar(name='Neighbor Average', x=metrics, y=neighbor_vals, marker_color='#34c759'))
    fig_bar.update_layout(
        template='plotly_white', barmode='group', height=400,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        font=dict(family="Inter, sans-serif", color="#1d1d1f")
    )
    fig_bar.update_xaxes(showgrid=False, linecolor='#d2d2d7')
    fig_bar.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.04)', linecolor='#d2d2d7')
    st.plotly_chart(fig_bar, use_container_width=True, key="spatial_bar")


# ================================================================
# PAGE 6: EXPLAINABILITY (SHAP)
# ================================================================
def page_explainability(df, trained_models, df_train, df_test):
    st.markdown("<h1>Explainability</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#86868b; font-size:1.05rem; margin-bottom:24px;'>Understand why the model makes each prediction using SHAP</p>", unsafe_allow_html=True)
    
    if not trained_models or 'XGBoost' not in trained_models.get('Energy', {}):
        st.warning("Train XGBoost models in the **Model Arena** first.")
        return
    
    target = st.radio("Target Variable", ['Energy', 'CO2'], horizontal=True, key="shap_target")
    
    xgb_model = trained_models[target]['XGBoost']['model']
    X_test = df_test[FEATURE_COLS]
    y_test = df_test[target]
    
    # Compute SHAP if not cached
    if st.session_state.get('shap_explainer') is None or st.session_state.get('shap_target_cached') != target:
        with st.spinner("Computing SHAP values..."):
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(X_test)
            st.session_state.shap_explainer = explainer
            st.session_state.shap_values = shap_values
            st.session_state.shap_target_cached = target
    else:
        explainer = st.session_state.shap_explainer
        shap_values = st.session_state.shap_values
    
    # Global summary
    st.markdown("<h2>Global Feature Importance</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#86868b; font-size:0.9rem;'>Which features matter most across all predictions?</p>", unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    shap.summary_plot(shap_values, X_test, feature_names=FEATURE_COLS, show=False, plot_size=(10, 6))
    ax.set_facecolor('white')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()
    
    # Bar chart version
    fig2, ax2 = plt.subplots(figsize=(10, 5), facecolor='white')
    shap.summary_plot(shap_values, X_test, feature_names=FEATURE_COLS, plot_type="bar", show=False, plot_size=(10, 5))
    ax2.set_facecolor('white')
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close()
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Individual prediction breakdown
    st.markdown("<h2>Individual Prediction Breakdown</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#86868b; font-size:0.9rem;'>Select a specific province-year to see why the model predicted that value.</p>", unsafe_allow_html=True)
    
    idx = st.slider("Select observation (test set index)", 0, len(df_test) - 1, 0, key="shap_idx")
    
    row_info = df_test.iloc[idx]
    actual = row_info[target]
    predicted = xgb_model.predict(X_test.iloc[idx:idx+1])[0]
    
    st.markdown(f"""
    <div class='apple-card' style='margin-bottom:20px;'>
        <div style='display:flex; justify-content:space-between; align-items:center;'>
            <div>
                <div style='font-size:0.8rem; color:#86868b; text-transform:uppercase; letter-spacing:0.08em; font-weight:600;'>Selected Observation</div>
                <div style='font-size:1.3rem; font-weight:600; color:#1d1d1f; margin-top:4px;'>{row_info['Province']} &middot; {row_info['Year']:.0f}</div>
            </div>
            <div style='text-align:right;'>
                <div style='font-size:0.8rem; color:#86868b; text-transform:uppercase; letter-spacing:0.08em; font-weight:600;'>Actual vs Predicted</div>
                <div style='font-size:1.3rem; font-weight:600; color:#1d1d1f; margin-top:4px;'>{actual:.1f} vs {predicted:.1f}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Waterfall plot
    fig3, ax3 = plt.subplots(figsize=(12, 5), facecolor='white')
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[idx],
        base_values=explainer.expected_value,
        data=X_test.iloc[idx].values,
        feature_names=FEATURE_COLS
    ), show=False)
    ax3.set_facecolor('white')
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close()
    
    # Feature values table
    st.markdown("<h3>Feature Values for This Observation</h3>", unsafe_allow_html=True)
    feat_df = pd.DataFrame({
        'Feature': FEATURE_COLS,
        'Value': X_test.iloc[idx].values,
        'SHAP Impact': shap_values[idx]
    })
    feat_df['Impact Direction'] = feat_df['SHAP Impact'].apply(lambda x: 'Increases prediction' if x > 0 else 'Decreases prediction')
    feat_df = feat_df.sort_values('SHAP Impact', key=abs, ascending=False)
    st.dataframe(feat_df.round(3), use_container_width=True, hide_index=True)


# ================================================================
# PAGE 7: TERMINAL
# ================================================================
def page_terminal(logger):
    st.markdown("<h1>Terminal</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#86868b; font-size:1.05rem; margin-bottom:24px;'>Transparent, real-time logs from every model training run</p>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class='terminal-window' style='min-height:500px;'>
        <div class='terminal-header'>
            <div class='terminal-dot dot-red'></div>
            <div class='terminal-dot dot-yellow'></div>
            <div class='terminal-dot dot-green'></div>
            <span style='color:#86868b; margin-left:8px; font-size:0.75rem;'>ml-live-lab -- training.log</span>
        </div>
        <pre style='margin:0; color:#00ff41; line-height:1.7; white-space:pre-wrap; word-wrap:break-word; font-weight:600;'>{logger.get_text() if logger.get_text() else '[No logs yet. Train models in Model Arena to see output.]'}</pre>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, _ = st.columns([1, 1, 4])
    with col1:
        if st.button("Copy All Logs", use_container_width=True):
            st.code(logger.get_text() if logger.get_text() else "No logs yet.", language="text")
    with col2:
        if st.button("Clear Logs", use_container_width=True):
            logger.clear()
            st.rerun()



# ================================================================
# MAIN APP
# ================================================================
def main():
    init_session()
    
    # Load data
    if st.session_state.df is None:
        with st.spinner("Loading dataset..."):
            st.session_state.df = load_and_engineer_data()
            st.session_state.logger.log("Dataset loaded: 30 provinces x 16 years = 480 observations", "")
            st.session_state.logger.log("Spatial neighbor features engineered", "")
            st.session_state.logger.log("Lag features created (Energy_lag1, CO2_lag1)", "")
    
    df = st.session_state.df
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; margin-bottom:24px; padding-bottom:20px; border-bottom:1px solid rgba(0,0,0,0.06);'>
            <div style='font-size:1.6rem; font-weight:700; color:#1d1d1f; letter-spacing:-0.02em;'>ML Live Lab</div>
            <div style='font-size:0.8rem; color:#86868b; margin-top:4px;'>Dr. Danish Research</div>
        </div>
        """, unsafe_allow_html=True)
        
        pages = [
            "Overview",
            "Model Arena",
            "Predictor",
            "Future Forecast",
            "Spatial Explorer",
            "Explainability",
            "Terminal"
        ]
        
        for page_name in pages:
            active = st.session_state.page == page_name
            btn_type = "primary" if active else "secondary"
            if st.button(page_name, use_container_width=True, type=btn_type, key=f"nav_{page_name}"):
                st.session_state.page = page_name
                st.rerun()
        
        st.markdown("<div style='margin-top:40px;'></div>", unsafe_allow_html=True)
        
        # Status indicator
        status_color = "#34c759" if st.session_state.models_trained else "#ff9500"
        status_text = "Models Ready" if st.session_state.models_trained else "Models Not Trained"
        st.markdown(f"""
        <div style='background-color: var(--secondary-background-color); border-radius:12px; padding:14px; margin-top:20px; border: 1px solid rgba(128,128,128,0.2);'>
            <div style='font-size:0.75rem; color: var(--text-color); opacity: 0.7; text-transform:uppercase; letter-spacing:0.06em; font-weight:600; margin-bottom:6px;'>System Status</div>
            <div style='display:flex; align-items:center; gap:8px;'>
                <div style='width:8px; height:8px; border-radius:50%; background:{status_color};'></div>
                <div style='font-size:0.9rem; font-weight:600; color: var(--text-color);'>{status_text}</div>
            </div>
            <div style='font-size:0.75rem; color: var(--text-color); opacity: 0.7; margin-top:4px;'>{len(df):,} rows loaded</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div class='footer-text' style='margin-top:60px;'>Built with Streamlit<br>Apple-inspired Design</div>", unsafe_allow_html=True)
    
    # Route to page
    page = st.session_state.page
    
    if page == "Overview":
        page_overview(df, st.session_state.results)
    elif page == "Model Arena":
        page_model_arena(df, st.session_state.results, st.session_state.trained_models,
                         st.session_state.df_train, st.session_state.df_test, st.session_state.logger)
    elif page == "Predictor":
        page_predictor(df, st.session_state.trained_models)
    elif page == "Future Forecast":
        page_forecast(df, st.session_state.trained_models)
    elif page == "Spatial Explorer":
        page_spatial(df)
    elif page == "Explainability":
        page_explainability(df, st.session_state.trained_models, st.session_state.df_train, st.session_state.df_test)
    elif page == "Terminal":
        page_terminal(st.session_state.logger)


if __name__ == "__main__":
    main()
