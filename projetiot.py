# ==================================================================================
# SYSTÈME INTELLIGENT D'ANALYSE ET D'OPTIMISATION ÉNERGÉTIQUE POUR CAFÉ
# Application Streamlit Complète | Master/Ingénieur | 2025
# ==================================================================================
# Auteur : Expert IA & Data Science
# Objectif : Analyse, prédiction et optimisation énergétique basée sur données CSV
# ==================================================================================

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import io
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import warnings
warnings.filterwarnings('ignore')

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# ===== MACHINE LEARNING AVANCÉ =====
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from scipy import stats

# ==================================================================================
# CONFIGURATION STREAMLIT
# ==================================================================================

st.set_page_config(
    page_title="EnergieCafé IA - Dashboard Intelligent",
    page_icon="EC",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={}
)

# CSS personnalisé pour design professionnel
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    h1, h2, h3, h4, h5, h6 {
        color: #1a1a1a !important;
        font-weight: bold !important;
    }
    .metric-container {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .alert-critical {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        font-weight: bold;
    }
    .alert-warning {
        background: linear-gradient(135deg, #ffa502 0%, #ff6b35 100%);
        color: white;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        font-weight: bold;
    }
    .alert-success {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        color: white;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        font-weight: bold;
    }
    .stMarkdown p, .stMarkdown h4 {
        color: #1a1a1a !important;
    }
</style>
""", unsafe_allow_html=True)

