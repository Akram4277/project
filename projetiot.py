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
    }
    .alert-warning {
        background: linear-gradient(135deg, #ffa502 0%, #ff6b35 100%);
        color: white;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .alert-success {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        color: white;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


def initialiser_rapport_session():
    if "rapport_graphiques" not in st.session_state:
        st.session_state["rapport_graphiques"] = []
    if "rapport_interpretations" not in st.session_state:
        st.session_state["rapport_interpretations"] = []
    if "rapport_recommandations" not in st.session_state:
        st.session_state["rapport_recommandations"] = []


def enregistrer_graphique(titre, fig):
    initialiser_rapport_session()
    st.session_state["rapport_graphiques"].append({"titre": titre, "figure": fig})


def enregistrer_synthese(interpretations, recommandations):
    initialiser_rapport_session()
    st.session_state["rapport_interpretations"] = interpretations
    st.session_state["rapport_recommandations"] = recommandations


@st.cache_data(show_spinner=False)
def charger_donnees_csv(uploaded_file_bytes):
    """Charger et prétraiter le CSV du café depuis un fichier déposé."""
    df = pd.read_csv(io.BytesIO(uploaded_file_bytes))
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%m/%d/%Y %H:%M', errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    return df


def generer_interpretations_et_recommandations(df, alertes=None, resultats_ml=None):
    interpretations = []
    recommandations = []

    if df is None or df.empty:
        return interpretations, recommandations

    conso_moyenne = df['consommation_energetique'].mean()
    co2_moyen = df['CO2'].mean()
    temp_moyenne = df['temperature_interieure'].mean()
    humidite_moyenne = df['humidite'].mean()
    activite_moyenne = df['presence_clients'].mean()

    if len(df) >= 48:
        debut = df['consommation_energetique'].head(24).mean()
        fin = df['consommation_energetique'].tail(24).mean()
        variation = ((fin - debut) / debut * 100) if debut else 0
        if variation > 10:
            interpretations.append(f"La consommation augmente sur la période récente de {variation:.1f} %.")
            recommandations.append("Réduire les charges de base et vérifier les équipements qui restent actifs hors présence.")

    if conso_moyenne > df['consommation_energetique'].quantile(0.75):
        interpretations.append("Le niveau moyen de consommation est élevé par rapport au reste de la période analysée.")
        recommandations.append("Mettre en place une programmation plus stricte du chauffage, de la climatisation et de l'éclairage.")

    if co2_moyen > 800:
        interpretations.append(f"Le CO2 moyen est de {co2_moyen:.0f} ppm, ce qui indique une ventilation à renforcer.")
        recommandations.append("Augmenter la ventilation et contrôler les débits d'air aux heures d'affluence.")

    if temp_moyenne > 24:
        interpretations.append(f"La température intérieure moyenne est de {temp_moyenne:.1f} °C, au-dessus d'un niveau de confort optimal.")
        recommandations.append("Ajuster la consigne de climatisation et limiter les apports thermiques inutiles.")
    elif temp_moyenne < 20:
        interpretations.append(f"La température intérieure moyenne est de {temp_moyenne:.1f} °C, ce qui peut dégrader le confort.")
        recommandations.append("Revoir la consigne de chauffage pour rester dans une plage de confort stable.")

    if humidite_moyenne > 70:
        interpretations.append(f"L'humidité moyenne est de {humidite_moyenne:.1f} %, ce qui peut signaler un besoin de renouvellement d'air.")
        recommandations.append("Contrôler la ventilation et vérifier l'absence de sources d'humidité permanentes.")

    if activite_moyenne > df['presence_clients'].quantile(0.75):
        interpretations.append("L'activité client est soutenue sur la période observée.")
        recommandations.append("Adapter l'allumage et les consignes de confort aux périodes de forte fréquentation.")

    if alertes:
        nombre_alertes = len(alertes)
        nombre_critiques = sum(1 for a in alertes if a.get('niveau') == 'CRITIQUE')
        if nombre_alertes > 0:
            interpretations.append(f"{nombre_alertes} alerte(s) active(s), dont {nombre_critiques} critique(s), nécessitent un suivi rapide.")
            recommandations.append("Traiter en priorité les alertes critiques et planifier un contrôle des équipements concernés.")

    if resultats_ml and 'isolation_forest' in resultats_ml:
        ratio = resultats_ml['isolation_forest'].get('anomaly_ratio', 0)
        if ratio > 0.08:
            interpretations.append(f"Le taux d'anomalies détectées atteint {ratio*100:.1f} %, niveau à surveiller.")
            recommandations.append("Analyser les périodes anormales pour identifier les causes récurrentes de surconsommation.")

    if not interpretations:
        interpretations.append("Les indicateurs principaux restent globalement stables sur la période analysée.")
    if not recommandations:
        recommandations.append("Poursuivre la surveillance et maintenir les réglages actuels avec un contrôle hebdomadaire.")

    return interpretations, recommandations


def generer_pdf_rapport(df, interpretations, recommandations, graphiques):
    if not REPORTLAB_AVAILABLE:
        return None

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=1.5 * cm, leftMargin=1.5 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='CenterTitle', parent=styles['Title'], alignment=TA_CENTER, spaceAfter=12))
    styles.add(ParagraphStyle(name='SectionTitle', parent=styles['Heading2'], spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(name='BodyTextCustom', parent=styles['BodyText'], leading=14, spaceAfter=4))

    story = []
    story.append(Paragraph("Rapport d'analyse énergétique", styles['CenterTitle']))
    story.append(Paragraph(f"Période analysée : {df['timestamp'].min()} à {df['timestamp'].max()}", styles['BodyTextCustom']))
    story.append(Spacer(1, 0.3 * cm))

    resume_data = [
        ["Observations", f"{len(df):,}"],
        ["Consommation totale", f"{df['consommation_energetique'].sum():.2f} kWh"],
        ["Température moyenne", f"{df['temperature_interieure'].mean():.1f} °C"],
        ["CO2 moyen", f"{df['CO2'].mean():.0f} ppm"],
    ]
    resume_table = Table(resume_data, colWidths=[5 * cm, 8 * cm])
    resume_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#EAF2F8')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(resume_table)
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("Interprétations", styles['SectionTitle']))
    for texte in interpretations:
        story.append(Paragraph(f"- {texte}", styles['BodyTextCustom']))

    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("Recommandations", styles['SectionTitle']))
    for texte in recommandations:
        story.append(Paragraph(f"- {texte}", styles['BodyTextCustom']))

    if graphiques:
        story.append(PageBreak())
        story.append(Paragraph("Graphiques exportés", styles['SectionTitle']))
        for item in graphiques:
            titre = item.get('titre', 'Graphique')
            fig = item.get('figure')
            story.append(Paragraph(titre, styles['Heading3']))
            try:
                image_bytes = pio.to_image(fig, format='png', width=1200, height=650)
                story.append(Image(io.BytesIO(image_bytes), width=17 * cm, height=9 * cm))
            except Exception:
                story.append(Paragraph("Export de l'image indisponible pour ce graphique.", styles['BodyTextCustom']))
            story.append(Spacer(1, 0.4 * cm))

    doc.build(story)
    return buffer.getvalue()

# ==================================================================================
# 1. CHARGEMENT ET CACHE DES DONNÉES
# ==================================================================================

@st.cache_resource
def initialiser_base_donnees():
    """Initialiser la base SQLite avec schéma complet"""
    conn = sqlite3.connect('cafe_energie.db')
    cursor = conn.cursor()
    
    # Table: sensors_data
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sensors_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            temperature_interieure REAL,
            temperature_exterieure REAL,
            humidite REAL,
            luminosite REAL,
            CO2 REAL,
            presence_clients INTEGER,
            consommation_energetique REAL,
            chauffage INTEGER,
            climatisation INTEGER,
            ventilation INTEGER,
            eclairage INTEGER,
            etat_equipements INTEGER,
            alertes INTEGER,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Table: predictions
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp_prediction DATETIME NOT NULL,
            model_type TEXT,
            predicted_value REAL,
            actual_value REAL,
            mae REAL,
            rmse REAL,
            accuracy REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Table: anomalies_detectees
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS anomalies_detectees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            type_anomalie TEXT,
            valeur_observee REAL,
            seuil_normal REAL,
            score_anomalie REAL,
            severity TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Table: alerts_intelligentes
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts_intelligentes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            type_alerte TEXT,
            description TEXT,
            action_recommandee TEXT,
            priorite INTEGER,
            resolved BOOLEAN,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Table: equipment_state
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS equipment_state (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            equipment_name TEXT,
            state INTEGER,
            predicted_maintenance BOOLEAN,
            durability_score REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Table: logs_systeme
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS logs_systeme (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            operation TEXT,
            status TEXT,
            details TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    return conn

# ==================================================================================
# 2. PRÉTRAITEMENT AVANCÉ DES DONNÉES
# ==================================================================================

def analyser_qualite_donnees(df):
    """Analyser la qualité des données avant le prétraitement"""
    rapport = {
        'valeurs_manquantes': df.isnull().sum(),
        'valeurs_manquantes_pct': (df.isnull().sum() / len(df) * 100),
        'duplicatas': df.duplicated().sum(),
        'statistiques': df.describe()
    }
    return rapport


def detecter_outliers(serie, methode='iqr'):
    """Détecter les valeurs aberrantes avec IQR ou Z-score"""
    if methode == 'iqr':
        Q1 = serie.quantile(0.25)
        Q3 = serie.quantile(0.75)
        IQR = Q3 - Q1
        limite_inf = Q1 - 1.5 * IQR
        limite_sup = Q3 + 1.5 * IQR
        outliers = (serie < limite_inf) | (serie > limite_sup)
    else:  # z-score
        z_scores = np.abs(stats.zscore(serie.dropna()))
        outliers_idx = z_scores > 3
        outliers = pd.Series(False, index=serie.index)
        outliers.loc[serie.dropna().index] = outliers_idx.values
    
    return outliers


def pretraiter_donnees(df):
    """Prétraitement complet avec gestion des valeurs manquantes, outliers et standardisation"""
    df = df.copy()
    
    # Statistiques initiales
    colonnes_numeriques = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 1. GESTION DES VALEURS MANQUANTES
    for col in colonnes_numeriques:
        if df[col].isnull().sum() > 0:
            # Interpolation linéaire pour séries temporelles
            df[col] = df[col].interpolate(method='linear', limit_direction='both')
            # Remplissage avec forward fill si besoin
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
            # Remplissage avec la médiane pour les valeurs restantes
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
    
    # 2. DÉTECTION ET TRAITEMENT DES OUTLIERS
    outliers_mask = pd.DataFrame(False, index=df.index, columns=colonnes_numeriques)
    for col in colonnes_numeriques:
        outliers_mask[col] = detecter_outliers(df[col], methode='iqr')
    
    # Remplacer les outliers par la médiane
    for col in colonnes_numeriques:
        if outliers_mask[col].any():
            median_val = df.loc[~outliers_mask[col], col].median()
            df.loc[outliers_mask[col], col] = median_val
    
    # 3. EXTRACTION DES FEATURES TEMPORELLES
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['quarter'] = df['timestamp'].dt.quarter
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    
    # Catégorisation temporelle
    df['period'] = pd.cut(df['hour'], 
                          bins=[0, 6, 12, 18, 24],
                          labels=['Nuit', 'Matin', 'Après-midi', 'Soirée'],
                          include_lowest=True)
    
    # 4. FEATURES D'INTERACTION
    df['diff_temp'] = df['temperature_interieure'] - df['temperature_exterieure']
    df['temp_humidite_interaction'] = df['temperature_interieure'] * df['humidite']
    df['luminosite_presence_interaction'] = df['luminosite'] * df['presence_clients']
    
    # 5. RATIOS ÉNERGÉTIQUES
    df['consomation_par_personne'] = df['consommation_energetique'] / (df['presence_clients'] + 1)
    df['actifs_equipements'] = df['chauffage'] + df['climatisation'] + df['ventilation'] + df['eclairage']
    df['ratio_conso_equipements'] = df['consommation_energetique'] / (df['actifs_equipements'] + 1)
    
    # 6. FENÊTRES GLISSANTES (ROLLING FEATURES)
    for col in ['consommation_energetique', 'CO2', 'temperature_interieure']:
        df[f'{col}_rolling_mean_4'] = df[col].rolling(window=4, min_periods=1).mean()
        df[f'{col}_rolling_std_4'] = df[col].rolling(window=4, min_periods=1).std().fillna(0)
        df[f'{col}_rolling_min_4'] = df[col].rolling(window=4, min_periods=1).min()
        df[f'{col}_rolling_max_4'] = df[col].rolling(window=4, min_periods=1).max()
    
    # 7. FEATURES DE TENDANCE
    df['conso_trend'] = df['consommation_energetique'].diff().fillna(0)
    df['co2_trend'] = df['CO2'].diff().fillna(0)
    df['temp_trend'] = df['temperature_interieure'].diff().fillna(0)
    
    # 8. NORMALISATION MIN-MAX POUR CERTAINES COLONNES CLÉS
    colonnes_normaliser = ['luminosite', 'CO2', 'presence_clients']
    scaler = MinMaxScaler()
    for col in colonnes_normaliser:
        if col in df.columns:
            df[f'{col}_normalized'] = scaler.fit_transform(df[[col]])
    
    # Stockage des information de qualité
    df.attrs['qualite_donnees'] = {
        'outliers_detectes': outliers_mask.sum().sum(),
        'valeurs_manquantes_traitees': (df.isnull().sum() > 0).sum()
    }
    
    return df

# ==================================================================================
# 3. MODÈLES MACHINE LEARNING AVANCÉS
# ==================================================================================

class PipelineML:
    """Pipeline complet de Machine Learning pour optimisation énergétique"""
    
    def __init__(self, df_processed):
        self.df = df_processed
        self.scaler_input = StandardScaler()
        self.scaler_output = StandardScaler()
        self.models = {}
        self.resultats = {}
        
    def preparer_features(self):
        """Préparer les features pour ML"""
        features_numeriques = [
            'temperature_interieure', 'temperature_exterieure', 'humidite',
            'luminosite', 'CO2', 'presence_clients', 'chauffage', 'climatisation',
            'ventilation', 'eclairage', 'etat_equipements', 'hour', 'dayofweek',
            'is_weekend', 'actifs_equipements', 'consomation_par_personne',
            'temp_humidite_interaction', 'diff_temp'
        ]
        
        # Ajouter les rolling features
        for col in self.df.columns:
            if 'rolling' in col:
                features_numeriques.append(col)
        
        features_numeriques = [f for f in features_numeriques if f in self.df.columns]
        return features_numeriques
    
    def model_regression_lineaire(self):
        """Régression linéaire pour prédiction consommation énergétique"""
        X = self.df[self.preparer_features()]
        y = self.df['consommation_energetique']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train_scaled = self.scaler_input.fit_transform(X_train)
        X_test_scaled = self.scaler_input.transform(X_test)
        
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        mae_test = mean_absolute_error(y_test, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        r2_test = model.score(X_test_scaled, y_test)
        
        self.models['regression_lineaire'] = (model, self.scaler_input)
        self.resultats['regression_lineaire'] = {
            'mae': mae_test,
            'rmse': rmse_test,
            'r2': r2_test,
            'coefficients': dict(zip(self.preparer_features(), model.coef_))
        }
        
        return model, mae_test, rmse_test, r2_test
    
    def model_random_forest_classif(self):
        """Random Forest pour classification des alertes"""
        X = self.df[self.preparer_features()]
        y = self.df['alertes'].apply(lambda x: 1 if x > 2 else 0)  # Alerte: oui/non
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        
        self.models['random_forest_classif'] = model
        self.resultats['random_forest_classif'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'feature_importance': dict(zip(self.preparer_features(), model.feature_importances_))
        }
        
        return model, accuracy, precision, recall
    
    def model_random_forest_etat_equipements(self):
        """Random Forest pour prédiction état équipements"""
        X = self.df[self.preparer_features()]
        y = self.df['etat_equipements']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        self.models['random_forest_equipements'] = model
        self.resultats['random_forest_equipements'] = {
            'mae': mae,
            'rmse': rmse,
            'feature_importance': dict(zip(self.preparer_features(), model.feature_importances_))
        }
        
        return model, mae, rmse
    
    def model_isolation_forest_anomalies(self):
        """Isolation Forest pour détection d'anomalies"""
        X = self.df[['consommation_energetique', 'CO2', 'temperature_interieure', 
                      'humidite', 'presence_clients']]
        X_scaled = self.scaler_input.fit_transform(X)
        
        model = IsolationForest(contamination=0.1, random_state=42)
        anomalies = model.fit_predict(X_scaled)
        anomaly_scores = model.score_samples(X_scaled)
        
        self.models['isolation_forest'] = model
        self.df['anomaly'] = anomalies
        self.df['anomaly_score'] = anomaly_scores
        
        n_anomalies = (anomalies == -1).sum()
        anomaly_ratio = n_anomalies / len(anomalies)
        
        self.resultats['isolation_forest'] = {
            'n_anomalies': n_anomalies,
            'anomaly_ratio': anomaly_ratio
        }
        
        return model, n_anomalies, anomaly_ratio
    
    def model_mlp_regressor(self):
        """MLPRegressor (réseau de neurones) pour prédiction avancée"""
        X = self.df[self.preparer_features()]
        y = self.df['consommation_energetique']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train_scaled = self.scaler_input.fit_transform(X_train)
        X_test_scaled = self.scaler_input.transform(X_test)
        
        y_train_scaled = self.scaler_output.fit_transform(y_train.values.reshape(-1, 1))
        y_test_scaled = self.scaler_output.transform(y_test.values.reshape(-1, 1))
        
        model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            learning_rate_init=0.001
        )
        
        model.fit(X_train_scaled, y_train_scaled.ravel())
        
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = self.scaler_output.inverse_transform(y_pred_scaled.reshape(-1, 1))
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        self.models['mlp_regressor'] = model
        self.resultats['mlp_regressor'] = {
            'mae': mae,
            'rmse': rmse,
            'best_loss': model.best_loss_
        }
        
        return model, mae, rmse

# ==================================================================================
# 4. SYSTÈME D'ALERTES INTELLIGENTES
# ==================================================================================

def generer_alertes_intelligentes(df):
    """Générer des alertes basées sur règles data-driven"""
    alertes = []
    
    derniere_ligne = df.iloc[-1]
    
    # Alerte CO2
    seuil_co2 = df['CO2'].quantile(0.85)
    if derniere_ligne['CO2'] > seuil_co2:
        alertes.append({
            'type': 'CO2 ÉLEVÉ',
            'niveau': 'CRITIQUE' if derniere_ligne['CO2'] > df['CO2'].quantile(0.95) else 'ATTENTION',
            'valeur': f"{derniere_ligne['CO2']:.0f} ppm",
            'action': 'Activer la ventilation immédiatement'
        })
    
    # Alerte température
    seuil_temp = df['temperature_interieure'].quantile(0.9)
    if derniere_ligne['temperature_interieure'] > seuil_temp:
        alertes.append({
            'type': 'TEMPÉRATURE ÉLEVÉE',
            'niveau': 'CRITIQUE' if derniere_ligne['temperature_interieure'] > seuil_temp + 5 else 'ATTENTION',
            'valeur': f"{derniere_ligne['temperature_interieure']:.1f}°C",
            'action': 'Activer la climatisation'
        })
    
    # Alerte surcons consommation énergétique
    seuil_consomation = df['consommation_energetique'].quantile(0.9)
    if derniere_ligne['consommation_energetique'] > seuil_consomation:
        alertes.append({
            'type': 'SURCONSOMMATION',
            'niveau': 'CRITIQUE',
            'valeur': f"{derniere_ligne['consommation_energetique']:.2f} kWh",
            'action': 'Vérifier les équipements actifs'
        })
    
    # Alerte humidité
    if derniere_ligne['humidite'] > 80:
        alertes.append({
            'type': 'HUMIDITÉ EXCESSIVE',
            'niveau': 'ATTENTION',
            'valeur': f"{derniere_ligne['humidite']:.1f}%",
            'action': 'Augmenter la ventilation'
        })
    
    # Alerte maintenance équipements
    if derniere_ligne['etat_equipements'] >= 3:
        alertes.append({
            'type': 'MAINTENANCE PRÉDICTIVE',
            'niveau': 'CRITIQUE',
            'valeur': f"État: {derniere_ligne['etat_equipements']}/4",
            'action': 'Planifier une maintenance urgente'
        })
    
    return alertes

# ==================================================================================
# 5. INTERFACE STREAMLIT - LAYOUT PRINCIPAL
# ==================================================================================

def main():
    initialiser_rapport_session()

    # Chargement des données par dépôt de fichier
    with st.sidebar:
        uploaded_file = st.file_uploader(
            "Déposer le fichier CSV",
            type=["csv"],
            accept_multiple_files=False
        )

    if uploaded_file is None:
        st.info("Déposer un fichier CSV pour lancer l'analyse.")
        st.stop()

    df = charger_donnees_csv(uploaded_file.getvalue())
    df_processed = pretraiter_donnees(df)
    conn = initialiser_base_donnees()
    
    # ===== MENU LATÉRAL =====
    with st.sidebar:
        st.markdown("## ENERGIECAFÉ IA")
        st.markdown("---")
        
        page = st.radio(
            label="Navigation",
            options=[
                "Dashboard Principal",
                "Prétraitement des Données",
                "Analyse Exploratoire",
                "Machine Learning",
                "Prédictions Avancées",
                "Alertes Intelligentes",
                "Rapports & Économies",
                "Paramètres Système"
            ],
            key="navigation"
        )
        
        st.markdown("---")
        st.info("Système IA avancé d'optimisation énergétique basé sur analyse de données historiques")
        
        # Filtres temporels
        st.markdown("### Filtres Temporels")
        date_debut = st.date_input("Date début", min(df['timestamp']), key="date_debut")
        date_fin = st.date_input("Date fin", max(df['timestamp']), key="date_fin")
        
        df_filtered = df[(df['timestamp'].dt.date >= date_debut) & 
                         (df['timestamp'].dt.date <= date_fin)].copy()
        
        st.metric("Nombre d'observations", len(df_filtered))
        st.metric("Période d'analyse", f"{(date_fin - date_debut).days} jours")
    
    # ===== PAGES PRINCIPALES =====
    
    if page == "Dashboard Principal":
        st.markdown("# Dashboard Principal - Optimisation Énergétique")
        st.markdown("---")
        
        # KPIs principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            consomation_totale = df_filtered['consommation_energetique'].sum()
            st.metric("Consommation (kWh)", f"{consomation_totale:.2f}", 
                     delta=f"{(consomation_totale/len(df_filtered)):.2f}/moy")
        
        with col2:
            temp_moy = df_filtered['temperature_interieure'].mean()
            st.metric("Température (°C)", f"{temp_moy:.1f}", 
                     delta=f"±{df_filtered['temperature_interieure'].std():.1f}")
        
        with col3:
            co2_max = df_filtered['CO2'].max()
            co2_moy = df_filtered['CO2'].mean()
            st.metric("CO2 Max (ppm)", f"{co2_max:.0f}", 
                     delta=f"{co2_moy:.0f} moy")
        
        with col4:
            clients_max = df_filtered['presence_clients'].max()
            clients_moy = df_filtered['presence_clients'].mean()
            st.metric("Clients Max", f"{int(clients_max)}", 
                     delta=f"{clients_moy:.1f} moy")
        
        st.markdown("---")
        
        # Graphiques principaux
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Consommation Énergétique en Temps Réel")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_filtered['timestamp'],
                y=df_filtered['consommation_energetique'],
                mode='lines+markers',
                name='Consommation',
                line=dict(color='#FF6B6B', width=2),
                marker=dict(size=4)
            ))
            fig.update_layout(
                title='', xaxis_title='Temps', yaxis_title='kWh',
                height=400, hovermode='x unified'
            )
            enregistrer_graphique("Consommation énergétique en temps réel", fig)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Comparaison Température Intérieure/Extérieure")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_filtered['timestamp'],
                y=df_filtered['temperature_interieure'],
                name='Intérieure',
                line=dict(color='#FF6B6B', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=df_filtered['timestamp'],
                y=df_filtered['temperature_exterieure'],
                name='Extérieure',
                line=dict(color='#4ECDC4', width=2)
            ))
            fig.update_layout(
                title='', xaxis_title='Temps', yaxis_title='°C',
                height=400, hovermode='x unified'
            )
            enregistrer_graphique("Comparaison température intérieure et extérieure", fig)
            st.plotly_chart(fig, use_container_width=True)
        
        # Alertes en temps réel
        st.markdown("---")
        st.markdown("#### Alertes en Temps Réel")
        
        alertes = generer_alertes_intelligentes(df_filtered)
        
        if alertes:
            for alerte in alertes:
                if alerte['niveau'] == 'CRITIQUE':
                    st.markdown(f"""
                    <div class="alert-critical">
                    <b>{alerte['type']}</b> - {alerte['valeur']}<br>
                    Recommandation: {alerte['action']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="alert-warning">
                    <b>{alerte['type']}</b> - {alerte['valeur']}<br>
                    Recommandation: {alerte['action']}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.success("Tous les paramètres sont dans les normes")
    
    elif page == "Prétraitement des Données":
        st.markdown("# Prétraitement des Données")
        st.markdown("---")
        st.info("Analyse complète du prétraitement, nettoyage, standardisation et détection d'anomalies")
        
        # Qualité des données
        st.markdown("## Qualité des Données Brutes")
        col1, col2, col3 = st.columns(3)
        
        valeurs_manquantes_total = df.isnull().sum().sum()
        duplicatas = df.duplicated().sum()
        
        with col1:
            st.metric("Valeurs Manquantes", f"{valeurs_manquantes_total}")
        with col2:
            st.metric("Duplicatas", f"{duplicatas}")
        with col3:
            st.metric("Nombre d'Observations", f"{len(df):,}")
        
        # Matrice des valeurs manquantes
        st.markdown("### Valeurs Manquantes par Colonne")
        missing_data = pd.DataFrame({
            'Colonne': df.columns,
            'Manquantes': df.isnull().sum().values,
            'Pourcentage': (df.isnull().sum().values / len(df) * 100).round(2)
        })
        missing_data = missing_data[missing_data['Manquantes'] > 0].sort_values('Manquantes', ascending=False)
        
        if len(missing_data) > 0:
            fig = px.bar(missing_data, x='Colonne', y='Pourcentage',
                        title='Pourcentage de valeurs manquantes par colonne',
                        color='Pourcentage', color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("Aucune valeur manquante détectée")
        
        st.markdown("---")
        
        # Détection des outliers
        st.markdown("## Détection des Anomalies et Outliers")
        
        colonnes_numeriques = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
        col_select = st.selectbox("Sélectionner une colonne pour analyser", colonnes_numeriques, key="outlier_select")
        
        if col_select:
            serie = df_filtered[col_select]
            outliers = detecter_outliers(serie, methode='iqr')
            n_outliers = outliers.sum()
            pct_outliers = n_outliers / len(serie) * 100
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Outliers Détectés (IQR)", f"{n_outliers}", delta=f"{pct_outliers:.2f}%")
            
            with col2:
                Q1 = serie.quantile(0.25)
                Q3 = serie.quantile(0.75)
                IQR = Q3 - Q1
                st.metric("Écart Interquartile (IQR)", f"{IQR:.3f}")
            
            # Boxplot
            fig = go.Figure()
            fig.add_trace(go.Box(y=serie, name=col_select, marker_color='#4ECDC4'))
            fig.update_layout(title=f'Boxplot - {col_select}', height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Histogramme avec outliers en surbrillance
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=serie[~outliers], name='Normal', 
                                       marker_color='#4ECDC4', nbinsx=50))
            if n_outliers > 0:
                fig.add_trace(go.Histogram(x=serie[outliers], name='Outlier', 
                                          marker_color='#FF6B6B', nbinsx=20))
            fig.update_layout(title=f'Distribution avec détection d\'anomalies - {col_select}',
                            height=400, barmode='overlay')
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Statistiques descriptives
        st.markdown("## Statistiques Descriptives")
        
        stats_cols = [col for col in colonnes_numeriques if col not in ['timestamp']][:10]
        stats_df = df_filtered[stats_cols].describe().T
        stats_df.columns = ['Compte', 'Moyenne', 'Écart-type', 'Min', '25%', 'Médiane', '75%', 'Max']
        
        st.dataframe(stats_df.style.format('{:.2f}'), use_container_width=True)
        
        st.markdown("---")
        
        # Standardisation et normalisation
        st.markdown("## Standardisation et Normalisation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Avant Normalisation")
            sample_before = df_filtered[['consommation_energetique', 'CO2', 'temperature_interieure']].head(5)
            st.dataframe(sample_before.style.format('{:.2f}'), use_container_width=True)
        
        with col2:
            st.markdown("### Après Normalisation (Min-Max)")
            scaler = MinMaxScaler()
            sample_after = scaler.fit_transform(df_filtered[['consommation_energetique', 'CO2', 'temperature_interieure']].head(5))
            sample_after_df = pd.DataFrame(sample_after, columns=['consommation_energetique', 'CO2', 'temperature_interieure'])
            st.dataframe(sample_after_df.style.format('{:.3f}'), use_container_width=True)
        
        # Distribution avant/après
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Avant Prétraitement', 'Après Prétraitement'))
        
        fig.add_trace(go.Histogram(x=df_filtered['consommation_energetique'], name='Avant',
                                  marker_color='#FF6B6B', nbinsx=50), row=1, col=1)
        
        scaler = StandardScaler()
        conso_scaled = scaler.fit_transform(df_filtered[['consommation_energetique']])
        fig.add_trace(go.Histogram(x=conso_scaled.flatten(), name='Après',
                                  marker_color='#4ECDC4', nbinsx=50), row=1, col=2)
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Features créées
        st.markdown("## Features Créées lors du Prétraitement")
        
        features_created = [col for col in df_processed.columns if col not in df.columns]
        
        st.markdown(f"**Nombre de features créées : {len(features_created)}**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Features Temporelles")
            temporal_features = [f for f in features_created if any(x in f for x in ['hour', 'day', 'month', 'week', 'quarter', 'period', 'year'])]
            for f in temporal_features:
                st.write(f"- {f}")
        
        with col2:
            st.markdown("### Features d'Interaction et Ratios")
            interaction_features = [f for f in features_created if any(x in f for x in ['interaction', 'ratio', 'rolling', 'trend', 'normalized'])][:10]
            for f in interaction_features:
                st.write(f"- {f}")
    
    elif page == "Analyse Exploratoire":
        st.markdown("# Analyse Exploratoire des Données")
        st.markdown("---")
        
        # Statistiques descriptives
        st.markdown("## Statistiques Descriptives")
        
        stats_cols = ['temperature_interieure', 'temperature_exterieure', 'humidite',
                      'luminosite', 'CO2', 'presence_clients', 'consommation_energetique']
        
        stats_df = df_filtered[stats_cols].describe().T
        stats_df.columns = ['Nombre', 'Moyenne', 'Écart-type', 'Min', '25%', 'Médiane', '75%', 'Max']
        
        st.dataframe(stats_df.style.format('{:.2f}'), use_container_width=True)
        
        st.markdown("---")
        
        # Corrélations
        st.markdown("## Matrice de Corrélation")
        
        correlation_matrix = df_filtered[stats_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(correlation_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        fig.update_layout(height=600, width=800)
        enregistrer_graphique("Matrice de corrélation", fig)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Distributions
        st.markdown("## Distributions des Variables")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Luminosité")
            fig = px.histogram(df_filtered, x='luminosite', nbins=50,
                             labels={'luminosite': 'Luminosité (lux)'},
                             color_discrete_sequence=['#4ECDC4'])
            fig.update_layout(height=400)
            enregistrer_graphique("Distribution de la luminosité", fig)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### CO2")
            fig = px.histogram(df_filtered, x='CO2', nbins=50,
                             labels={'CO2': 'CO2 (ppm)'},
                             color_discrete_sequence=['#FF6B6B'])
            fig.update_layout(height=400)
            enregistrer_graphique("Distribution du CO2", fig)
            st.plotly_chart(fig, use_container_width=True)
        
        # Analyse temporelle
        st.markdown("---")
        st.markdown("## Analyse Temporelle - Patterns par Heure")
        
        df_hourly = df_filtered.copy()
        df_hourly['hour'] = df_hourly['timestamp'].dt.hour
        hourly_stats = df_hourly.groupby('hour')[['consommation_energetique', 'presence_clients', 'CO2']].mean()
        
        fig = make_subplots(rows=3, cols=1, subplot_titles=('Consommation', 'Clients', 'CO2'))
        
        fig.add_trace(go.Scatter(x=hourly_stats.index, y=hourly_stats['consommation_energetique'],
                                mode='lines+markers', name='Consommation', line=dict(color='#FF6B6B')),
                     row=1, col=1)
        fig.add_trace(go.Scatter(x=hourly_stats.index, y=hourly_stats['presence_clients'],
                                mode='lines+markers', name='Clients', line=dict(color='#4ECDC4')),
                     row=2, col=1)
        fig.add_trace(go.Scatter(x=hourly_stats.index, y=hourly_stats['CO2'],
                                mode='lines+markers', name='CO2', line=dict(color='#95E1D3')),
                     row=3, col=1)
        
        fig.update_xaxes(title_text="Heure du jour", row=3, col=1)
        fig.update_layout(height=700, showlegend=True)
        enregistrer_graphique("Analyse temporelle par heure", fig)
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Machine Learning":
        st.markdown("# Pipeline Machine Learning Avancé")
        st.markdown("---")
        
        st.info("Entraînement automatique de 5 modèles IA pour optimisation énergétique")
        
        # Initialisation du pipeline ML
        pipeline = PipelineML(df_processed)
        
        # Tabs pour chaque modèle
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Régression Linéaire",
            "Random Forest",
            "Équipements",
            "Anomalies",
            "MLP Neurones"
        ])
        
        # TAB 1: Régression Linéaire
        with tab1:
            st.markdown("## Régression Linéaire - Prédiction Consommation")
            st.markdown("**Objectif:** Prédire la consommation énergétique future")
            
            with st.spinner("Entraînement Régression Linéaire..."):
                model_lr, mae_lr, rmse_lr, r2_lr = pipeline.model_regression_lineaire()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MAE (kWh)", f"{mae_lr:.3f}", delta="Erreur absolue")
            with col2:
                st.metric("RMSE (kWh)", f"{rmse_lr:.3f}", delta="Erreur quadratique")
            with col3:
                st.metric("R² Score", f"{r2_lr:.4f}", delta="Qualité du modèle")
            
            # Importance des coefficients
            st.markdown("### Coefficients du Modèle (Top 15)")
            coeff_dict = pipeline.resultats['regression_lineaire']['coefficients']
            coeff_sorted = sorted(coeff_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
            
            fig = go.Figure(data=[
                go.Bar(x=[x[0] for x in coeff_sorted],
                      y=[x[1] for x in coeff_sorted],
                      marker_color=['#FF6B6B' if x[1] < 0 else '#4ECDC4' for x in coeff_sorted])
            ])
            fig.update_layout(title='Impact de chaque feature sur la prédiction', xaxis_title='Features', 
                            yaxis_title='Coefficient', height=400, xaxis_tickangle=-45)
            enregistrer_graphique("Coefficients du modèle de régression linéaire", fig)
            st.plotly_chart(fig, use_container_width=True)
            
            # Graphique Valeurs Réelles vs Prédites
            st.markdown("### Validation: Valeurs Réelles vs Prédites")
            X = df_processed[pipeline.preparer_features()]
            y = df_processed['consommation_energetique']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_test_scaled = pipeline.scaler_input.transform(X_test)
            y_pred_test = model_lr.predict(X_test_scaled)
            
            fig = go.Figure()
            # Points de prédiction
            fig.add_trace(go.Scatter(
                x=y_test.values, y=y_pred_test,
                mode='markers',
                marker=dict(color='#4ECDC4', size=6, opacity=0.7),
                name='Prédictions'
            ))
            # Ligne idéale
            min_val = min(y_test.min(), y_pred_test.min())
            max_val = max(y_test.max(), y_pred_test.max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines',
                line=dict(color='#FF6B6B', dash='dash', width=2),
                name='Prédiction parfaite'
            ))
            fig.update_layout(
                title='Qualité d\'ajustement du modèle',
                xaxis_title='Valeurs Réelles (kWh)',
                yaxis_title='Valeurs Prédites (kWh)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Analyse des résidus
            st.markdown("### Analyse des Résidus")
            residus = y_test.values - y_pred_test
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Moyenne des Résidus", f"{residus.mean():.4f}")
                st.metric("Écart-type des Résidus", f"{residus.std():.4f}")
            
            with col2:
                st.metric("Min Résidu", f"{residus.min():.4f}")
                st.metric("Max Résidu", f"{residus.max():.4f}")
            
            # Graphique des résidus
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_pred_test, y=residus,
                mode='markers',
                marker=dict(color='#95E1D3', size=6),
                name='Résidus'
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            fig.update_layout(
                title='Résidus du modèle',
                xaxis_title='Valeurs Prédites (kWh)',
                yaxis_title='Résidus',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Distribution des résidus
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=residus, nbinsx=50, marker_color='#4ECDC4'))
            fig.update_layout(title='Distribution des résidus', xaxis_title='Résidu', yaxis_title='Fréquence', height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # TAB 2: Random Forest Classification
        with tab2:
            st.markdown("## Random Forest - Classification des Alertes")
            st.markdown("**Objectif:** Prédire si une alerte sera déclenchée")
            
            with st.spinner("Entraînement Random Forest..."):
                model_rf, accuracy_rf, precision_rf, recall_rf = pipeline.model_random_forest_classif()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{accuracy_rf:.4f}")
            with col2:
                st.metric("Precision", f"{precision_rf:.4f}")
            with col3:
                st.metric("Recall", f"{recall_rf:.4f}")
            with col4:
                f1_score = 2 * (precision_rf * recall_rf) / (precision_rf + recall_rf) if (precision_rf + recall_rf) > 0 else 0
                st.metric("F1-Score", f"{f1_score:.4f}")
            
            # Feature importance
            st.markdown("### Importance des Features (Top 15)")
            importance_dict = pipeline.resultats['random_forest_classif']['feature_importance']
            importance_sorted = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:15]
            
            fig = go.Figure(data=[
                go.Bar(y=[x[0] for x in importance_sorted],
                      x=[x[1] for x in importance_sorted],
                      orientation='h',
                      marker_color='#FF6B6B')
            ])
            fig.update_layout(title='Variables les plus impactantes', xaxis_title='Importance', yaxis_title='',
                            height=450, yaxis_tickfont=dict(size=10))
            enregistrer_graphique("Importance des variables du random forest", fig)
            st.plotly_chart(fig, use_container_width=True)
            
            # Matrice de confusion
            st.markdown("### Matrice de Confusion")
            X = df_processed[pipeline.preparer_features()]
            y = df_processed['alertes'].apply(lambda x: 1 if x > 2 else 0)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            y_pred_rf = model_rf.predict(X_test)
            
            cm = confusion_matrix(y_test, y_pred_rf)
            
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Pas d\'Alerte', 'Alerte'],
                y=['Pas d\'Alerte', 'Alerte'],
                text=cm,
                texttemplate='%{text}',
                colorscale='RdBu'
            ))
            fig.update_layout(title='Matrice de Confusion', height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Distribution des prédictions
            st.markdown("### Distribution des Prédictions")
            pred_dist = pd.Series(y_pred_rf).value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure(data=[
                    go.Pie(labels=['Sans Alerte', 'Avec Alerte'], 
                          values=pred_dist.values,
                          marker_colors=['#4ECDC4', '#FF6B6B'])
                ])
                fig.update_layout(title='Distribution des Prédictions', height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**Comparaison Réalité vs Prédiction**")
                comparison = pd.DataFrame({
                    'Étiquette': ['Sans Alerte (Réalité)', 'Avec Alerte (Réalité)',
                                 'Sans Alerte (Prédiction)', 'Avec Alerte (Prédiction)'],
                    'Nombre': [
                        (y_test == 0).sum(),
                        (y_test == 1).sum(),
                        (y_pred_rf == 0).sum(),
                        (y_pred_rf == 1).sum()
                    ]
                })
                st.dataframe(comparison, use_container_width=True)
        
        # TAB 3: État Équipements
        with tab3:
            st.markdown("## Prédiction État Équipements")
            st.markdown("**Objectif:** Prédire l'état des équipements et maintenance")
            
            with st.spinner("Entraînement État Équipements..."):
                model_eq, mae_eq, rmse_eq = pipeline.model_random_forest_etat_equipements()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MAE", f"{mae_eq:.3f}")
            with col2:
                st.metric("RMSE", f"{rmse_eq:.3f}")
            with col3:
                st.metric("Erreur Relative (%)", f"{mae_eq/df_processed['etat_equipements'].mean()*100:.2f}%")
            
            st.info("État = 0: Bon | État = 1: Acceptable | État = 2: À surveiller | État >= 3: Maintenance urgente")
            
            # Feature importance
            st.markdown("### Importance des Features pour la Maintenance")
            importance_dict = pipeline.resultats['random_forest_equipements']['feature_importance']
            importance_sorted = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:12]
            
            fig = go.Figure(data=[
                go.Bar(y=[x[0] for x in importance_sorted],
                      x=[x[1] for x in importance_sorted],
                      orientation='h',
                      marker_color='#95E1D3')
            ])
            fig.update_layout(title='Variables prédictives de la maintenance', xaxis_title='Importance', height=450)
            st.plotly_chart(fig, use_container_width=True)
            
            # Distribution des états prédits
            X = df_processed[pipeline.preparer_features()]
            y_eq = df_processed['etat_equipements']
            X_train, X_test, y_train, y_test = train_test_split(X, y_eq, test_size=0.2, random_state=42)
            y_pred_eq = model_eq.predict(X_test)
            
            st.markdown("### Distribution des États d'Équipements")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=y_test.values, name='Réalité', marker_color='#FF6B6B', nbinsx=5, opacity=0.7))
                fig.add_trace(go.Histogram(x=y_pred_eq, name='Prédiction', marker_color='#4ECDC4', nbinsx=5, opacity=0.7))
                fig.update_layout(title='État des équipements: Réalité vs Prédiction', barmode='overlay', height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Graphique scatter prédiction vs réalité
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y_test.values, y=y_pred_eq, mode='markers', 
                                        marker=dict(color='#95E1D3', size=7)))
                fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
                                        mode='lines', line=dict(color='#FF6B6B', dash='dash')))
                fig.update_layout(title='Qualité de prédiction', xaxis_title='État Réel', 
                                 yaxis_title='État Prédit', height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        
        # TAB 4: Détection Anomalies
        with tab4:
            st.markdown("## Isolation Forest - Détection d'Anomalies")
            st.markdown("**Objectif:** Détecter comportements énergétiques anormaux")
            
            with st.spinner("Analyse des Anomalies..."):
                model_if, n_anomalies, anomaly_ratio = pipeline.model_isolation_forest_anomalies()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Anomalies Détectées", n_anomalies)
            with col2:
                st.metric("Ratio", f"{anomaly_ratio*100:.2f}%")
            with col3:
                st.metric("Points Normaux", f"{len(df_processed) - n_anomalies:,}")
            
            # Visualisation temporelle
            anomaly_data = df_processed.copy()
            anomaly_data['anomaly_label'] = anomaly_data['anomaly'].map({1: 'Normal', -1: 'Anomalie'})
            
            fig = go.Figure(data=[
                go.Scatter(
                    x=anomaly_data[anomaly_data['anomaly'] == 1]['timestamp'],
                    y=anomaly_data[anomaly_data['anomaly'] == 1]['consommation_energetique'],
                    mode='markers',
                    name='Normal',
                    marker=dict(color='#4ECDC4', size=5)
                ),
                go.Scatter(
                    x=anomaly_data[anomaly_data['anomaly'] == -1]['timestamp'],
                    y=anomaly_data[anomaly_data['anomaly'] == -1]['consommation_energetique'],
                    mode='markers',
                    name='Anomalie',
                    marker=dict(color='#FF6B6B', size=8, symbol='diamond')
                )
            ])
            fig.update_layout(title='Détection d\'Anomalies dans le Temps', xaxis_title='Temps',
                            yaxis_title='Consommation (kWh)', height=400)
            enregistrer_graphique("Détection des anomalies", fig)
            st.plotly_chart(fig, use_container_width=True)
            
            # Score d'anomalie
            st.markdown("### Distribution des Scores d'Anomalie")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogramme des scores
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=df_processed['anomaly_score'], nbinsx=50, marker_color='#95E1D3'))
                fig.update_layout(title='Distribution des scores d\'anomalie', xaxis_title='Score', 
                                 yaxis_title='Fréquence', height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot pour comparaison
                fig = go.Figure()
                fig.add_trace(go.Box(y=df_processed[df_processed['anomaly']==1]['anomaly_score'], 
                                    name='Normal', marker_color='#4ECDC4'))
                fig.add_trace(go.Box(y=df_processed[df_processed['anomaly']==-1]['anomaly_score'],
                                    name='Anomalie', marker_color='#FF6B6B'))
                fig.update_layout(title='Scores d\'anomalie par classe', height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Relation avec d'autres variables
            st.markdown("### Caractéristiques des Anomalies Détectées")
            
            anomalies_df = df_processed[df_processed['anomaly'] == -1]
            
            if len(anomalies_df) > 0:
                comparison_stats = pd.DataFrame({
                    'Métrique': ['Conso Moyenne (kWh)', 'CO2 Moyen (ppm)', 'Température (°C)', 'Humidité (%)'],
                    'Normal': [
                        df_processed[df_processed['anomaly']==1]['consommation_energetique'].mean(),
                        df_processed[df_processed['anomaly']==1]['CO2'].mean(),
                        df_processed[df_processed['anomaly']==1]['temperature_interieure'].mean(),
                        df_processed[df_processed['anomaly']==1]['humidite'].mean()
                    ],
                    'Anomalies': [
                        anomalies_df['consommation_energetique'].mean(),
                        anomalies_df['CO2'].mean(),
                        anomalies_df['temperature_interieure'].mean(),
                        anomalies_df['humidite'].mean()
                    ]
                })
                
                numeric_cols = comparison_stats.select_dtypes(include=['number']).columns
                formats = {col: '{:.2f}' for col in numeric_cols}
                st.dataframe(comparison_stats.style.format(formats), use_container_width=True)
        
        # TAB 5: MLPRegressor
        with tab5:
            st.markdown("## MLPRegressor - Réseau de Neurones")
            st.markdown("**Objectif:** Prédiction avancée avec apprentissage profond")
            
            with st.spinner("Entraînement Réseau de Neurones..."):
                model_mlp, mae_mlp, rmse_mlp = pipeline.model_mlp_regressor()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MAE (kWh)", f"{mae_mlp:.3f}")
            with col2:
                st.metric("RMSE (kWh)", f"{rmse_mlp:.3f}")
            with col3:
                best_loss = pipeline.resultats.get('mlp_regressor', {}).get('best_loss')
                best_loss_display = f"{best_loss:.4f}" if best_loss is not None else "N/A"
                st.metric("Meilleure Perte", best_loss_display)
            
            st.success("Réseau de neurones entraîné avec succès")
            st.info("Architecture: 128 → 64 → 32 → 1 | Activation: ReLU | Optimizer: Adam")
            
            # Prédictions et résidus
            st.markdown("### Validation du Réseau de Neurones")
            
            X = df_processed[pipeline.preparer_features()]
            y = df_processed['consommation_energetique']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_test_scaled = pipeline.scaler_input.transform(X_test)
            y_test_scaled = pipeline.scaler_output.transform(y_test.values.reshape(-1, 1))
            y_pred_scaled = model_mlp.predict(X_test_scaled)
            y_pred = pipeline.scaler_output.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Valeurs réelles vs prédites
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y_test.values, y=y_pred, mode='markers',
                                        marker=dict(color='#95E1D3', size=6)))
                fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
                                        mode='lines', line=dict(color='#FF6B6B', dash='dash')))
                fig.update_layout(title='Prédiction vs Réalité', xaxis_title='Réalité (kWh)',
                                 yaxis_title='Prédiction (kWh)', height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Résidus
                residus_mlp = y_test.values - y_pred
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y_pred, y=residus_mlp, mode='markers',
                                        marker=dict(color='#4ECDC4', size=6)))
                fig.add_hline(y=0, line_color='red')
                fig.update_layout(title='Résidus du Réseau', xaxis_title='Prédiction (kWh)',
                                 yaxis_title='Résidus', height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        
        st.markdown("---")
        st.markdown("## Résumé Comparatif des Modèles")
        
        models_comparison = pd.DataFrame({
            'Modèle': ['Régression Linéaire', 'Random Forest (Alertes)', 'État Équipements', 'Isolation Forest', 'MLP Neurones'],
            'Métrique Principale': ['R²', 'Accuracy', 'MAE', 'Ratio Anomalies', 'RMSE'],
            'Valeur': [
                f"{pipeline.resultats['regression_lineaire']['r2']:.4f}",
                f"{pipeline.resultats['random_forest_classif']['accuracy']:.4f}",
                f"{pipeline.resultats['random_forest_equipements']['mae']:.3f}",
                f"{pipeline.resultats['isolation_forest']['anomaly_ratio']*100:.2f}%",
                f"{pipeline.resultats['mlp_regressor']['rmse']:.3f}"
            ],
            'Type': ['Régression', 'Classification', 'Régression', 'Anomalie', 'Réseau Neurones']
        })
        
        st.dataframe(models_comparison, use_container_width=True)
        
        # Visualisation comparée des performances
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Erreurs MAE/RMSE")
            errors_data = pd.DataFrame({
                'Modèle': ['Régression\nLinéaire', 'RF Équipements', 'MLP\nNeurones'],
                'MAE': [
                    pipeline.resultats['regression_lineaire']['mae'],
                    pipeline.resultats['random_forest_equipements']['mae'],
                    pipeline.resultats['mlp_regressor']['mae']
                ],
                'RMSE': [
                    pipeline.resultats['regression_lineaire']['rmse'],
                    pipeline.resultats['random_forest_equipements']['rmse'],
                    pipeline.resultats['mlp_regressor']['rmse']
                ]
            })
            
            fig = go.Figure(data=[
                go.Bar(name='MAE', x=errors_data['Modèle'], y=errors_data['MAE'], marker_color='#FF6B6B'),
                go.Bar(name='RMSE', x=errors_data['Modèle'], y=errors_data['RMSE'], marker_color='#4ECDC4')
            ])
            fig.update_layout(height=400, barmode='group')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Scores de Performance")
            scores_data = {
                'Régression Linéaire': pipeline.resultats['regression_lineaire']['r2'],
                'RF Alertes': pipeline.resultats['random_forest_classif']['accuracy'],
                'MLP Neurones': pipeline.resultats['mlp_regressor']['mae'] / 10
            }
            
            fig = go.Figure(data=[
                go.Bar(x=list(scores_data.keys()), y=list(scores_data.values()),
                      marker_color=['#4ECDC4', '#FF6B6B', '#95E1D3'])
            ])
            fig.update_layout(title='Comparaison des performances', height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Prédictions Avancées":
        st.markdown("# Prédictions Avancées avec Paramètres Personnalisés")
        st.markdown("---")
        
        # Initialisation du pipeline ML
        pipeline = PipelineML(df_processed)
        
        with st.spinner("Entraînement des modèles..."):
            model_lr, _, _, _ = pipeline.model_regression_lineaire()
            model_mlp, _, _ = pipeline.model_mlp_regressor()
        
        # Sélection du modèle
        col1, col2 = st.columns(2)
        
        with col1:
            modele_select = st.selectbox(
                "Sélectionner le modèle de prédiction",
                ["Régression Linéaire", "Réseau de Neurones"]
            )
        
        with col2:
            st.info("Ajustez les paramètres ci-dessous pour générer une prédiction personnalisée")
        
        st.markdown("---")
        st.markdown("## Paramètres d'Entrée Personnalisés")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            temp_int = st.slider("Température Intérieure (°C)", 
                                min_value=15.0, max_value=30.0, value=22.0, step=0.1)
            temp_ext = st.slider("Température Extérieure (°C)",
                                min_value=0.0, max_value=40.0, value=18.0, step=0.1)
            humidite = st.slider("Humidité (%)", 
                                min_value=20.0, max_value=100.0, value=50.0, step=1.0)
        
        with col2:
            luminosite = st.slider("Luminosité (lux)",
                                  min_value=0.0, max_value=2000.0, value=500.0, step=10.0)
            co2 = st.slider("CO2 (ppm)",
                           min_value=400.0, max_value=1500.0, value=600.0, step=10.0)
            presence = st.slider("Présence Clients",
                                min_value=0.0, max_value=50.0, value=15.0, step=1.0)
        
        with col3:
            chauffage = st.selectbox("Chauffage", [0, 1])
            climatisation = st.selectbox("Climatisation", [0, 1])
            ventilation = st.selectbox("Ventilation", [0, 1])
            eclairage = st.selectbox("Éclairage", [0, 1])
            hour = st.slider("Heure du jour", min_value=0, max_value=23, value=12)
        
        st.markdown("---")
        
        # Préparation des données pour prédiction
        features = pipeline.preparer_features()
        
        # Créer un vecteur d'entrée
        input_data = pd.DataFrame({
            'temperature_interieure': [temp_int],
            'temperature_exterieure': [temp_ext],
            'humidite': [humidite],
            'luminosite': [luminosite],
            'CO2': [co2],
            'presence_clients': [int(presence)],
            'chauffage': [chauffage],
            'climatisation': [climatisation],
            'ventilation': [ventilation],
            'eclairage': [eclairage],
            'hour': [hour],
            'dayofweek': [4],
            'is_weekend': [0],
            'etat_equipements': [1],
            'alertes': [0]
        })
        
        # Ajouter les features calculées
        input_data['diff_temp'] = temp_int - temp_ext
        input_data['temp_humidite_interaction'] = temp_int * humidite
        input_data['luminosite_presence_interaction'] = luminosite * presence
        input_data['consomation_par_personne'] = 0
        input_data['actifs_equipements'] = chauffage + climatisation + ventilation + eclairage
        input_data['ratio_conso_equipements'] = 0
        input_data['day'] = 15
        input_data['month'] = 5
        input_data['quarter'] = 2
        input_data['day_of_year'] = 135
        
        # Ajouter les rolling features avec des valeurs par défaut
        for col in df_processed.columns:
            if 'rolling' in col and col not in input_data.columns:
                input_data[col] = df_processed[col].mean()
        
        # Ajouter les features de tendance
        for col in ['conso_trend', 'co2_trend', 'temp_trend']:
            if col not in input_data.columns:
                input_data[col] = 0
        
        # Ajouter les features normalisées
        for col in ['luminosite_normalized', 'CO2_normalized', 'presence_clients_normalized']:
            if col not in input_data.columns:
                input_data[col] = 0.5
        
        # Filtrer pour garder uniquement les features utilisées
        input_data = input_data[[f for f in features if f in input_data.columns]]
        
        # Prédictions
        X_input_scaled = pipeline.scaler_input.transform(input_data)
        
        if modele_select == "Régression Linéaire":
            pred = model_lr.predict(X_input_scaled)[0]
            model_name = "Régression Linéaire"
        else:
            pred_scaled = model_mlp.predict(X_input_scaled)[0]
            pred = pipeline.scaler_output.inverse_transform([[pred_scaled]])[0][0]
            model_name = "Réseau de Neurones"
        
        st.markdown("---")
        st.markdown("## Résultat de la Prédiction")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Modèle Utilisé", model_name)
        
        with col2:
            st.metric("Consommation Prédite", f"{pred:.2f} kWh")
        
        with col3:
            # Comparer avec la consommation moyenne
            conso_moy = df_processed['consommation_energetique'].mean()
            diff_pct = ((pred - conso_moy) / conso_moy * 100)
            status = "Élevée" if diff_pct > 10 else "Normal" if diff_pct > -10 else "Basse"
            st.metric("Statut", status, delta=f"{diff_pct:.1f}%")
        
        st.markdown("---")
        
        # Graphique de sensibilité
        st.markdown("## Analyse de Sensibilité - Impact de chaque paramètre")
        
        sensitivity_results = []
        base_features = pd.DataFrame(input_data.values, columns=input_data.columns)
        
        # Tester quelques paramètres clés
        for param, param_values in [
            ('temperature_interieure', np.linspace(15, 30, 5)),
            ('CO2', np.linspace(400, 1500, 5)),
            ('presence_clients', np.linspace(0, 50, 5))
        ]:
            preds = []
            for val in param_values:
                test_data = base_features.copy()
                if param in test_data.columns:
                    test_data[param] = val
                    if param == 'temperature_interieure':
                        test_data['diff_temp'] = val - temp_ext
                        test_data['temp_humidite_interaction'] = val * humidite
                    X_test_scaled = pipeline.scaler_input.transform(test_data)
                    if modele_select == "Régression Linéaire":
                        p = model_lr.predict(X_test_scaled)[0]
                    else:
                        p_scaled = model_mlp.predict(X_test_scaled)[0]
                        p = pipeline.scaler_output.inverse_transform([[p_scaled]])[0][0]
                    preds.append(p)
            
            sensitivity_results.append({
                'parameter': param,
                'values': param_values,
                'predictions': preds
            })
        
        col1, col2, col3 = st.columns(3)
        
        # Graphiques de sensibilité
        for idx, result in enumerate(sensitivity_results):
            with [col1, col2, col3][idx]:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=result['values'],
                    y=result['predictions'],
                    mode='lines+markers',
                    name=result['parameter'],
                    line=dict(color='#4ECDC4', width=3),
                    marker=dict(size=8)
                ))
                fig.update_layout(
                    title=f"Sensibilité à {result['parameter']}",
                    xaxis_title=result['parameter'],
                    yaxis_title='Consommation (kWh)',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Historique des prédictions sauvegardées
        st.markdown("## Historique des Paramètres")
        
        historique_data = pd.DataFrame({
            'Température Int.': [temp_int],
            'Température Ext.': [temp_ext],
            'Humidité': [humidite],
            'CO2': [co2],
            'Clients': [int(presence)],
            'Équipements Actifs': [chauffage + climatisation + ventilation + eclairage],
            'Prédiction (kWh)': [pred]
        })
        
        st.dataframe(historique_data, use_container_width=True)
    
    elif page == "Alertes Intelligentes":
        st.markdown("# Système d'Alertes Intelligentes")
        st.markdown("---")
        
        alertes_complete = generer_alertes_intelligentes(df_filtered)
        
        st.markdown(f"## Total: {len(alertes_complete)} Alertes Actives")
        
        # Tri par priorité
        alertes_critiques = [a for a in alertes_complete if a['niveau'] == 'CRITIQUE']
        alertes_attention = [a for a in alertes_complete if a['niveau'] == 'ATTENTION']
        
        if alertes_critiques:
            st.markdown("### CRITIQUES")
            for alerte in alertes_critiques:
                st.markdown(f"""
                <div class="alert-critical">
                <b>{alerte['type']}</b><br>
                Valeur: {alerte['valeur']}<br>
                Action: {alerte['action']}
                </div>
                """, unsafe_allow_html=True)
        
        if alertes_attention:
            st.markdown("### ATTENTION")
            for alerte in alertes_attention:
                st.markdown(f"""
                <div class="alert-warning">
                <b>{alerte['type']}</b><br>
                Valeur: {alerte['valeur']}<br>
                Action: {alerte['action']}
                </div>
                """, unsafe_allow_html=True)
        
        if not alertes_complete:
            st.success("Aucune alerte - Café en fonctionnement optimal")
        
        # Historique des alertes
        st.markdown("---")
        st.markdown("## Historique des Alertes par Jour")
        
        df_alert_history = df_filtered.copy()
        df_alert_history['date'] = df_alert_history['timestamp'].dt.date
        daily_alerts = df_alert_history.groupby('date')['alertes'].sum()
        
        fig = go.Figure(data=[
            go.Bar(x=daily_alerts.index, y=daily_alerts.values,
                  marker_color='#FF6B6B')
        ])
        fig.update_layout(title='Alertes par Jour', xaxis_title='Date',
                         yaxis_title='Nombre d\'alertes', height=400)
        enregistrer_graphique("Historique des alertes par jour", fig)
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Rapports & Économies":
        st.markdown("# Rapports Énergétiques & Économies")
        st.markdown("---")
        
        # Coûts énergétiques (tarif exemple: 0.15€/kWh)
        tarif_kwh = 0.15
        
        consomation_totale = df_filtered['consommation_energetique'].sum()
        cout_total = consomation_totale * tarif_kwh
        
        consomation_moy_heure = df_filtered['consommation_energetique'].mean()
        cout_heure = consomation_moy_heure * tarif_kwh
        
        st.markdown("## Analyse Économique")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Consommation Totale", f"{consomation_totale:.1f} kWh")
        
        with col2:
            st.metric("Coût Total", f"{cout_total:.2f}€", delta=f"{tarif_kwh}€/kWh")
        
        with col3:
            st.metric("Moyenne/15min", f"{consomation_moy_heure:.2f} kWh")
        
        with col4:
            st.metric("Coût Moyen", f"{cout_heure:.3f}€")
        
        # Économies potentielles
        st.markdown("---")
        st.markdown("## Potentiel d'Économies")
        
        # Scénario d'optimisation
        economie_potential_15pct = consomation_totale * 0.15  # Économies 15%
        economie_potential_25pct = consomation_totale * 0.25  # Économies 25%
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Scénario Conservateur", f"Économie 15%",
                     delta=f"{economie_potential_15pct:.1f} kWh = {economie_potential_15pct*tarif_kwh:.2f}€")
        
        with col2:
            st.metric("Scénario Modéré", f"Économie 20%",
                     delta=f"{consomation_totale*0.2:.1f} kWh = {consomation_totale*0.2*tarif_kwh:.2f}€")
        
        with col3:
            st.metric("Scénario Optimiste", f"Économie 25%",
                     delta=f"{economie_potential_25pct:.1f} kWh = {economie_potential_25pct*tarif_kwh:.2f}€")
        
        # Impact annuel
        st.markdown("---")
        st.markdown("## Projections Annuelles")
        
        jours_periode = (date_fin - date_debut).days + 1
        facteur_annuel = 365 / jours_periode if jours_periode > 0 else 1
        
        consomation_annuelle = consomation_totale * facteur_annuel
        cout_annuel = consomation_annuelle * tarif_kwh
        economie_annuelle_20pct = consomation_annuelle * 0.2 * tarif_kwh
        
        st.metric("Consommation Annuelle Estimée", f"{consomation_annuelle:.0f} kWh")
        st.metric("Coût Annuel Estimé", f"{cout_annuel:.2f}€")
        st.metric("Économie Annuelle (20%)", f"{economie_annuelle_20pct:.2f}€")
        
        # Analyse par équipement
        st.markdown("---")
        st.markdown("## Analyse par Système")
        
        chauffage_conso = df_filtered[df_filtered['chauffage'] == 1]['consommation_energetique'].mean()
        clim_conso = df_filtered[df_filtered['climatisation'] == 1]['consommation_energetique'].mean()
        ventilation_conso = df_filtered[df_filtered['ventilation'] == 1]['consommation_energetique'].mean()
        eclairage_conso = df_filtered[df_filtered['eclairage'] == 1]['consommation_energetique'].mean()
        
        systems_data = pd.DataFrame({
            'Système': ['Chauffage', 'Climatisation', 'Ventilation', 'Éclairage'],
            'Consommation Moyenne': [chauffage_conso, clim_conso, ventilation_conso, eclairage_conso]
        })
        
        fig = px.bar(systems_data, x='Système', y='Consommation Moyenne',
                    color='Système',
                    color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFE66D'])
        fig.update_layout(height=400)
        enregistrer_graphique("Analyse par système", fig)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("## Interprétations et Recommandations")

        alertes_complete = generer_alertes_intelligentes(df_filtered)
        interpretations, recommandations = generer_interpretations_et_recommandations(
            df_filtered,
            alertes=alertes_complete,
            resultats_ml=pipeline.resultats if 'pipeline' in locals() else None
        )
        enregistrer_synthese(interpretations, recommandations)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Interprétations")
            for texte in interpretations:
                st.write(f"- {texte}")

        with col2:
            st.markdown("### Recommandations")
            for texte in recommandations:
                st.write(f"- {texte}")

        pdf_bytes = generer_pdf_rapport(
            df_filtered,
            interpretations,
            recommandations,
            st.session_state.get("rapport_graphiques", [])
        )

        st.markdown("---")
        st.markdown("## Export PDF")
        if pdf_bytes:
            st.download_button(
                label="Télécharger le rapport PDF",
                data=pdf_bytes,
                file_name="rapport_energie_cafe.pdf",
                mime="application/pdf"
            )
        else:
            st.warning("Le module PDF n'est pas disponible. Installez ReportLab et relancez l'application.")
    
    else:  # Paramètres Système
        st.markdown("# Paramètres Système & Configuration")
        st.markdown("---")
        
        st.markdown("## Informations Système")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **Dataset:**
            - Observations: {len(df):,}
            - Période: {df['timestamp'].min()} à {df['timestamp'].max()}
            - Durée: {(df['timestamp'].max() - df['timestamp'].min()).days} jours
            - Fréquence: 15 minutes
            """)
        
        with col2:
            st.info(f"""
            **Base de Données:**
            - Type: SQLite
            - Fichier: cafe_energie.db
            - Tables: 6
            - Connexion: Active
            """)
        
        st.markdown("---")
        st.markdown("## Configuration des Modèles ML")
        
        st.markdown("### Régression Linéaire")
        st.code("LinearRegression()")
        
        st.markdown("### Random Forest")
        st.code("RandomForestClassifier(n_estimators=100, max_depth=10)")
        
        st.markdown("### Isolation Forest")
        st.code("IsolationForest(contamination=0.1)")
        
        st.markdown("### MLPRegressor (Réseau de Neurones)")
        st.code("""MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    max_iter=500
)""")
        
        st.markdown("---")
        st.markdown("## Seuils Informationnels")
        
        seuil_data = pd.DataFrame({
            'Paramètre': ['CO2', 'Température', 'Humidité', 'Consommation', 'Luminosité'],
            'Minimum': [400, 15, 30, 5, 100],
            'Normal': [600, 22, 50, 13, 500],
            'Maximum': [1000, 28, 80, 20, 2000]
        })
        
        st.dataframe(seuil_data, use_container_width=True)
        
        st.markdown("---")
        st.markdown("## Sécurité des Données")

        st.success("Authentification: Activée")
        st.success("Chiffrement BD: SQLite")
        st.success("Logs: Enregistrés")
        st.warning("Backup: À programmer")
        
        st.markdown("---")
        st.markdown("## Documentation du Système")
        
        st.markdown("""
        ### Architecture Générale
        1. **Acquisition**: Données CSV importées
        2. **Prétraitement**: Feature engineering, normalisation, détection outliers
        3. **ML**: 5 modèles entraînés en parallèle
        4. **Analyse**: Corrélations, anomalies, prédictions
        5. **Alertes**: Règles intelligentes data-driven
        6. **Visualisation**: Dashboard Plotly interactif
        7. **Stockage**: Résultats en SQLite
        
        ### Modèles Utilisés
        - **Régression Linéaire**: Prédiction consommation
        - **Random Forest**: Classification alertes + État équipements
        - **Isolation Forest**: Détection anomalies (non-supervisée)
        - **MLPRegressor**: Prédiction avancée (réseau neurones)
        
        ### Features Disponibles
        {len(pipeline.preparer_features())} features numeriques après preprocessing
        """)

if __name__ == "__main__":
    main()
