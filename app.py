# app.py
# NEXUS - Recomendación de carga de catering por vuelo (MVP Streamlit)
# --------------------------------------------------------------
# Versión mejorada con diseño visual moderno tipo dashboard aeronáutico

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from typing import Optional, Tuple, List, Any
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# --------------------------------------------------------------
st.set_page_config(
    page_title="NEXUS - Intelligent Catering Optimization",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------------
# CSS PERSONALIZADO PARA DISEÑO MODERNO
# --------------------------------------------------------------
st.markdown("""
<style>
:root {
  --prim:#1e3c72; --prim-2:#2a5298; --acc:#FFD700;
  --bg:#f6f8fb; --card:#ffffff; --text:#1b1f24; --muted:#6b7280;
  --radius:12px; --shadow:0 6px 12px rgba(0,0,0,.08);
}
/* Fondo */
.main { background: linear-gradient(135deg, var(--bg) 0%, #e9eef7 100%); }
/* Header */
.main-header { background: linear-gradient(90deg, var(--prim), var(--prim-2)); 
  padding: 1.75rem; border-radius: var(--radius); box-shadow: var(--shadow); }
.main-header h1 { color: var(--acc); font-size: 2.6rem; font-weight:700; margin:0; }
.main-header p { color: #e7eaf3; font-size:1.1rem; margin:.4rem 0 0; }
/* Secciones */
.section-container { 
  background: transparent;
  padding:1.25rem; 
  border-radius:var(--radius);
  box-shadow: none;
  margin-bottom:1rem; 
  border-left: none;
}
/* Cards KPI */
.kpi { text-align:center; color:#fff; border-radius:var(--radius);
  padding:1.25rem; box-shadow:var(--shadow); }
.kpi--1{ background: linear-gradient(135deg, #4158D0 0%, #1e3c72 100%); }
.kpi--2{ background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); }
.kpi--3{ background: linear-gradient(135deg, #2a5298 0%, #1e3c72 100%); }
/* Botón primario */
.stButton > button { background: linear-gradient(90deg, var(--acc), #ffc933);
  color: var(--prim); font-weight:700; border:0; border-radius:10px; padding:.6rem 1.2rem; }
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 8px 16px rgba(255,215,0,.3); }
/* Tabs */
.stTabs [data-baseweb="tab"] { background:#eef2ff; border-radius:8px 8px 0 0; padding:10px 16px; color:var(--prim); font-weight:600; }
.stTabs [aria-selected="true"] { background: linear-gradient(90deg, var(--prim), var(--prim-2)); color:#fff !important; }
/* Sidebar */
[data-testid="stSidebar"] { 
  background: linear-gradient(180deg, var(--prim), var(--prim-2)); 
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown {
  color: #fff !important;
}
/* Asegurar que los inputs del sidebar sean legibles */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] select,
[data-testid="stSidebar"] .stSelectbox label {
  color: #1b1f24 !important;
  background-color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------
# HEADER PRINCIPAL
# --------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>NEXUS</h1>
    <p>Intelligent Catering Optimization Platform</p>
    <p style="font-size: 0.9rem; opacity: 0.8;">Advanced predictive analytics for Gategroup flight operations</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------
# RUTAS LOCALES
# --------------------------------------------------------------
CSV_PATH = "C:/Users/ismae/Downloads/[HackMTY2025]_ConsumptionPrediction_Dataset_v1.csv"
MODEL_PATH = "modelo_consumo.pkl"

# --------------------------------------------------------------
# ESPECIFICACIONES
# --------------------------------------------------------------
NUMERIC_COLS = [
    "Passenger_Count", "Standard_Specification_Qty",
    "Quantity_Consumed", "Quantity_Returned", "Unit_Cost"
]
CATEGORICAL_TO_ENCODE = ["Origin", "Flight_Type", "Service_Type", "Product_ID", "Product_Name"]

FEATURE_COLS = [
    "Origin_Encoded", "Flight_Type_Encoded", "Service_Type_Encoded",
    "Product_ID_Encoded", "Product_Name_Encoded", "Passenger_Count",
    "Standard_Specification_Qty", "DayOfWeek", "Month", "Unit_Cost",
    "Qty_Per_Passenger", "Has_Feedback", "Ran_Out", "Low_Demand"
]

PRODUCT_WEIGHTS = {
    "Juice 200ml": 0.22,
    "Still Water 500ml": 0.55,
    "Sparkling Water 330ml": 0.40,
    "Snack Box Economy": 0.35,
    "Butter Cookies 75g": 0.08,
    "Bread Roll Pack": 0.10,
    "Instant Coffee Stick": 0.02,
    "Herbal Tea Bag": 0.01,
    "Chocolate Bar 50g": 0.06,
    "Mixed Nuts 30g": 0.03
}

TROLLEY_CAPACITY_KG = 80.0
TROLLEY_EMPTY_WEIGHT_KG = 14.0

# --------------------------------------------------------------
# FUNCIONES DE CARGA
# --------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_model(path: str = MODEL_PATH) -> Tuple[Optional[Any], Optional[str], Optional[List[str]]]:
    try:
        obj = joblib.load(path)
    except Exception as e:
        return None, f"No se pudo cargar el modelo desde '{path}'. Detalle: {e}", None

    if hasattr(obj, "predict"):
        return obj, None, None

    if isinstance(obj, dict):
        est = None
        for k in ["model", "estimator", "pipeline", "rf"]:
            if k in obj and hasattr(obj[k], "predict"):
                est = obj[k]
                break

        feat_cols = None
        for k in ["feature_cols", "features", "feature_columns"]:
            if k in obj and isinstance(obj[k], (list, tuple)):
                feat_cols = list(obj[k])
                break

        if est is None:
            return None, ("El .pkl es un dict pero no contiene un estimador con .predict "
                          "en las llaves ('model','estimator','pipeline','rf')."), None
        return est, None, feat_cols

    return None, (f"Objeto {type(obj)} no soportado: no es estimador ni dict."), None

@st.cache_data(show_spinner=True)
def load_dataset(path: str = CSV_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"No se encontró el CSV en la ruta:\n{path}")
        st.stop()
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"No se pudo cargar el CSV local: {e}")
        st.stop()

# --------------------------------------------------------------
# FUNCIONES DE PREPROCESAMIENTO
# --------------------------------------------------------------
def preprocess_base(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = 0.0

    df["DayOfWeek"] = df["Date"].dt.dayofweek if "Date" in df.columns else 0
    df["Month"] = df["Date"].dt.month if "Date" in df.columns else 0

    df["Qty_Per_Passenger"] = (
        df["Standard_Specification_Qty"].fillna(0)
        .div(df["Passenger_Count"].replace(0, np.nan))
        .fillna(0)
    )

    if "Crew_Feedback" in df.columns:
        cf = df["Crew_Feedback"].astype(str).str.lower()
        df["Has_Feedback"] = df["Crew_Feedback"].notna().astype(int)
        df["Ran_Out"] = cf.str.contains("ran out", na=False).astype(int)
        df["Low_Demand"] = cf.str.contains("low demand", na=False).astype(int)
    else:
        df["Has_Feedback"] = 0
        df["Ran_Out"] = 0
        df["Low_Demand"] = 0

    for c in ["Passenger_Count", "Standard_Specification_Qty", "DayOfWeek", "Month",
              "Unit_Cost", "Qty_Per_Passenger", "Has_Feedback", "Ran_Out", "Low_Demand"]:
        df[c] = df[c].fillna(0)

    return df

def fit_and_apply_encoders(df: pd.DataFrame):
    df = df.copy()
    encoders = {}
    for col in CATEGORICAL_TO_ENCODE:
        if col not in df.columns:
            df[col] = "Unknown"
        le = LabelEncoder()
        vals = df[col].astype(str).fillna("Unknown")
        le.fit(vals)
        df[f"{col}_Encoded"] = le.transform(vals)
        encoders[col] = le
    return df, encoders

def ensure_feature_columns(df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
    return [c for c in feature_cols if c not in df.columns]

def compute_recommendations(df_flight: pd.DataFrame, model: Any, buffer_pct: float, feature_cols: List[str]) -> pd.DataFrame:
    work = df_flight.copy()

    missing = ensure_feature_columns(work, feature_cols)
    if missing:
        st.warning(f"Faltan columnas requeridas para el modelo: {missing}. No se realizarán predicciones.")
        work["Predicted_Consumption"] = np.nan
        work["Optimal_Specification"] = work.get("Standard_Specification_Qty", 0)
        work["Peso_Actual_kg"] = 0.0
        work["Peso_Optimo_kg"] = 0.0
        work["Peso_Ahorrado_kg"] = 0.0
        return work

    try:
        X = work[feature_cols].to_numpy()
        y_pred = model.predict(X)
    except Exception as e:
        st.error(f"Error al predecir con el modelo: {e}")
        work["Predicted_Consumption"] = np.nan
        work["Optimal_Specification"] = work.get("Standard_Specification_Qty", 0)
        work["Peso_Actual_kg"] = 0.0
        work["Peso_Optimo_kg"] = 0.0
        work["Peso_Ahorrado_kg"] = 0.0
        return work

    work["Predicted_Consumption"] = np.ceil(np.maximum(y_pred, 0)).astype(int)

    with_buffer = np.ceil(work["Predicted_Consumption"] * (1.0 + buffer_pct)).astype(int)
    work["Optimal_Specification"] = np.minimum(
        with_buffer,
        work["Standard_Specification_Qty"].fillna(0).astype(int)
    )

    if "Product_Name" not in work.columns:
        work["Product_Name"] = "Unknown"
    work["Product_Weight_kg"] = work["Product_Name"].map(PRODUCT_WEIGHTS).fillna(0.0)

    missing_weights = work.loc[work["Product_Weight_kg"] == 0.0, "Product_Name"].unique()
    if len(missing_weights) > 0:
        st.warning("Sin peso definido para: " + ", ".join(sorted(map(str, missing_weights))) + ". Se asume 0 kg.")

    work["Peso_Actual_kg"] = work["Standard_Specification_Qty"].fillna(0) * work["Product_Weight_kg"]
    work["Peso_Optimo_kg"] = work["Optimal_Specification"].fillna(0) * work["Product_Weight_kg"]
    work["Peso_Ahorrado_kg"] = np.clip(work["Peso_Actual_kg"] - work["Peso_Optimo_kg"], 0, None)

    return work

def summarize_impacts(work: pd.DataFrame):
    peso_total_ahorrado = work["Peso_Ahorrado_kg"].sum()
    combustible_ahorrado = peso_total_ahorrado * 0.03
    ahorro_dinero = combustible_ahorrado * 0.8
    return peso_total_ahorrado, combustible_ahorrado, ahorro_dinero

def calcular_trolleys(work: pd.DataFrame, capacidad_kg: float = TROLLEY_CAPACITY_KG) -> Tuple[int, float]:
    peso_total_optimo = work["Peso_Optimo_kg"].sum()
    num_trolleys = int(np.ceil(peso_total_optimo / capacidad_kg))
    if num_trolleys == 0:
        return 0, 0.0
    peso_restante = peso_total_optimo % capacidad_kg
    pct_ultimo = (peso_restante / capacidad_kg * 100) if peso_restante > 0 else 100.0
    return num_trolleys, pct_ultimo

# --------------------------------------------------------------
# SIDEBAR MEJORADO
# --------------------------------------------------------------
with st.sidebar:
    
    with st.spinner("🔄 Loading AI model..."):
        model, model_err, feature_cols_from_pkl = load_model(MODEL_PATH)
    
    if model_err:
        st.error(f"❌ {model_err}")
        st.stop()
#    else:
#        st.success("✅ Model loaded successfully")

    if feature_cols_from_pkl:
        FEATURE_COLS = feature_cols_from_pkl

    with st.spinner("📊 Loading flight database..."):
        df_raw = load_dataset(CSV_PATH)
        df_base = preprocess_base(df_raw)
        df_enc, _ = fit_and_apply_encoders(df_base)

    if "Flight_ID" not in df_enc.columns:
        st.error("❌ El CSV no contiene la columna 'Flight_ID'.")
        st.stop()

    st.markdown("### Flight Selection")
    flight_options = df_enc["Flight_ID"].astype(str).sort_values().unique().tolist()
    selected_flight = st.selectbox(
        "Choose Flight ID",
        flight_options,
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### Optimization Parameters")
    
    buffer_pct_ui = st.slider(
        "Safety Buffer (%)",
        5, 20, 10, 1,
        help="Extra margin over prediction to prevent stockouts"
    )
    buffer_fraction = buffer_pct_ui / 100.0
    
    # ⭐ NUEVA SECCIÓN: Ajuste de Pasajeros ⭐
    st.markdown("---")
    st.markdown("### Passenger Adjustment")
    
    # Obtener datos del vuelo seleccionado para mostrar pasajeros originales
    df_flight_original = df_enc[df_enc["Flight_ID"].astype(str) == str(selected_flight)].copy()
    
    if not df_flight_original.empty:
        original_passenger_count = int(df_flight_original.iloc[0]['Passenger_Count'])
        
        adjusted_passenger_count = st.number_input(
            "Modify Passenger Count",
            min_value=1,
            max_value=500,
            value=original_passenger_count,
            step=1,
            help="Adjust passenger count to simulate different load scenarios",
            label_visibility="collapsed"
        )
        
        # Indicador visual de cambio
        if adjusted_passenger_count != original_passenger_count:
            delta = adjusted_passenger_count - original_passenger_count
            delta_pct = (delta / original_passenger_count * 100)
            st.markdown(f"""
            <div style="background: rgba(255, 215, 0, 0.2); padding: 0.5rem; border-radius: 8px; border-left: 4px solid #FFD700;">
                <strong>Simulation Active</strong><br/>
                Original: {original_passenger_count}<br/>
                Adjusted: {adjusted_passenger_count}<br/>
                Change: {delta:+d} ({delta_pct:+.1f}%)
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: rgba(67, 233, 123, 0.2); padding: 0.5rem; border-radius: 8px; border-left: 4px solid #43e97b;">
                <strong>Using Original Data</strong><br/>
                Passengers: {original_passenger_count}
            </div>
            """, unsafe_allow_html=True)
    else:
        original_passenger_count = 0
        adjusted_passenger_count = 0
    
    st.markdown("---")
    st.markdown("### System Info")
    st.info(f"**Database:** {len(df_enc)} records\n\n**Flights:** {len(flight_options)}")

# --------------------------------------------------------------
# CONTENIDO PRINCIPAL
# --------------------------------------------------------------

# ⭐ APLICAR AJUSTE DE PASAJEROS ANTES DE FILTRAR ⭐
df_enc_adjusted = df_enc.copy()

# Verificar que tengamos un vuelo válido
df_flight_check = df_enc[df_enc["Flight_ID"].astype(str) == str(selected_flight)].copy()

if not df_flight_check.empty:
    # Si hay ajuste de pasajeros, aplicarlo al dataset
    if adjusted_passenger_count != original_passenger_count:
        # Modificar solo las filas del vuelo seleccionado
        mask = df_enc_adjusted["Flight_ID"].astype(str) == str(selected_flight)
        df_enc_adjusted.loc[mask, 'Passenger_Count'] = adjusted_passenger_count
        
        # Recalcular Qty_Per_Passenger con el nuevo valor
        df_enc_adjusted.loc[mask, 'Qty_Per_Passenger'] = (
            df_enc_adjusted.loc[mask, 'Standard_Specification_Qty'].fillna(0)
            .div(adjusted_passenger_count)
        )

# Ahora sí filtrar el vuelo con los datos ajustados
df_flight = df_enc_adjusted[df_enc_adjusted["Flight_ID"].astype(str) == str(selected_flight)].copy()

if df_flight.empty:
    st.warning("⚠️ No se encontraron filas para el vuelo seleccionado.")
    st.stop()

# --------------------------------------------------------------
# CONTEXTO DEL VUELO CON DISEÑO MEJORADO
# --------------------------------------------------------------
st.markdown('<div class="section-container">', unsafe_allow_html=True)
st.markdown("## Flight Context & Details")

row0 = df_flight.iloc[0]

col1, col2, col3, col4, col5 = st.columns(5)

# Estilo uniforme azul marino para todas las tarjetas
card_style = """
    text-align: center; 
    padding: 1rem; 
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
    border-radius: 10px; 
    color: white;
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
"""

with col1:
    st.markdown(f"""
    <div style="{card_style}">
        <div style="font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;">{row0.get('Origin', 'N/A')}</div>
        <div style="font-size: 0.8rem; opacity: 0.9;">ORIGIN</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style="{card_style}">
        <div style="font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;">{row0.get('Flight_Type', 'N/A')}</div>
        <div style="font-size: 0.8rem; opacity: 0.9;">FLIGHT TYPE</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div style="{card_style}">
        <div style="font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;">{row0.get('Service_Type', 'N/A')}</div>
        <div style="font-size: 0.8rem; opacity: 0.9;">SERVICE</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    # Mostrar pasajeros con indicador visual si fue modificado
    passenger_display = int(row0.get('Passenger_Count', 0))
    is_modified = adjusted_passenger_count != original_passenger_count
    
    # Cambiar color si está modificado
    card_style_pax = card_style.replace('#1e3c72 0%, #2a5298 100%', 
                                        '#FFD700 0%, #FFA500 100%' if is_modified else '#1e3c72 0%, #2a5298 100%')
    
    st.markdown(f"""
    <div style="{card_style_pax}">
        <div style="font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;">{passenger_display}</div>
        <div style="font-size: 0.8rem; opacity: 0.9;">PASSENGERS {'(ADJUSTED)' if is_modified else ''}</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div style="{card_style}">
        <div style="font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;">{str(row0.get('Date'))[:10]}</div>
        <div style="font-size: 0.8rem; opacity: 0.9;">DATE</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------------------
# PREDICCIÓN CON TABS
# --------------------------------------------------------------
st.markdown('<div class="section-container">', unsafe_allow_html=True)
st.markdown("## AI-Powered Consumption Prediction")

work = compute_recommendations(df_flight, model, buffer_fraction, FEATURE_COLS)

tab1, tab2, tab3 = st.tabs(["Prediction Table", "Visual Analysis", "Details"])

with tab1:
    visible_cols = [
        "Product_Name", "Standard_Specification_Qty",
        "Predicted_Consumption", "Optimal_Specification", "Quantity_Consumed"
    ]
    for col in visible_cols:
        if col not in work.columns:
            work[col] = np.nan
    
    display_df = work[visible_cols].copy()
    display_df.columns = ["Product", "Standard Qty", "AI Prediction", "Optimal Qty", "Actual Consumed"]
    st.dataframe(display_df, use_container_width=True, height=400)

with tab2:
    # Gráfico de barras comparativo
    chart_data = work[["Product_Name", "Standard_Specification_Qty", "Optimal_Specification"]].copy()
    chart_data.columns = ["Product", "Standard", "Optimal"]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Standard Qty',
        x=chart_data["Product"],
        y=chart_data["Standard"],
        marker_color='#667eea'
    ))
    fig.add_trace(go.Bar(
        name='Optimal Qty',
        x=chart_data["Product"],
        y=chart_data["Optimal"],
        marker_color='#FFD700'
    ))
    
    fig.update_layout(
        title="Standard vs Optimal Quantities",
        barmode='group',
        xaxis_tickangle=-45,
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown(f"""
    **Model Confidence:** High  
    **Buffer Applied:** {buffer_pct_ui}%  
    **Products Analyzed:** {len(work)}  
    **Prediction Algorithm:** Random Forest Regression  
    **Passenger Count:** {adjusted_passenger_count} {'(Adjusted from ' + str(original_passenger_count) + ')' if adjusted_passenger_count != original_passenger_count else '(Original)'}
    """)

st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------------------
# KPIs MEJORADOS CON GRÁFICOS
# --------------------------------------------------------------
st.markdown('<div class="section-container">', unsafe_allow_html=True)
st.markdown("## 📉 Impact & Savings Dashboard")

peso_total_ahorrado, combustible_ahorrado, ahorro_dinero = summarize_impacts(work)

col1, col2, col3, col4 = st.columns(4)

# Estilo uniforme azul marino para KPIs
kpi_style = """
    text-align: center; 
    padding: 1.5rem; 
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
    border-radius: 10px; 
    color: white; 
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
"""

with col1:
    st.markdown(f"""
    <div style="{kpi_style}">
        <div style="font-size: 2.5rem;">⚖️</div>
        <div style="font-size: 2rem; font-weight: bold; margin: 0.5rem 0;">{peso_total_ahorrado:.2f}</div>
        <div style="font-size: 0.85rem; opacity: 0.9; text-transform: uppercase;">kg Weight Saved</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style="{kpi_style}">
        <div style="font-size: 2.5rem;">⛽</div>
        <div style="font-size: 2rem; font-weight: bold; margin: 0.5rem 0;">{combustible_ahorrado:.2f}</div>
        <div style="font-size: 0.85rem; opacity: 0.9; text-transform: uppercase;">kg Fuel Saved</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div style="{kpi_style}">
        <div style="font-size: 2.5rem;">💰</div>
        <div style="font-size: 2rem; font-weight: bold; margin: 0.5rem 0;">${ahorro_dinero:.2f}</div>
        <div style="font-size: 0.85rem; opacity: 0.9; text-transform: uppercase;">USD Cost Savings</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    reduction_pct = (peso_total_ahorrado / work["Peso_Actual_kg"].sum() * 100) if work["Peso_Actual_kg"].sum() > 0 else 0
    st.markdown(f"""
    <div style="{kpi_style}">
        <div style="font-size: 2.5rem;">📊</div>
        <div style="font-size: 2rem; font-weight: bold; margin: 0.5rem 0;">{reduction_pct:.1f}%</div>
        <div style="font-size: 0.85rem; opacity: 0.9; text-transform: uppercase;">Weight Reduction</div>
    </div>
    """, unsafe_allow_html=True)

# Gráfico de distribución de peso
st.markdown("### Weight Distribution by Product")
weight_dist = work[["Product_Name", "Peso_Optimo_kg"]].copy()
weight_dist = weight_dist[weight_dist["Peso_Optimo_kg"] > 0]

if not weight_dist.empty:
    fig_pie = px.pie(
        weight_dist,
        values="Peso_Optimo_kg",
        names="Product_Name",
        color_discrete_sequence=px.colors.sequential.RdBu,
        hole=0.4
    )
    fig_pie.update_layout(
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------------------
# TROLLEYS CON VISUALIZACIÓN MEJORADA
# --------------------------------------------------------------
st.markdown('<div class="section-container">', unsafe_allow_html=True)
st.markdown("## Trolley Configuration & Loading Plan")

num_trolleys, pct_ultimo = calcular_trolleys(work, TROLLEY_CAPACITY_KG)
peso_optimo_total = work["Peso_Optimo_kg"].sum()

col1, col2 = st.columns([2, 1])

with col1:
    if num_trolleys > 0:
        st.markdown(f"""
        **Total Optimal Weight:** `{peso_optimo_total:.2f} kg`  
        **Trolley Capacity:** `{TROLLEY_CAPACITY_KG:.1f} kg`  
        **Trolleys Required:** `{num_trolleys}`
        """)
        
        # Visualización de trolleys
        for i in range(num_trolleys):
            if i == num_trolleys - 1:
                pct = pct_ultimo
            else:
                pct = 100.0
            
            st.progress(
                min(pct / 100, 1.0),
                text=f"🧳 Trolley {i+1}: {pct:.1f}% loaded ({pct * TROLLEY_CAPACITY_KG / 100:.1f} kg)"
            )
    else:
        st.info("ℹ️ No optimal weight calculated yet.")

with col2:
    if num_trolleys > 0:
        # Gráfico de gauge para último trolley
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pct_ultimo,
            title={'text': "Last Trolley Utilization"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#FFD700"},
                'steps': [
                    {'range': [0, 50], 'color': "#E0E0E0"},
                    {'range': [50, 80], 'color': "#B0B0B0"},
                    {'range': [80, 100], 'color': "#808080"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------------------
# CHECKLIST + EXPORT MEJORADO
# --------------------------------------------------------------
st.markdown('<div class="section-container">', unsafe_allow_html=True)
st.markdown("## Packing Checklist & Export")

checklist_df = work[["Flight_ID", "Product_Name", "Optimal_Specification"]].copy().sort_values("Product_Name")

if checklist_df.empty:
    st.warning("⚠️ No hay recomendaciones para mostrar.")
else:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        for idx, r in enumerate(checklist_df.iterrows(), 1):
            _, row = r
            st.markdown(f"""
            <div style="padding: 0.5rem; margin: 0.3rem 0; background: #f5f5f5; border-radius: 8px; 
                        border-left: 4px solid #2a5298;">
                {idx}. Pack <strong>{int(row['Optimal_Specification'])}</strong> units of <strong>{row['Product_Name']}</strong>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Nombre de archivo con indicador de ajuste
        filename_suffix = f"_{adjusted_passenger_count}pax" if adjusted_passenger_count != original_passenger_count else ""
        st.download_button(
            label="💾 Export CSV",
            data=checklist_df.to_csv(index=False).encode("utf-8"),
            file_name=f"nexus_packing_{selected_flight}{filename_suffix}.csv",
            mime="text/csv",
            use_container_width=True
        )

st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------------------
# FOOTER
# --------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; opacity: 0.7;">
    <p style="margin: 0;">🤖 <strong>NEXUS AI Engine</strong> | Powered by Machine Learning</p>
    <p style="margin: 0.3rem 0 0 0; font-size: 0.85rem;">
        Predictive model automatically adjusts to flight characteristics and historical patterns
    </p>
</div>
""", unsafe_allow_html=True)
