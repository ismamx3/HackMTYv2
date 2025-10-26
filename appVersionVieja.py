# app.py
# NEXUS - Recomendación de carga de catering por vuelo (MVP Streamlit)
# --------------------------------------------------------------
# Versión offline (lee CSV local) y compatible con .pkl que trae dict {"model": ..., "feature_cols": ...}

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from typing import Optional, Tuple, List, Any

# --------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# --------------------------------------------------------------
st.set_page_config(
    page_title="NEXUS - Recomendación de Catering por Vuelo",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("✈️ NEXUS — Recomendación de carga por vuelo")
st.caption("MVP para operadores de Gategroup — Predicción de consumo y ahorros de peso/combustible/costo")

# --------------------------------------------------------------
# RUTAS LOCALES (ajusta el CSV si cambia tu usuario/ruta)
# --------------------------------------------------------------
CSV_PATH = "/Users/ismael/Downloads/[HackMTY2025]_ConsumptionPrediction_Dataset_v1.csv"
MODEL_PATH = "modelo_consumo.pkl"

# --------------------------------------------------------------
# ESPECIFICACIONES
# --------------------------------------------------------------
NUMERIC_COLS = [
    "Passenger_Count", "Standard_Specification_Qty",
    "Quantity_Consumed", "Quantity_Returned", "Unit_Cost"
]
CATEGORICAL_TO_ENCODE = ["Origin", "Flight_Type", "Service_Type", "Product_ID", "Product_Name"]

# Orden por defecto; si el pkl trae su propio orden, se sobrescribe.
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
# --------------------------------------------------------------
# PARÁMETROS DE TROLLEY
# --------------------------------------------------------------
TROLLEY_CAPACITY_KG = 80.0  # capacidad máxima por trolley (carga útil)
TROLLEY_EMPTY_WEIGHT_KG = 14.0  # peso del trolley vacío (opcional, para cálculo realista)

# --------------------------------------------------------------
# CARGA DE MODELO Y CSV
# --------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_model(path: str = MODEL_PATH) -> Tuple[Optional[Any], Optional[str], Optional[List[str]]]:
    """
    Devuelve (estimador, error, feature_cols_desde_pkl|None).
    Soporta:
      - Estimador directo con .predict
      - Dict con llaves: 'model'/'estimator'/'pipeline'/'rf' + opcional 'feature_cols'/'features'/'feature_columns'
    """
    try:
        obj = joblib.load(path)
    except Exception as e:
        return None, f"No se pudo cargar el modelo desde '{path}'. Detalle: {e}", None

    # Caso 1: estimador directo
    if hasattr(obj, "predict"):
        return obj, None, None

    # Caso 2: dict contenedor
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
# PREPROCESAMIENTO
# --------------------------------------------------------------
def preprocess_base(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Fechas
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Numéricas
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = 0.0

    # Calendario
    df["DayOfWeek"] = df["Date"].dt.dayofweek if "Date" in df.columns else 0
    df["Month"] = df["Date"].dt.month if "Date" in df.columns else 0

    # Qty por pasajero (pandas puro; evita ndarray.fillna)
    df["Qty_Per_Passenger"] = (
        df["Standard_Specification_Qty"].fillna(0)
        .div(df["Passenger_Count"].replace(0, np.nan))
        .fillna(0)
    )

    # Feedback
    if "Crew_Feedback" in df.columns:
        cf = df["Crew_Feedback"].astype(str).str.lower()
        df["Has_Feedback"] = df["Crew_Feedback"].notna().astype(int)
        df["Ran_Out"] = cf.str.contains("ran out", na=False).astype(int)
        df["Low_Demand"] = cf.str.contains("low demand", na=False).astype(int)
    else:
        df["Has_Feedback"] = 0
        df["Ran_Out"] = 0
        df["Low_Demand"] = 0

    # Rellenos finales
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

# --------------------------------------------------------------
# INFERENCIA Y CÁLCULOS
# --------------------------------------------------------------
def compute_recommendations(df_flight: pd.DataFrame, model: Any, buffer_pct: float, feature_cols: List[str]) -> pd.DataFrame:
    work = df_flight.copy()

    # Validar features
    missing = ensure_feature_columns(work, feature_cols)
    if missing:
        st.warning(f"Faltan columnas requeridas para el modelo: {missing}. No se realizarán predicciones.")
        work["Predicted_Consumption"] = np.nan
        work["Optimal_Specification"] = work.get("Standard_Specification_Qty", 0)
        work["Peso_Actual_kg"] = 0.0
        work["Peso_Optimo_kg"] = 0.0
        work["Peso_Ahorrado_kg"] = 0.0
        return work

    # Predicción
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

    # Política de empaque con buffer (suma % pero no excede el estándar)
    with_buffer = np.ceil(work["Predicted_Consumption"] * (1.0 + buffer_pct)).astype(int)
    work["Optimal_Specification"] = np.minimum(
        with_buffer,
        work["Standard_Specification_Qty"].fillna(0).astype(int)
    )

    # Pesos
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
    """
    Calcula el número mínimo de trolleys necesarios según el peso óptimo total.
    Devuelve (num_trolleys, porcentaje_utilización_último_trolley)
    """
    peso_total_optimo = work["Peso_Optimo_kg"].sum()
    num_trolleys = int(np.ceil(peso_total_optimo / capacidad_kg))
    if num_trolleys == 0:
        return 0, 0.0
    # Porcentaje de uso del último trolley
    peso_restante = peso_total_optimo % capacidad_kg
    pct_ultimo = (peso_restante / capacidad_kg * 100) if peso_restante > 0 else 100.0
    return num_trolleys, pct_ultimo

# --------------------------------------------------------------
# SIDEBAR (carga y controles)
# --------------------------------------------------------------
with st.sidebar:
    st.header("🔧 Configuración")

    model, model_err, feature_cols_from_pkl = load_model(MODEL_PATH)
    if model_err:
        st.error(model_err)
        st.stop()

    # Si el pkl trae su orden de features, úsalo
    if feature_cols_from_pkl:
        FEATURE_COLS = feature_cols_from_pkl

    df_raw = load_dataset(CSV_PATH)
    df_base = preprocess_base(df_raw)
    df_enc, _ = fit_and_apply_encoders(df_base)

    if "Flight_ID" not in df_enc.columns:
        st.error("El CSV no contiene la columna 'Flight_ID'.")
        st.stop()

    flight_options = df_enc["Flight_ID"].astype(str).sort_values().unique().tolist()
    selected_flight = st.selectbox("✈️ Flight_ID", flight_options)

    buffer_pct_ui = st.slider("Buffer (%)", 5, 20, 10, 1,
                              help="Margen extra sobre la predicción para evitar desabasto.")
    buffer_fraction = buffer_pct_ui / 100.0

# --------------------------------------------------------------
# CONTEXTO DEL VUELO
# --------------------------------------------------------------
df_flight = df_enc[df_enc["Flight_ID"].astype(str) == str(selected_flight)].copy()
st.subheader("📋 Contexto del vuelo")

if df_flight.empty:
    st.warning("No se encontraron filas para el vuelo seleccionado.")
    st.stop()

row0 = df_flight.iloc[0]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Origin", str(row0.get("Origin", "N/A")))
c2.metric("Flight Type", str(row0.get("Flight_Type", "N/A")))
c3.metric("Service Type", str(row0.get("Service_Type", "N/A")))
c4.metric("Passengers", f"{int(row0.get('Passenger_Count', 0))}")
c5.metric("Date", str(row0.get("Date"))[:10])

# --------------------------------------------------------------
# PREDICCIÓN Y TABLA
# --------------------------------------------------------------
st.subheader("🔮 Predicción de consumo")
work = compute_recommendations(df_flight, model, buffer_fraction, FEATURE_COLS)

visible_cols = [
    "Product_Name", "Standard_Specification_Qty",
    "Predicted_Consumption", "Optimal_Specification", "Quantity_Consumed"
]
for col in visible_cols:
    if col not in work.columns:
        work[col] = np.nan

st.dataframe(work[visible_cols], use_container_width=True)

# --------------------------------------------------------------
# KPIs
# --------------------------------------------------------------
st.subheader("📉 Ahorros estimados")
peso_total_ahorrado, combustible_ahorrado, ahorro_dinero = summarize_impacts(work)
k1, k2, k3 = st.columns(3)
k1.metric("Peso ahorrado (kg)", f"{peso_total_ahorrado:,.2f}")
k2.metric("Combustible ahorrado (kg)", f"{combustible_ahorrado:,.2f}")
k3.metric("Ahorro económico (USD)", f"${ahorro_dinero:,.2f}")

# --------------------------------------------------------------
# TROLLEYS
# --------------------------------------------------------------
st.subheader("🧳 Configuración de trolleys")

num_trolleys, pct_ultimo = calcular_trolleys(work, TROLLEY_CAPACITY_KG)
peso_optimo_total = work["Peso_Optimo_kg"].sum()

if num_trolleys > 0:
    st.write(f"**Peso total óptimo:** {peso_optimo_total:,.2f} kg")
    st.write(f"**Capacidad por trolley:** {TROLLEY_CAPACITY_KG:.1f} kg")
    st.write(f"**Trolleys necesarios:** {num_trolleys}")
    st.progress(min(pct_ultimo / 100, 1.0), text=f"Último trolley al {pct_ultimo:.1f}% de su capacidad")
else:
    st.info("No hay peso óptimo calculado todavía.")

# --------------------------------------------------------------
# CHECKLIST + EXPORT
# --------------------------------------------------------------
st.subheader("✅ Checklist de empaque")
checklist_df = work[["Flight_ID", "Product_Name", "Optimal_Specification"]].copy().sort_values("Product_Name")

if checklist_df.empty:
    st.warning("No hay recomendaciones para mostrar.")
else:
    for _, r in checklist_df.iterrows():
        st.write(f"- Empaca **{int(r['Optimal_Specification'])}** de **{r['Product_Name']}**")

    st.download_button(
        label="💾 Exportar CSV — recomendacion_empaque.csv",
        data=checklist_df.to_csv(index=False).encode("utf-8"),
        file_name="recomendacion_empaque.csv",
        mime="text/csv"
    )

# --------------------------------------------------------------
# NOTAS
# --------------------------------------------------------------
st.caption("Si el .pkl trae {'model': ..., 'feature_cols': [...]}, se usa ese orden de features automáticamente.")
