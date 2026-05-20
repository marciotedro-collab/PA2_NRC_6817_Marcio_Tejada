"""
╔══════════════════════════════════════════════════════════════════════╗
║        STOCK ALERT PRO — Prevención de Quiebres de Stock            ║
║        Basado en: logistics_dataset.csv  (3,204 registros)          ║
╚══════════════════════════════════════════════════════════════════════╝

INSTALACIÓN:
    pip install streamlit pandas numpy plotly joblib scikit-learn openpyxl

CORRER:
    streamlit run stock_alert_app.py

MODELOS PREDICTIVOS:
    Coloca tus archivos en una carpeta 'models/' junto a este script:
        models/modelo_riesgo.pkl   ← clasificador (predice P(quiebre))
        models/modelo_demanda.pkl  ← regresor (predice demanda futura)
        models/scaler.pkl          ← StandardScaler u otro preprocesador
    Si no existen, la app usa el sistema de reglas de negocio como fallback.
"""

# ─────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Alert Pro",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# TEMA Y CSS
# ─────────────────────────────────────────────────────────────────────
COLORS = {
    "bg":        "#0f172a",
    "surface":   "#1e293b",
    "border":    "#334155",
    "muted":     "#64748b",
    "text":      "#e2e8f0",
    "subtext":   "#94a3b8",
    "accent":    "#38bdf8",
    "critico":   "#f87171",
    "alto":      "#fb923c",
    "medio":     "#fbbf24",
    "bajo":      "#4ade80",
    "info":      "#818cf8",
}

RISK_COLORS = {
    "Crítico": COLORS["critico"],
    "Alto":    COLORS["alto"],
    "Medio":   COLORS["medio"],
    "Bajo":    COLORS["bajo"],
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;400;600;700;800&display=swap');

*, html, body, [class*="css"] {{
    font-family: 'Sora', sans-serif !important;
}}

/* ── Fondo principal ── */
.stApp {{ background: {COLORS['bg']}; }}
.block-container {{ padding-top: 1.5rem !important; max-width: 1400px; }}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background: {COLORS['surface']};
    border-right: 1px solid {COLORS['border']};
}}
section[data-testid="stSidebar"] .block-container {{ padding-top: 1.5rem; }}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown {{ color: {COLORS['subtext']} !important; }}

/* ── Header ── */
.app-header {{
    background: linear-gradient(135deg, #0f172a 0%, #1a2f4a 60%, #0f172a 100%);
    border: 1px solid {COLORS['border']};
    border-radius: 14px;
    padding: 22px 32px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}}
.app-header::after {{
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(56,189,248,0.07) 0%, transparent 70%);
    border-radius: 50%;
}}
.app-header h1 {{
    color: #f1f5f9; font-size: 1.7rem; font-weight: 800;
    margin: 0; letter-spacing: -0.5px;
}}
.app-header p {{ color: {COLORS['subtext']}; margin: 5px 0 0 0; font-size: 0.85rem; }}
.badge-ml {{
    display: inline-block;
    background: rgba(56,189,248,0.15); color: {COLORS['accent']};
    border: 1px solid rgba(56,189,248,0.3);
    font-size: 0.62rem; font-weight: 700; padding: 2px 9px;
    border-radius: 20px; letter-spacing: 1.2px; text-transform: uppercase;
    margin-left: 10px; vertical-align: middle;
}}

/* ── Tarjetas de métricas ── */
.metric-card {{
    background: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 12px;
    padding: 20px 18px;
    text-align: center;
    transition: transform .2s, border-color .2s;
    height: 100%;
}}
.metric-card:hover {{ transform: translateY(-3px); border-color: {COLORS['muted']}; }}
.mv {{ font-size: 2rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; line-height: 1.1; }}
.ml {{ font-size: 0.7rem; color: {COLORS['subtext']}; text-transform: uppercase;
       letter-spacing: 1.2px; margin-top: 5px; }}
.md {{ font-size: 0.78rem; margin-top: 5px; font-weight: 600; }}

/* ── Colores de riesgo ── */
.c-critico {{ color: {COLORS['critico']}; }}
.c-alto    {{ color: {COLORS['alto']}; }}
.c-medio   {{ color: {COLORS['medio']}; }}
.c-bajo    {{ color: {COLORS['bajo']}; }}
.c-accent  {{ color: {COLORS['accent']}; }}
.c-info    {{ color: {COLORS['info']}; }}

/* ── Sección título ── */
.sec-title {{
    font-size: 0.72rem; font-weight: 700; color: {COLORS['subtext']};
    text-transform: uppercase; letter-spacing: 2px;
    border-bottom: 1px solid {COLORS['border']};
    padding-bottom: 8px; margin: 28px 0 16px 0;
}}

/* ── Tabla de alertas ── */
.alert-table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
.alert-table th {{
    background: {COLORS['bg']}; color: {COLORS['subtext']};
    font-size: 0.65rem; text-transform: uppercase; letter-spacing: 1.5px;
    padding: 10px 12px; text-align: left; border-bottom: 1px solid {COLORS['border']};
    font-weight: 600;
}}
.alert-table td {{ padding: 9px 12px; border-bottom: 1px solid rgba(51,65,85,0.5); color: {COLORS['text']}; }}
.row-critico {{ background: rgba(248,113,113,0.08); border-left: 3px solid {COLORS['critico']}; }}
.row-alto    {{ background: rgba(251,146,60,0.07);  border-left: 3px solid {COLORS['alto']}; }}
.row-medio   {{ background: rgba(251,191,36,0.06);  border-left: 3px solid {COLORS['medio']}; }}
.row-bajo    {{ background: rgba(74,222,128,0.05);  border-left: 3px solid {COLORS['bajo']}; }}

/* ── Chip de riesgo ── */
.chip {{
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.5px;
}}
.chip-critico {{ background: rgba(248,113,113,0.18); color: {COLORS['critico']}; border: 1px solid rgba(248,113,113,0.4); }}
.chip-alto    {{ background: rgba(251,146,60,0.18);  color: {COLORS['alto']};    border: 1px solid rgba(251,146,60,0.4); }}
.chip-medio   {{ background: rgba(251,191,36,0.15);  color: {COLORS['medio']};   border: 1px solid rgba(251,191,36,0.4); }}
.chip-bajo    {{ background: rgba(74,222,128,0.12);  color: {COLORS['bajo']};    border: 1px solid rgba(74,222,128,0.4); }}

/* ── Simulador ── */
.sim-box {{
    background: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 12px;
    padding: 22px 24px;
    margin-top: 4px;
}}
.sim-result {{
    background: linear-gradient(135deg, rgba(56,189,248,0.08) 0%, rgba(129,140,248,0.06) 100%);
    border: 1px solid rgba(56,189,248,0.25);
    border-radius: 10px;
    padding: 20px;
    margin-top: 16px;
    text-align: center;
}}
.sim-result .big-num {{
    font-size: 3rem; font-weight: 800; font-family: 'JetBrains Mono', monospace;
    color: {COLORS['accent']}; line-height: 1;
}}
.sim-result .label {{
    font-size: 0.72rem; color: {COLORS['subtext']};
    text-transform: uppercase; letter-spacing: 1.5px; margin-top: 6px;
}}

/* ── Inputs ── */
.stSelectbox > div > div,
.stMultiSelect > div > div {{
    background: {COLORS['bg']} !important;
    border: 1px solid {COLORS['border']} !important;
    border-radius: 8px !important;
    color: {COLORS['text']} !important;
}}
.stSlider > div > div > div {{ background: {COLORS['accent']} !important; }}
.stNumberInput > div > div > input {{
    background: {COLORS['bg']} !important;
    border: 1px solid {COLORS['border']} !important;
    color: {COLORS['text']} !important;
    border-radius: 8px !important;
}}

/* ── Botones ── */
.stButton > button {{
    background: {COLORS['accent']} !important;
    color: #0f172a !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.83rem !important;
    padding: 8px 22px !important;
    letter-spacing: 0.3px;
    transition: opacity .2s !important;
}}
.stButton > button:hover {{ opacity: 0.88 !important; }}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background: {COLORS['surface']};
    border-radius: 10px;
    padding: 4px;
    border: 1px solid {COLORS['border']};
    gap: 4px;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent !important;
    color: {COLORS['subtext']} !important;
    border-radius: 7px !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
}}
.stTabs [aria-selected="true"] {{
    background: {COLORS['bg']} !important;
    color: {COLORS['text']} !important;
}}

/* ── Info / warning boxes ── */
.stAlert {{ border-radius: 8px !important; }}

/* Ocultar footer */
footer {{ visibility: hidden; }}
#MainMenu {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# CONSTANTES Y CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────
CSV_PATH = "logistics_dataset.csv"   # ← nombre de tu archivo

# Features que usan los modelos (ajusta si tu modelo usa otras)
MODEL_FEATURES = [
    "stock_level", "reorder_point", "lead_time_days", "daily_demand",
    "demand_std_dev", "stockout_count_last_month", "turnover_ratio",
    "order_fulfillment_rate", "item_popularity_score", "forecasted_demand_next_7d",
]

# Factor de seguridad Z para nivel de servicio 95% → 1.65
Z_SERVICE_LEVEL = 1.65

# Umbrales de cobertura (días) para clasificar riesgo (si no hay modelo)
RISK_THRESHOLDS = {"Crítico": 3, "Alto": 7, "Medio": 14}

# ─────────────────────────────────────────────────────────────────────
# CARGA DE MODELOS PREDICTIVOS
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_ml_models():
    """
    Intenta cargar tus modelos entrenados desde la carpeta 'models/'.
    Si no existen, devuelve None y la app usa reglas de negocio.

    Guarda tus modelos así:
        import joblib
        joblib.dump(clf_riesgo,   "models/modelo_riesgo.pkl")   # clasificador
        joblib.dump(reg_demanda,  "models/modelo_demanda.pkl")  # regresor
        joblib.dump(scaler,       "models/scaler.pkl")          # preprocesador
    """
    models = {"riesgo": None, "demanda": None, "scaler": None, "cargados": []}
    try:
        import joblib
        if os.path.exists("models/modelo_riesgo.pkl"):
            models["riesgo"] = joblib.load("models/modelo_riesgo.pkl")
            models["cargados"].append("Clasificador de riesgo")
        if os.path.exists("models/modelo_demanda.pkl"):
            models["demanda"] = joblib.load("models/modelo_demanda.pkl")
            models["cargados"].append("Regresor de demanda")
        if os.path.exists("models/scaler.pkl"):
            models["scaler"] = joblib.load("models/scaler.pkl")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Error al cargar modelos: {e}")
    return models

# ─────────────────────────────────────────────────────────────────────
# CARGA Y CACHÉ DE DATOS
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data():
    if not os.path.exists(CSV_PATH):
        st.error(f"❌ No se encontró '{CSV_PATH}'. Colócalo en la misma carpeta que este script.")
        st.stop()
    df = pd.read_csv(CSV_PATH)
    df["last_restock_date"] = pd.to_datetime(df["last_restock_date"], errors="coerce")
    return df

# ─────────────────────────────────────────────────────────────────────
# CÁLCULO DE MÉTRICAS Y NIVEL DE RIESGO
# ─────────────────────────────────────────────────────────────────────
def enrich_data(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    df = df.copy()

    # 1. Días de cobertura actual
    df["dias_cobertura"] = (
        df["stock_level"] / df["daily_demand"].replace(0, np.nan)
    ).round(1)

    # 2. Stock de seguridad recomendado (fórmula EOQ con variabilidad de demanda)
    #    SS = Z × σ_demanda × √lead_time
    df["stock_seguridad_rec"] = (
        Z_SERVICE_LEVEL * df["demand_std_dev"] * np.sqrt(df["lead_time_days"])
    ).round(0).astype(int)

    # 3. Punto de reorden dinámico
    df["punto_reorden_calc"] = (
        df["daily_demand"] * df["lead_time_days"] + df["stock_seguridad_rec"]
    ).round(0).astype(int)

    # 4. Brecha respecto al punto de reorden (negativo = ya deberías haber pedido)
    df["brecha_stock"] = (df["stock_level"] - df["punto_reorden_calc"]).round(0)

    # 5. EOQ clásico: Q* = √(2·D_anual·S / H)
    #    Usamos handling_cost como costo de pedido (S) y holding_cost × 365 como H anual
    D_anual = df["daily_demand"] * 365
    S = df["handling_cost_per_unit"].replace(0, 1)
    H = (df["holding_cost_per_unit_day"] * 365).replace(0, 0.01)
    df["eoq"] = np.sqrt(2 * D_anual * S / H).round(0).astype(int)

    # 6. Estimación demanda próximos 7 días (usa modelo si existe)
    if models["demanda"] is not None:
        try:
            X = df[MODEL_FEATURES].fillna(0)
            if models["scaler"]:
                X = models["scaler"].transform(X)
            df["demanda_pred_7d"] = models["demanda"].predict(X).round(1)
        except Exception:
            df["demanda_pred_7d"] = df["forecasted_demand_next_7d"]
    else:
        df["demanda_pred_7d"] = df["forecasted_demand_next_7d"]

    # 7. Clasificación de riesgo (usa modelo si existe, si no usa reglas)
    if models["riesgo"] is not None:
        try:
            X = df[MODEL_FEATURES].fillna(0)
            if models["scaler"]:
                X_scaled = models["scaler"].transform(X)
            else:
                X_scaled = X
            proba = models["riesgo"].predict_proba(X_scaled)[:, 1]
            df["prob_quiebre_pct"] = (proba * 100).round(1)
            df["nivel_riesgo"] = pd.cut(
                df["prob_quiebre_pct"],
                bins=[-1, 25, 50, 75, 101],
                labels=["Bajo", "Medio", "Alto", "Crítico"],
            ).astype(str)
        except Exception as e:
            st.warning(f"Modelo de riesgo falló, usando reglas: {e}")
            df = _clasificar_por_reglas(df)
    else:
        df = _clasificar_por_reglas(df)

    return df

def _clasificar_por_reglas(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback: clasifica riesgo por días de cobertura y stockout_count."""
    cond_critico = (
        (df["dias_cobertura"] <= RISK_THRESHOLDS["Crítico"]) |
        (df["stock_level"] <= df["reorder_point"]) & (df["stockout_count_last_month"] >= 7)
    )
    cond_alto = (
        (df["dias_cobertura"] <= RISK_THRESHOLDS["Alto"]) |
        (df["stockout_count_last_month"] >= 5)
    ) & ~cond_critico
    cond_medio = (
        (df["dias_cobertura"] <= RISK_THRESHOLDS["Medio"]) |
        (df["stockout_count_last_month"] >= 2)
    ) & ~cond_critico & ~cond_alto

    df["nivel_riesgo"] = "Bajo"
    df.loc[cond_medio,   "nivel_riesgo"] = "Medio"
    df.loc[cond_alto,    "nivel_riesgo"] = "Alto"
    df.loc[cond_critico, "nivel_riesgo"] = "Crítico"

    # Probabilidad estimada por heurística para consistencia con la UI
    score = (
        (1 - df["order_fulfillment_rate"].clip(0, 1)) * 40 +
        (df["stockout_count_last_month"] / 9) * 35 +
        (1 - (df["dias_cobertura"].clip(0, 30) / 30)) * 25
    )
    df["prob_quiebre_pct"] = score.clip(0, 100).round(1)
    return df

# ─────────────────────────────────────────────────────────────────────
# SIMULADOR DE PEDIDO ÓPTIMO
# ─────────────────────────────────────────────────────────────────────
def calcular_pedido_optimo(
    stock_actual: float,
    demanda_diaria: float,
    std_demanda: float,
    lead_time: int,
    nivel_servicio_z: float = Z_SERVICE_LEVEL,
    costo_pedido: float = 50.0,
    costo_holding_dia: float = 1.0,
    usar_eoq: bool = True,
) -> dict:
    """
    Calcula el pedido óptimo usando EOQ + stock de seguridad.

    Returns dict con todos los componentes del cálculo.
    """
    # Stock de seguridad
    ss = nivel_servicio_z * std_demanda * np.sqrt(lead_time)

    # Demanda durante lead time
    demanda_lt = demanda_diaria * lead_time

    # Punto de reorden
    rop = demanda_lt + ss

    # EOQ (si aplica)
    if usar_eoq and costo_holding_dia > 0:
        D_anual = demanda_diaria * 365
        H_anual = costo_holding_dia * 365
        eoq = np.sqrt(2 * D_anual * costo_pedido / H_anual)
    else:
        # Si no usas EOQ, pedir para cubrir el ciclo de reabastecimiento
        eoq = demanda_diaria * 30  # 30 días de cobertura por defecto

    # Cantidad a pedir = max(EOQ, lo necesario para llegar al ROP)
    deficit = rop - stock_actual
    qty_pedir = max(eoq, deficit if deficit > 0 else eoq)

    # Costo total estimado del pedido
    costo_total = costo_pedido + (qty_pedir * costo_holding_dia * lead_time / 2)

    # Días de cobertura después del pedido (cuando llegue)
    stock_al_llegar = stock_actual - (demanda_diaria * lead_time)
    stock_tras_pedido = max(stock_al_llegar, 0) + qty_pedir
    cobertura_tras_pedido = stock_tras_pedido / demanda_diaria if demanda_diaria > 0 else 0

    return {
        "stock_seguridad": round(ss, 0),
        "demanda_lead_time": round(demanda_lt, 0),
        "punto_reorden": round(rop, 0),
        "eoq": round(eoq, 0),
        "cantidad_a_pedir": round(qty_pedir, 0),
        "costo_estimado": round(costo_total, 2),
        "cobertura_dias_post_pedido": round(cobertura_tras_pedido, 1),
        "stock_al_llegar_proveedor": round(max(stock_al_llegar, 0), 0),
        "deficit_detectado": deficit > 0,
    }

# ─────────────────────────────────────────────────────────────────────
# TEMA PLOTLY
# ─────────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Sora, sans-serif", color=COLORS["subtext"], size=11),
    title_font=dict(color=COLORS["text"], size=14, family="Sora"),
    legend=dict(
        bgcolor="rgba(30,41,59,0.8)",
        bordercolor=COLORS["border"],
        borderwidth=1,
        font=dict(color=COLORS["text"]),
    ),
    xaxis=dict(
        gridcolor=COLORS["border"], zerolinecolor=COLORS["border"],
        tickfont=dict(color=COLORS["subtext"]),
        title_font=dict(color=COLORS["subtext"]),
    ),
    yaxis=dict(
        gridcolor=COLORS["border"], zerolinecolor=COLORS["border"],
        tickfont=dict(color=COLORS["subtext"]),
        title_font=dict(color=COLORS["subtext"]),
    ),
    margin=dict(l=10, r=10, t=40, b=10),
)

# ─────────────────────────────────────────────────────────────────────
# APP PRINCIPAL
# ─────────────────────────────────────────────────────────────────────
def main():
    # ── Cargar datos y modelos ──────────────────────────────────────
    raw_df = load_data()
    models = load_ml_models()
    df_full = enrich_data(raw_df, models)

    # ── SIDEBAR ────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style='margin-bottom:20px;'>
            <div style='color:#e2e8f0;font-size:1rem;font-weight:700;'>⚙️ Filtros</div>
            <div style='color:#64748b;font-size:0.72rem;margin-top:2px;'>Ajusta la vista del dashboard</div>
        </div>
        """, unsafe_allow_html=True)

        # Categoría
        cats = ["Todas"] + sorted(df_full["category"].unique().tolist())
        sel_cat = st.selectbox("📦 Categoría", cats)

        # Zona
        zonas = ["Todas"] + sorted(df_full["zone"].unique().tolist())
        sel_zona = st.selectbox("🗺️ Zona de Almacén", zonas)

        # Nivel de riesgo
        risk_opts = ["Crítico", "Alto", "Medio", "Bajo"]
        sel_riesgo = st.multiselect(
            "🔴 Nivel de Riesgo",
            risk_opts,
            default=risk_opts,
        )

        st.markdown("---")
        st.markdown("""
        <div style='color:#64748b;font-size:0.72rem;text-transform:uppercase;
                    letter-spacing:1.5px;margin-bottom:8px;'>Estado de Modelos</div>
        """, unsafe_allow_html=True)

        if models["cargados"]:
            for m in models["cargados"]:
                st.markdown(f"✅ <span style='color:#4ade80;font-size:0.78rem;'>{m}</span>",
                            unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background:rgba(251,191,36,0.1);border:1px solid rgba(251,191,36,0.3);
                        border-radius:8px;padding:10px;font-size:0.75rem;color:#fbbf24;'>
            ⚠️ Modelos no encontrados.<br>
            <span style='color:#64748b;'>Usando reglas de negocio.<br>
            Coloca tus .pkl en models/</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(f"""
        <div style='color:#64748b;font-size:0.7rem;'>
            📊 {len(df_full):,} artículos · {df_full['category'].nunique()} categorías<br>
            🗓️ Datos actualizados al cargar
        </div>
        """, unsafe_allow_html=True)

    # ── FILTROS APLICADOS ─────────────────────────────────────────
    df = df_full.copy()
    if sel_cat != "Todas":
        df = df[df["category"] == sel_cat]
    if sel_zona != "Todas":
        df = df[df["zone"] == sel_zona]
    if sel_riesgo:
        df = df[df["nivel_riesgo"].isin(sel_riesgo)]

    if df.empty:
        st.warning("⚠️ Ningún registro coincide con los filtros seleccionados.")
        return

    # ── HEADER ───────────────────────────────────────────────────
    st.markdown(f"""
    <div class="app-header">
        <h1>📦 Stock Alert Pro
            <span class="badge-ml">{'ML ACTIVO' if models['cargados'] else 'REGLAS'}</span>
        </h1>
        <p>Sistema de prevención de quiebres de stock · {len(df):,} artículos filtrados
           de {len(df_full):,} totales</p>
    </div>
    """, unsafe_allow_html=True)

    # ── MÉTRICAS CLAVE ────────────────────────────────────────────
    n_critico  = (df["nivel_riesgo"] == "Crítico").sum()
    n_alto     = (df["nivel_riesgo"] == "Alto").sum()
    n_bajo_rep = (df["stock_level"] <= df["reorder_point"]).sum()
    avg_cob    = df["dias_cobertura"].median()
    total_stockouts = df["stockout_count_last_month"].sum()
    avg_fulfill = df["order_fulfillment_rate"].mean() * 100
    valor_riesgo = (
        df[df["nivel_riesgo"].isin(["Crítico", "Alto"])]["stock_level"] *
        df[df["nivel_riesgo"].isin(["Crítico", "Alto"])]["unit_price"]
    ).sum()

    m1, m2, m3, m4, m5, m6 = st.columns(6)

    metrics = [
        (m1, str(n_critico), "Artículos Críticos", "⚡ Requieren acción inmediata", "c-critico"),
        (m2, str(n_alto), "Riesgo Alto", "⚠️ Monitorear esta semana", "c-alto"),
        (m3, str(n_bajo_rep), "Bajo Punto Reorden", "📉 Stock debajo del ROP", "c-medio"),
        (m4, f"{avg_cob:.1f}d", "Cobertura Mediana", "📅 Días de stock disponible", "c-accent"),
        (m5, f"{avg_fulfill:.1f}%", "Fill Rate Promedio", "📊 Tasa de cumplimiento", "c-bajo"),
        (m6, f"S/{valor_riesgo:,.0f}", "Valor en Riesgo", "💰 Stock crítico + alto", "c-info"),
    ]

    for col, val, lbl, delta, cls in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="mv {cls}">{val}</div>
                <div class="ml">{lbl}</div>
                <div class="md" style="color:#64748b;">{delta}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── TABS PRINCIPALES ─────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Análisis Visual",
        "🚨 Alertas & Riesgo",
        "🤖 Simulador de Pedido",
        "📋 Datos Completos",
    ])

    # ════════════════════════════════════════════════════════════════
    # TAB 1: ANÁLISIS VISUAL
    # ════════════════════════════════════════════════════════════════
    with tab1:
        col_a, col_b = st.columns(2)

        # ── Gráfico: Rotación por categoría ──────────────────────
        with col_a:
            st.markdown('<div class="sec-title">Rotación de Inventario por Categoría</div>',
                        unsafe_allow_html=True)
            rot_cat = (
                df.groupby("category")["turnover_ratio"]
                .agg(["mean", "median", "std"])
                .reset_index()
                .rename(columns={"mean": "Promedio", "median": "Mediana", "std": "Desv. Std"})
                .sort_values("Promedio", ascending=True)
            )
            fig_rot = go.Figure()
            fig_rot.add_trace(go.Bar(
                y=rot_cat["category"],
                x=rot_cat["Promedio"],
                name="Rotación promedio",
                orientation="h",
                marker_color=COLORS["accent"],
                marker_opacity=0.85,
            ))
            fig_rot.add_trace(go.Scatter(
                y=rot_cat["category"],
                x=rot_cat["Mediana"],
                name="Mediana",
                mode="markers",
                marker=dict(color=COLORS["bajo"], size=10, symbol="diamond"),
            ))
            fig_rot.update_layout(
                **PLOTLY_LAYOUT,
                title="Rotación de Stock",
                height=320,
                barmode="group",
            )
            st.plotly_chart(fig_rot, use_container_width=True)

        # ── Gráfico: Quiebres por categoría ──────────────────────
        with col_b:
            st.markdown('<div class="sec-title">Quiebres de Stock por Categoría</div>',
                        unsafe_allow_html=True)
            qb_cat = (
                df.groupby("category")["stockout_count_last_month"]
                .agg(["sum", "mean"])
                .reset_index()
                .rename(columns={"sum": "Total Quiebres", "mean": "Promedio"})
                .sort_values("Total Quiebres", ascending=False)
            )
            fig_qb = go.Figure()
            fig_qb.add_trace(go.Bar(
                x=qb_cat["category"],
                y=qb_cat["Total Quiebres"],
                marker_color=[COLORS["critico"], COLORS["alto"], COLORS["medio"],
                               COLORS["bajo"], COLORS["accent"]][:len(qb_cat)],
                name="Total quiebres",
            ))
            fig_qb.update_layout(
                **PLOTLY_LAYOUT,
                title="Quiebres Último Mes",
                height=320,
            )
            st.plotly_chart(fig_qb, use_container_width=True)

        col_c, col_d = st.columns(2)

        # ── Gráfico: Distribución de riesgo ─────────────────────
        with col_c:
            st.markdown('<div class="sec-title">Distribución de Nivel de Riesgo</div>',
                        unsafe_allow_html=True)
            risk_dist = df["nivel_riesgo"].value_counts().reset_index()
            risk_dist.columns = ["Nivel", "Cantidad"]
            color_map = {"Crítico": COLORS["critico"], "Alto": COLORS["alto"],
                         "Medio": COLORS["medio"], "Bajo": COLORS["bajo"]}
            fig_risk = px.pie(
                risk_dist,
                names="Nivel",
                values="Cantidad",
                color="Nivel",
                color_discrete_map=color_map,
                hole=0.52,
            )
            fig_risk.update_traces(
                textposition="inside",
                textinfo="percent+label",
                textfont_size=11,
                marker=dict(line=dict(color=COLORS["bg"], width=2)),
            )
            fig_risk.update_layout(**PLOTLY_LAYOUT, height=320, title="Riesgo Global")
            st.plotly_chart(fig_risk, use_container_width=True)

        # ── Gráfico: Top 15 productos con más quiebres ──────────
        with col_d:
            st.markdown('<div class="sec-title">Top 15 Artículos con Más Quiebres</div>',
                        unsafe_allow_html=True)
            top_items = (
                df.nlargest(15, "stockout_count_last_month")[
                    ["item_id", "stockout_count_last_month", "nivel_riesgo", "category"]
                ].sort_values("stockout_count_last_month")
            )
            bar_colors = [color_map.get(r, COLORS["muted"]) for r in top_items["nivel_riesgo"]]
            fig_top = go.Figure(go.Bar(
                y=top_items["item_id"],
                x=top_items["stockout_count_last_month"],
                orientation="h",
                marker_color=bar_colors,
                text=top_items["category"],
                textposition="outside",
                textfont=dict(size=9, color=COLORS["subtext"]),
            ))
            fig_top.update_layout(
                **PLOTLY_LAYOUT,
                title="Quiebres Último Mes",
                height=380,
                showlegend=False,
            )
            st.plotly_chart(fig_top, use_container_width=True)

        # ── Gráfico: Stock vs Punto de Reorden (scatter) ─────────
        st.markdown('<div class="sec-title">Stock Actual vs Punto de Reorden</div>',
                    unsafe_allow_html=True)
        df_scatter = df.copy()
        df_scatter["color_risk"] = df_scatter["nivel_riesgo"].map(color_map)
        fig_sc = px.scatter(
            df_scatter.sample(min(len(df_scatter), 600), random_state=42),
            x="reorder_point",
            y="stock_level",
            color="nivel_riesgo",
            color_discrete_map=color_map,
            hover_data=["item_id", "category", "zone", "dias_cobertura"],
            opacity=0.75,
            size="daily_demand",
            size_max=18,
            labels={
                "reorder_point": "Punto de Reorden",
                "stock_level": "Stock Actual",
                "nivel_riesgo": "Riesgo",
            },
        )
        # Línea diagonal de referencia (stock == ROP)
        mx = max(df["reorder_point"].max(), df["stock_level"].max()) * 1.05
        fig_sc.add_trace(go.Scatter(
            x=[0, mx], y=[0, mx],
            mode="lines",
            line=dict(color=COLORS["critico"], dash="dash", width=1),
            name="Stock = ROP (zona crítica)",
        ))
        fig_sc.update_layout(
            **PLOTLY_LAYOUT,
            title="Artículos por debajo de la línea → necesitan reabastecimiento",
            height=380,
        )
        st.plotly_chart(fig_sc, use_container_width=True)

        # ── Gráfico: Heatmap quiebres por zona y categoría ───────
        st.markdown('<div class="sec-title">Mapa de Calor — Quiebres por Zona y Categoría</div>',
                    unsafe_allow_html=True)
        hm_data = (
            df.groupby(["zone", "category"])["stockout_count_last_month"]
            .mean().round(1)
            .reset_index()
            .pivot(index="zone", columns="category", values="stockout_count_last_month")
            .fillna(0)
        )
        fig_hm = go.Figure(go.Heatmap(
            z=hm_data.values,
            x=hm_data.columns.tolist(),
            y=[f"Zona {z}" for z in hm_data.index.tolist()],
            colorscale=[[0, "#1e293b"], [0.5, "#fb923c"], [1, "#f87171"]],
            text=hm_data.values.round(1),
            texttemplate="%{text}",
            textfont=dict(size=12, color="white"),
            showscale=True,
            colorbar=dict(title="Quiebres<br>Promedio", tickfont=dict(color=COLORS["subtext"])),
        ))
        fig_hm.update_layout(
            **PLOTLY_LAYOUT,
            title="Promedio de quiebres de stock por zona y categoría",
            height=280,
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    # ════════════════════════════════════════════════════════════════
    # TAB 2: ALERTAS Y RIESGO
    # ════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown('<div class="sec-title">Tabla de Alertas — Artículos en Riesgo</div>',
                    unsafe_allow_html=True)

        # Sub-filtros inline
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            solo_bajo_rop = st.checkbox("Solo artículos bajo punto de reorden", value=False)
        with c2:
            sort_col = st.selectbox(
                "Ordenar por",
                ["prob_quiebre_pct", "stockout_count_last_month", "dias_cobertura", "stock_level"],
                format_func=lambda x: {
                    "prob_quiebre_pct": "Probabilidad quiebre %",
                    "stockout_count_last_month": "Quiebres último mes",
                    "dias_cobertura": "Días de cobertura",
                    "stock_level": "Stock actual",
                }[x],
            )
        with c3:
            top_n = st.slider("Mostrar top N artículos", 10, 200, 50, step=10)

        df_alert = df.copy()
        if solo_bajo_rop:
            df_alert = df_alert[df_alert["stock_level"] <= df_alert["reorder_point"]]

        df_alert = df_alert.sort_values(
            sort_col,
            ascending=(sort_col == "dias_cobertura"),
        ).head(top_n)

        # Construir HTML de la tabla
        rows_html = ""
        for _, row in df_alert.iterrows():
            nivel   = row["nivel_riesgo"]
            cls_row = f"row-{nivel.lower()}"
            cls_chip = f"chip-{nivel.lower()}"
            cob_color = (
                COLORS["critico"] if row["dias_cobertura"] <= 3 else
                COLORS["alto"]    if row["dias_cobertura"] <= 7 else
                COLORS["medio"]   if row["dias_cobertura"] <= 14 else
                COLORS["bajo"]
            )
            rop_ok = "✅" if row["stock_level"] > row["reorder_point"] else "❌"
            rows_html += f"""
            <tr class="{cls_row}">
                <td><b>{row['item_id']}</b></td>
                <td>{row['category']}</td>
                <td>Zona {row['zone']}</td>
                <td style="font-family:monospace;color:{COLORS['accent']};">{int(row['stock_level']):,}</td>
                <td style="font-family:monospace;">{int(row['reorder_point']):,} {rop_ok}</td>
                <td style="color:{cob_color};font-weight:600;font-family:monospace;">
                    {row['dias_cobertura']:.1f}d
                </td>
                <td style="font-family:monospace;">{row['stockout_count_last_month']}</td>
                <td style="font-family:monospace;">{row['prob_quiebre_pct']:.1f}%</td>
                <td><span class="chip {cls_chip}">{nivel}</span></td>
            </tr>
            """

        table_html = f"""
        <div style="overflow-x:auto; background:{COLORS['surface']};
                    border:1px solid {COLORS['border']}; border-radius:10px; padding:4px;">
            <table class="alert-table">
                <thead>
                    <tr>
                        <th>Artículo</th>
                        <th>Categoría</th>
                        <th>Zona</th>
                        <th>Stock Actual</th>
                        <th>Punto Reorden</th>
                        <th>Cobertura</th>
                        <th>Quiebres/Mes</th>
                        <th>P(Quiebre)</th>
                        <th>Riesgo</th>
                    </tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        """
        st.markdown(table_html, unsafe_allow_html=True)

        # Resumen rápido
        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2, r3, r4 = st.columns(4)
        for col, nivel, color in [
            (r1, "Crítico", COLORS["critico"]),
            (r2, "Alto",    COLORS["alto"]),
            (r3, "Medio",   COLORS["medio"]),
            (r4, "Bajo",    COLORS["bajo"]),
        ]:
            cnt = (df_alert["nivel_riesgo"] == nivel).sum()
            with col:
                st.markdown(f"""
                <div class="metric-card" style="border-color:{color}22;">
                    <div class="mv" style="color:{color};">{cnt}</div>
                    <div class="ml">Artículos {nivel}</div>
                </div>
                """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════
    # TAB 3: SIMULADOR DE PEDIDO ÓPTIMO
    # ════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown('<div class="sec-title">Simulador Inteligente de Pedido al Proveedor</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div style='color:#64748b;font-size:0.82rem;margin-bottom:20px;'>
        Calcula la cantidad óptima de pedido usando la fórmula EOQ extendida con stock de seguridad
        dinámico. Puedes elegir un artículo del dataset o ingresar valores manuales.
        </div>
        """, unsafe_allow_html=True)

        col_sel, col_params = st.columns([1, 2])

        with col_sel:
            st.markdown("""
            <div style='font-size:0.78rem;color:#94a3b8;text-transform:uppercase;
                        letter-spacing:1px;margin-bottom:8px;'>Fuente de datos</div>
            """, unsafe_allow_html=True)
            modo = st.radio("", ["📦 Artículo del Dataset", "✏️ Valores Manuales"],
                            label_visibility="collapsed")

            if modo == "📦 Artículo del Dataset":
                # Filtrar por riesgo para que sea útil
                df_sim_pool = df.sort_values("prob_quiebre_pct", ascending=False)
                item_options = df_sim_pool["item_id"].tolist()
                sel_item = st.selectbox("Artículo", item_options)
                row_item = df_sim_pool[df_sim_pool["item_id"] == sel_item].iloc[0]

                sim_stock    = float(row_item["stock_level"])
                sim_demanda  = float(row_item["daily_demand"])
                sim_std      = float(row_item["demand_std_dev"])
                sim_lt       = int(row_item["lead_time_days"])
                sim_cp       = float(row_item["handling_cost_per_unit"])
                sim_ch       = float(row_item["holding_cost_per_unit_day"])
                sim_precio   = float(row_item["unit_price"])

                # Info del artículo
                nivel_item = row_item["nivel_riesgo"]
                chip_cls = f"chip chip-{nivel_item.lower()}"
                st.markdown(f"""
                <div style='background:{COLORS['bg']};border:1px solid {COLORS['border']};
                            border-radius:8px;padding:12px;margin-top:12px;font-size:0.8rem;'>
                    <div style='color:{COLORS['subtext']};margin-bottom:6px;'>Info del artículo</div>
                    <div>Categoría: <b style='color:{COLORS['text']};'>{row_item['category']}</b></div>
                    <div>Zona: <b style='color:{COLORS['text']};'>Zona {row_item['zone']}</b></div>
                    <div style='margin-top:6px;'>Riesgo: <span class="{chip_cls}">{nivel_item}</span></div>
                </div>
                """, unsafe_allow_html=True)

        with col_params:
            st.markdown('<div class="sim-box">', unsafe_allow_html=True)

            if modo == "✏️ Valores Manuales":
                # Valores por defecto si es manual
                sim_stock   = 100.0
                sim_demanda = 15.0
                sim_std     = 3.0
                sim_lt      = 7
                sim_cp      = 50.0
                sim_ch      = 1.0
                sim_precio  = 50.0

            c1, c2, c3 = st.columns(3)
            with c1:
                sim_stock   = st.number_input("Stock Actual (unidades)", value=sim_stock,   min_value=0.0, step=10.0)
                sim_demanda = st.number_input("Demanda Diaria (und/día)",  value=sim_demanda, min_value=0.1, step=1.0)
                sim_std     = st.number_input("Desv. Estándar Demanda",    value=sim_std,     min_value=0.0, step=0.5)
            with c2:
                sim_lt  = st.number_input("Lead Time (días)",          value=float(sim_lt), min_value=1.0, step=1.0)
                sim_cp  = st.number_input("Costo de Pedido (S/)",      value=sim_cp,        min_value=0.0, step=5.0)
                sim_ch  = st.number_input("Costo Holding (S//und/día)", value=sim_ch,       min_value=0.0, step=0.1)
            with c3:
                sim_precio = st.number_input("Precio Unitario (S/)",   value=sim_precio,    min_value=0.0, step=5.0)
                nivel_z = st.select_slider(
                    "Nivel de Servicio",
                    options=[1.28, 1.44, 1.65, 1.96, 2.33],
                    value=Z_SERVICE_LEVEL,
                    format_func=lambda z: {
                        1.28: "90%", 1.44: "92.5%", 1.65: "95%",
                        1.96: "97.5%", 2.33: "99%"
                    }[z],
                )
                usar_eoq = st.checkbox("Usar fórmula EOQ", value=True)

            st.markdown("</div>", unsafe_allow_html=True)

            if st.button("🧮 Calcular Pedido Óptimo", use_container_width=True):
                res = calcular_pedido_optimo(
                    stock_actual=sim_stock,
                    demanda_diaria=sim_demanda,
                    std_demanda=sim_std,
                    lead_time=int(sim_lt),
                    nivel_servicio_z=nivel_z,
                    costo_pedido=sim_cp,
                    costo_holding_dia=sim_ch,
                    usar_eoq=usar_eoq,
                )

                # Resultado principal
                st.markdown(f"""
                <div class="sim-result">
                    <div class="big-num">{int(res['cantidad_a_pedir']):,}</div>
                    <div class="label">Unidades a Pedir al Proveedor</div>
                </div>
                """, unsafe_allow_html=True)

                # Desglose
                st.markdown("<br>", unsafe_allow_html=True)
                d1, d2, d3, d4 = st.columns(4)
                desglose = [
                    (d1, int(res['stock_seguridad']),      "Stock de Seguridad",     COLORS["medio"]),
                    (d2, int(res['punto_reorden']),         "Punto de Reorden",       COLORS["alto"]),
                    (d3, int(res['eoq']),                   "EOQ Calculado",          COLORS["accent"]),
                    (d4, f"S/{res['costo_estimado']:,.0f}", "Costo Est. del Pedido",  COLORS["bajo"]),
                ]
                for col, val, lbl, clr in desglose:
                    with col:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="mv" style="color:{clr};font-size:1.5rem;">{val}</div>
                            <div class="ml">{lbl}</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                a1, a2 = st.columns(2)
                with a1:
                    st.markdown(f"""
                    <div style='background:{COLORS['surface']};border:1px solid {COLORS['border']};
                                border-radius:10px;padding:16px;font-size:0.82rem;'>
                        <div style='color:{COLORS['subtext']};text-transform:uppercase;
                                    font-size:0.7rem;letter-spacing:1px;margin-bottom:10px;'>
                            Análisis de la orden
                        </div>
                        <div>📦 Stock al llegar proveedor: <b style='color:{COLORS['text']};'>
                            {int(res['stock_al_llegar_proveedor'])} und</b></div>
                        <div>📅 Cobertura post-pedido: <b style='color:{COLORS['bajo']};'>
                            {res['cobertura_dias_post_pedido']:.1f} días</b></div>
                        <div>💰 Valor del pedido: <b style='color:{COLORS['accent']};'>
                            S/{int(res['cantidad_a_pedir']) * sim_precio:,.2f}</b></div>
                        <div>{'⚠️ <b style="color:' + COLORS["critico"] + ';">DÉFICIT DETECTADO — Pedir urgente</b>'
                              if res['deficit_detectado']
                              else '✅ <b style="color:' + COLORS["bajo"] + ';">Stock suficiente hasta reorden</b>'}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with a2:
                    # Mini gráfico de flujo de stock
                    dias = list(range(0, int(sim_lt) + 30))
                    stock_proj = [max(sim_stock - sim_demanda * d, 0) for d in dias]
                    stock_post = [
                        max(sim_stock - sim_demanda * d, 0) +
                        (res['cantidad_a_pedir'] if d >= sim_lt else 0)
                        for d in dias
                    ]
                    fig_proj = go.Figure()
                    fig_proj.add_trace(go.Scatter(
                        x=dias, y=stock_proj,
                        mode="lines", name="Sin pedido",
                        line=dict(color=COLORS["critico"], dash="dash", width=2),
                    ))
                    fig_proj.add_trace(go.Scatter(
                        x=dias, y=stock_post,
                        mode="lines", name="Con pedido",
                        line=dict(color=COLORS["bajo"], width=2),
                        fill="tozeroy",
                        fillcolor="rgba(74,222,128,0.07)",
                    ))
                    fig_proj.add_hline(
                        y=res["punto_reorden"],
                        line_dash="dot",
                        line_color=COLORS["alto"],
                        annotation_text="Punto Reorden",
                        annotation_font_color=COLORS["alto"],
                    )
                    fig_proj.add_vline(
                        x=sim_lt,
                        line_dash="dot",
                        line_color=COLORS["accent"],
                        annotation_text="Llegada",
                        annotation_font_color=COLORS["accent"],
                    )
                    fig_proj.update_layout(
                        **PLOTLY_LAYOUT,
                        title="Proyección de Stock (30 días)",
                        height=220,
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom", y=1.02,
                            font=dict(size=10),
                        ),
                        margin=dict(l=10, r=10, t=50, b=10),
                    )
                    st.plotly_chart(fig_proj, use_container_width=True)

    # ════════════════════════════════════════════════════════════════
    # TAB 4: DATOS COMPLETOS
    # ════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown('<div class="sec-title">Dataset Completo con Métricas Calculadas</div>',
                    unsafe_allow_html=True)

        cols_mostrar = [
            "item_id", "category", "zone", "stock_level", "reorder_point",
            "daily_demand", "lead_time_days", "dias_cobertura", "stock_seguridad_rec",
            "punto_reorden_calc", "stockout_count_last_month", "order_fulfillment_rate",
            "turnover_ratio", "prob_quiebre_pct", "nivel_riesgo", "unit_price",
        ]
        cols_disponibles = [c for c in cols_mostrar if c in df.columns]
        df_show = df[cols_disponibles].copy()

        # Renombrar para español
        df_show.columns = df_show.columns.str.replace("_", " ").str.title()

        st.dataframe(
            df_show,
            use_container_width=True,
            height=500,
        )

        col_dl1, col_dl2 = st.columns([1, 4])
        with col_dl1:
            csv_export = df[cols_disponibles].to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Exportar CSV",
                data=csv_export,
                file_name="stock_alert_export.csv",
                mime="text/csv",
            )

        st.markdown(f"""
        <div style='color:{COLORS['subtext']};font-size:0.75rem;margin-top:12px;'>
        Mostrando {len(df):,} artículos · {len(df.columns)} columnas ·
        {df['nivel_riesgo'].value_counts().to_dict()}
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()# Main application file
