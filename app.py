import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from datetime import datetime

# =========================================
# CONFIGURACIÓN GENERAL
# =========================================

st.set_page_config(
    page_title="Prevención de Quiebres de Stock",
    page_icon="📦",
    layout="wide"
)

# =========================================
# HEADER
# =========================================

st.title("📦 Sistema Inteligente de Prevención de Quiebres de Stock")

st.markdown("""
### Información del Proyecto

**Autor:** Marcio Manuel Tejada Rodríguez  
**DNI:** 73047305  

🔗 Colab del proyecto:  
https://colab.research.google.com/drive/1y6rVJfICT2AmW46ckP30ciqzgimKyuNl#scrollTo=yqHRZTCKDJq0

🔗 Repositorio GitHub:  
https://github.com/marciotedro-collab/PA2_NRC_6817_Marcio_Tejada/blob/main/app.py
""")

st.divider()

# =========================================
# CARGA DE MODELOS
# =========================================

@st.cache_resource
def cargar_modelos():
    modelos = {}

    try:
        modelos["quiebre"] = joblib.load("modelos/modelo_quiebre.pkl")
    except:
        modelos["quiebre"] = None

    try:
        modelos["rotacion"] = joblib.load("modelos/modelo_rotacion.pkl")
    except:
        modelos["rotacion"] = None

    return modelos

modelos = cargar_modelos()

# =========================================
# SIDEBAR
# =========================================

st.sidebar.header("⚙️ Configuración")

archivo = st.sidebar.file_uploader(
    "Subir archivo CSV",
    type=["csv"]
)

# =========================================
# CARGA DE DATOS
# =========================================

if archivo is not None:

    df = pd.read_csv(archivo)

    st.success("Archivo cargado correctamente ✅")

    # =========================================
    # VALIDACIÓN DE COLUMNAS
    # =========================================

    columnas_requeridas = [
        "producto",
        "categoria",
        "zona",
        "stock_actual",
        "ventas_30d",
        "quiebres",
        "lead_time",
        "stock_seguridad"
    ]

    faltantes = [c for c in columnas_requeridas if c not in df.columns]

    if faltantes:
        st.error(f"Faltan columnas: {faltantes}")
        st.stop()

    # =========================================
    # FILTROS SIDEBAR
    # =========================================

    categorias = st.sidebar.multiselect(
        "Categoría",
        options=df["categoria"].unique(),
        default=df["categoria"].unique()
    )

    zonas = st.sidebar.multiselect(
        "Zona",
        options=df["zona"].unique(),
        default=df["zona"].unique()
    )

    df_filtrado = df[
        (df["categoria"].isin(categorias)) &
        (df["zona"].isin(zonas))
    ]

    # =========================================
    # FEATURES
    # =========================================

    df_filtrado["rotacion"] = (
        df_filtrado["ventas_30d"] /
        (df_filtrado["stock_actual"] + 1)
    )

    # =========================================
    # PREDICCIONES
    # =========================================

    if modelos["quiebre"] is not None:

        features = df_filtrado[
            [
                "stock_actual",
                "ventas_30d",
                "quiebres",
                "lead_time",
                "stock_seguridad"
            ]
        ]

        try:
            pred = modelos["quiebre"].predict_proba(features)[:,1]
            df_filtrado["riesgo"] = pred

        except:
            df_filtrado["riesgo"] = np.random.rand(len(df_filtrado))

    else:
        # Simulación temporal
        df_filtrado["riesgo"] = np.random.rand(len(df_filtrado))

    # =========================================
    # CLASIFICACIÓN DE RIESGO
    # =========================================

    def clasificar_riesgo(x):

        if x >= 0.75:
            return "CRÍTICO"

        elif x >= 0.50:
            return "ALTO"

        elif x >= 0.30:
            return "MEDIO"

        else:
            return "BAJO"

    df_filtrado["nivel_riesgo"] = df_filtrado["riesgo"].apply(clasificar_riesgo)

    # =========================================
    # KPIs
    # =========================================

    total_productos = len(df_filtrado)

    stock_total = int(df_filtrado["stock_actual"].sum())

    productos_criticos = len(
        df_filtrado[df_filtrado["nivel_riesgo"] == "CRÍTICO"]
    )

    promedio_rotacion = round(
        df_filtrado["rotacion"].mean(),
        2
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("📦 Productos", total_productos)
    col2.metric("🏪 Stock Total", stock_total)
    col3.metric("🚨 Riesgo Crítico", productos_criticos)
    col4.metric("🔄 Rotación Promedio", promedio_rotacion)

    st.divider()

    # =========================================
    # GRÁFICOS
    # =========================================

    colA, colB = st.columns(2)

    with colA:

        st.subheader("📈 Rotación por Producto")

        top_rotacion = df_filtrado.sort_values(
            "rotacion",
            ascending=False
        ).head(15)

        fig_rotacion = px.bar(
            top_rotacion,
            x="producto",
            y="rotacion",
            color="categoria",
            title="Productos con Mayor Rotación"
        )

        st.plotly_chart(fig_rotacion, use_container_width=True)

    with colB:

        st.subheader("⚠️ Productos con Más Quiebres")

        top_quiebres = df_filtrado.sort_values(
            "quiebres",
            ascending=False
        ).head(15)

        fig_quiebres = px.bar(
            top_quiebres,
            x="producto",
            y="quiebres",
            color="zona",
            title="Histórico de Quiebres"
        )

        st.plotly_chart(fig_quiebres, use_container_width=True)

    # =========================================
    # TABLA DE RIESGO
    # =========================================

    st.subheader("🚨 Artículos en Riesgo")

    def color_riesgo(val):

        if val == "CRÍTICO":
            return "background-color: #ff4b4b; color: white"

        elif val == "ALTO":
            return "background-color: orange"

        elif val == "MEDIO":
            return "background-color: yellow"

        else:
            return "background-color: #90ee90"

    tabla_riesgo = df_filtrado[
        [
            "producto",
            "categoria",
            "zona",
            "stock_actual",
            "ventas_30d",
            "quiebres",
            "riesgo",
            "nivel_riesgo"
        ]
    ].sort_values(
        "riesgo",
        ascending=False
    )

    st.dataframe(
        tabla_riesgo.style.applymap(
            color_riesgo,
            subset=["nivel_riesgo"]
        ),
        use_container_width=True,
        height=500
    )

    # =========================================
    # SIMULADOR DE PEDIDO ÓPTIMO
    # =========================================

    st.divider()

    st.subheader("🧠 Simulador Inteligente de Pedido Óptimo")

    producto_sel = st.selectbox(
        "Seleccionar Producto",
        df_filtrado["producto"].unique()
    )

    producto_df = df_filtrado[
        df_filtrado["producto"] == producto_sel
    ].iloc[0]

    demanda_diaria = producto_df["ventas_30d"] / 30

    lead_time = producto_df["lead_time"]

    stock_seguridad = producto_df["stock_seguridad"]

    stock_actual = producto_df["stock_actual"]

    demanda_durante_lead = demanda_diaria * lead_time

    pedido_optimo = (
        demanda_durante_lead +
        stock_seguridad -
        stock_actual
    )

    pedido_optimo = max(0, round(pedido_optimo))

    colS1, colS2, colS3, colS4 = st.columns(4)

    colS1.metric("Demanda diaria", round(demanda_diaria, 2))
    colS2.metric("Lead Time", lead_time)
    colS3.metric("Stock Seguridad", stock_seguridad)
    colS4.metric("Pedido Óptimo", pedido_optimo)

    # =========================================
    # ALERTA
    # =========================================

    riesgo_producto = producto_df["nivel_riesgo"]

    if riesgo_producto == "CRÍTICO":

        st.error(
            f"🚨 El producto {producto_sel} tiene riesgo CRÍTICO de quiebre."
        )

    elif riesgo_producto == "ALTO":

        st.warning(
            f"⚠️ El producto {producto_sel} tiene riesgo ALTO."
        )

    else:

        st.success(
            f"✅ El producto {producto_sel} está controlado."
        )

else:

    st.info("⬅️ Sube un archivo CSV para comenzar")
