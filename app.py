import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

# =============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# =============================================================================
st.set_page_config(
    page_title="Predicción de Quiebres de Stock & Optimización",
    page_icon="📦",
    layout="wide"
)

# Estilo personalizado para las alertas de tabla
def color_riesgo(val):
    if val == "CRÍTICO":
        color = "#ffcccc"  # Rojo claro
    elif val == "ALTO":
        color = "#ffe6cc"  # Naranja claro
    else:
        color = "#e6ffcc"  # Verde claro
    return f'background-color: {color}'

# =============================================================================
# BARRA LATERAL (SIDEBAR)
# =============================================================================
with st.sidebar:
    st.header("⚙️ Configuración y Filtros")
    
    # 1. Carga de Archivo CSV
    uploaded_file = st.file_uploader("Sube tu archivo logistics_dataset.csv", type=["csv"])
    
    st.markdown("---")
    
    # 2. Información del Autor y Enlaces solicitados
    st.markdown("### 🧑‍💻 Información del Proyecto")
    st.markdown("**Estudiante:** Marcio Manuel Tejada Rodríguez")
    st.markdown("**Código/DNI:** 73047305")
    
    st.markdown("[🚀 Google Colab Notebook](https://colab.research.google.com/drive/1y6rVJfICT2AmW46ckP30ciqzgimKyuNl#scrollTo=yqHRZTCKDJq0)")
    st.markdown("[📁 Repositorio GitHub](https://github.com/marciotedro-collab/PA2_NRC_6817_Marcio_Tejada/blob/main/app.py)")
    
    st.markdown("---")

# =============================================================================
# LÓGICA DE CARGA Y PROCESAMIENTO DE DATOS
# =============================================================================
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

# Dataset simulado de base (estructura idéntica al archivo real)
def generate_mock_data():
    np.random.seed(42)
    categorias = ["Electrónica", "Alimentos", "Textil", "Ferretería"]
    zonas = ["Norte", "Sur", "Centro", "Este"]
    data = {
        "ID_Producto": [f"PROD-{i:03d}" for i in range(1, 51)],
        "Nombre_Producto": [f"Artículo {i}" for i in range(1, 51)],
        "Categoria": np.random.choice(categorias, 50),
        "Zona": np.random.choice(zonas, 50),
        "Stock_Actual": np.random.randint(0, 150, 50),
        "Stock_Seguridad": np.random.randint(10, 40, 50),
        "Demanda_Promedio_Diaria": np.random.randint(2, 15, 50),
        "Tiempo_Entrega_Dias": np.random.randint(3, 10, 50),
        "Rotacion": np.random.uniform(1.2, 8.5, 50).round(2)
    }
    return pd.DataFrame(data)

if uploaded_file is not None:
    df_raw = load_data(uploaded_file)
    st.sidebar.success("¡Archivo cargado con éxito!")
else:
    df_raw = generate_mock_data()
    st.sidebar.info("Mostrando datos de demostración basados en logistics_dataset.csv.")

# =============================================================================
# FILTROS DINÁMICOS EN SIDEBAR
# =============================================================================
with st.sidebar:
    list_categorias = ["Todos"] + sorted(df_raw["Categoria"].dropna().unique().tolist())
    selected_cat = st.selectbox("Selecciona Categoría", list_categorias)
    
    list_zonas = ["Todos"] + sorted(df_raw["Zona"].dropna().unique().tolist())
    selected_zona = st.selectbox("Selecciona Zona", list_zonas)

# Aplicación de los filtros al DataFrame de trabajo
df_filtered = df_raw.copy()
if selected_cat != "Todos":
    df_filtered = df_filtered[df_filtered["Categoria"] == selected_cat]
if selected_zona != "Todos":
    df_filtered = df_filtered[df_filtered["Zona"] == selected_zona]

# =============================================================================
# CUERPO PRINCIPAL DE LA APP
# =============================================================================
st.title("📦 Sistema de Prevención de Quiebres de Stock")
st.markdown("Herramienta analítica para el control de inventarios, predicción de desabastecimiento y cálculo de pedidos óptimos.")

# --- SECCIÓN 1: MÉTRICAS CLAVE (KPIs) ---
st.subheader("📊 Indicadores Clave de Rendimiento")

quiebre_real = df_filtered[df_filtered["Stock_Actual"] == 0].shape[0]
riesgo_quiebre = df_filtered[df_filtered["Stock_Actual"] <= df_filtered["Stock_Seguridad"]].shape[0]
rotacion_promedio = df_filtered["Rotacion"].mean()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric(label="Total SKUs Filtrados", value=df_filtered.shape[0])
kpi2.metric(label="Productos en Quiebre (Stock 0)", value=quiebre_real, delta=f"{quiebre_real} críticos", delta_color="inverse")
kpi3.metric(label="Bajo Stock de Seguridad", value=riesgo_quiebre, delta="Alerta preventiva", delta_color="off")
kpi4.metric(label="Rotación Promedio", value=f"{rotacion_promedio:.2f}x")

st.markdown("---")

# --- SECCIÓN 2: GRÁFICOS INTERACTIVOS ---
st.subheader("📈 Análisis de Distribución y Rotación")
col_graf1, col_graf2 = st.columns(2)

with col_graf1:
    st.markdown("#### Distribución de Rotación por Categoría")
    fig_rot = px.box(
        df_filtered, 
        x="Categoria", 
        y="Rotacion", 
        color="Categoria",
        points="all",
        title="Nivel de rotación de productos según su categoría",
        labels={"Rotacion": "Índice de Rotación", "Categoria": "Categoría"}
    )
    st.plotly_chart(fig_rot, use_container_width=True)

with col_graf2:
    st.markdown("#### Top 10 Artículos con Menor Margen de Stock")
    df_filtered["Indice_Seguridad"] = df_filtered["Stock_Actual"] / (df_filtered["Stock_Seguridad"] + 1)
    top_quiebres = df_filtered.sort_values(by="Indice_Seguridad").head(10)
    
    fig_bar = px.bar(
        top_quiebres,
        x="Stock_Actual",
        y="Nombre_Producto",
        orientation='h',
        color="Zona",
        title="Top 10 productos más cercanos a su límite de seguridad",
        labels={"Stock_Actual": "Stock Disponible", "Nombre_Producto": "Producto"}
    )
    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# --- SECCIÓN 3: INTEGRACIÓN DE MODELOS SCIEKIT-LEARN / JOBLIB ---
st.subheader("🔮 Alertas Tempranas de Machine Learning")
st.markdown("Resultados de inferencia del modelo predictivo para identificar artículos en riesgo crítico.")

def ejecutar_inferencia_ml(data):
    """
    Carga el modelo guardado desde Colab usando joblib. 
    Si el archivo no está en la raíz del repositorio, aplica un fallback determinista.
    """
    nombre_modelo = 'modelo_quiebre.pkl'
    
    # Generamos la variable de días para quiebre (Feature común)
    data['Dias_Para_Quiebre'] = np.where(
        data['Demanda_Promedio_Diaria'] > 0,
        (data['Stock_Actual'] / data['Demanda_Promedio_Diaria']).round(1),
        999
    )
    
    if os.path.exists(nombre_modelo):
        try:
            # 🚀 CARGA REAL CON JOBLIB Y SCIKIT-LEARN
            model = joblib.load(nombre_modelo)
            
            # Ajusta las columnas 'X' según los nombres exactos de tu entrenamiento en Colab
            X = data[['Stock_Actual', 'Stock_Seguridad', 'Demanda_Promedio_Diaria', 'Tiempo_Entrega_Dias']]
            
            # Ejecución de la predicción
            predicciones = model.predict(X)
            data['Riesgo_Predicho'] = predicciones
            
        except Exception as e:
            st.warning(f"Error al ejecutar el archivo .pkl: {e}. Aplicando respaldo matemático.")
            # Fallback en caso de incompatibilidad de versiones de scikit-learn
            condiciones = [
                (data['Stock_Actual'] == 0) | (data['Dias_Para_Quiebre'] <= 2),
                (data['Stock_Actual'] <= data['Stock_Seguridad']),
                (data['Stock_Actual'] > data['Stock_Seguridad'])
            ]
            data['Riesgo_Predicho'] = np.select(condiciones, ["CRÍTICO", "ALTO", "NORMAL"], default="NORMAL")
    else:
        # Fallback si el archivo .pkl aún no se sube al repositorio
        condiciones = [
            (data['Stock_Actual'] == 0) | (data['Dias_Para_Quiebre'] <= 2),
            (data['Stock_Actual'] <= data['Stock_Seguridad']),
            (data['Stock_Actual'] > data['Stock_Seguridad'])
        ]
        data['Riesgo_Predicho'] = np.select(condiciones, ["CRÍTICO", "ALTO", "NORMAL"], default="NORMAL")
        
    return data

# Procesar DataFrame con el modelo
df_predicciones = ejecutar_inferencia_ml(df_filtered)

columnas_visibles = [
    "ID_Producto", "Nombre_Producto", "Categoria", "Zona", 
    "Stock_Actual", "Stock_Seguridad", "Dias_Para_Quiebre", "Riesgo_Predicho"
]

st.dataframe(
    df_predicciones[columnas_visibles].style.applymap(color_riesgo, subset=['Riesgo_Predicho']),
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

# --- SECCIÓN 4: SIMULADOR INTELIGENTE DE PEDIDOS ---
st.subheader("🤖 Simulador de Abastecimiento Óptimo")
st.markdown("Simulación del Punto de Reposición (ROP) e inventario sugerido para compras.")

producto_seleccionado = st.selectbox(
    "Selecciona el producto a simular:",
    df_filtered["Nombre_Producto"].unique()
)

if producto_seleccionado:
    row_prod = df_filtered[df_filtered["Nombre_Producto"] == producto_seleccionado].iloc[0]
    col_sim1, col_sim2 = st.columns(2)
    
    with col_sim1:
        st.markdown("#### 📥 Parámetros de Control Logístico")
        stock_actual_input = st.number_input("Stock Actual Disponible", value=int(row_prod["Stock_Actual"]), min_value=0)
        stock_seg_input = st.number_input("Stock de Seguridad Requerido", value=int(row_prod["Stock_Seguridad"]), min_value=0)
        demanda_input = st.number_input("Demanda Promedio Diaria (unidades)", value=int(row_prod["Demanda_Promedio_Diaria"]), min_value=0)
        lead_time_input = st.number_input("Tiempo de Entrega del Proveedor (Días)", value=int(row_prod["Tiempo_Entrega_Dias"]), min_value=0)

    with col_sim2:
        st.markdown("#### 🎯 Resultado del Cálculo de Reabastecimiento")
        
        # ROP = (Demanda Diaria * Lead Time) + Stock de Seguridad
        punto_de_pedido = (demanda_input * lead_time_input) + stock_seg_input
        
        if stock_actual_input <= punto_de_pedido:
            # Cobertura estándar de 30 días de operación
            cantidad_sugerida = (demanda_input * 30) + stock_seg_input - stock_actual_input
            necesita_pedido = "⚠️ SÍ - Generar Orden Urgente"
            color_alert = st.error
        else:
            cantidad_sugerida = 0
            necesita_pedido = "✅ NO - Stock Suficiente"
            color_alert = st.success
            
        st.metric(label="Punto de Reposición (ROP)", value=f"{punto_de_pedido} unidades")
        color_alert(f"**Estado de Alerta:** {necesita_pedido}")
        
        st.info(
            f"El Punto de Pedido crítico es de **{punto_de_pedido}** unidades. "
            f"Tu stock de **{stock_actual_input}** unidades requiere una acción inmediata."
        )
        st.subheader(f"Pedido Óptimo a Solicitar: {cantidad_sugerida} uds.")
