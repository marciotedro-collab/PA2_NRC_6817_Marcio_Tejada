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

# Estilo personalizado para las alertas de la tabla interactiva
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
    
    uploaded_file = st.file_uploader("Sube tu archivo logistics_dataset.csv", type=["csv"])
    
    st.markdown("---")
    st.markdown("### 🧑‍💻 Información del Proyecto")
    st.markdown("**Estudiante:** Marcio Manuel Tejada Rodríguez")
    st.markdown("**Código/DNI:** 73047305")
    
    st.markdown("[🚀 Google Colab Notebook](https://colab.research.google.com/drive/1y6rVJfICT2AmW46ckP30ciqzgimKyuNl#scrollTo=yqHRZTCKDJq0)")
    st.markdown("[📁 Repositorio GitHub](https://github.com/marciotedro-collab/PA2_NRC_6817_Marcio_Tejada/blob/main/app.py)")
    st.markdown("---")

# =============================================================================
# LÓGICA DE CARGA Y MAPEADO EXACTO CON LOGISTICS_DATASET.CSV
# =============================================================================
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    
    # Mapeo directo y estricto basado en las columnas reales detectadas en tu archivo
    mapeo_columnas = {
        'item_id': 'ID_Producto',
        'category': 'Categoria',
        'zone': 'Zona',
        'stock_level': 'Stock_Actual',
        'reorder_point': 'Stock_Seguridad', # Usado temporalmente como base de resguardo
        'daily_demand': 'Demanda_Promedio_Diaria',
        'lead_time_days': 'Tiempo_Entrega_Dias',
        'turnover_ratio': 'Rotacion'
    }
    
    df = df.rename(columns=mapeo_columnas)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    
    # Si falta la columna Nombre_Producto, la clonamos del ID por estética
    if 'Nombre_Producto' not in df.columns:
        df['Nombre_Producto'] = df['ID_Producto']
        
    # Verificar y asegurar tipos numéricos limpios
    columnas_numericas = ['Stock_Actual', 'Stock_Seguridad', 'Demanda_Promedio_Diaria', 'Tiempo_Entrega_Dias', 'Rotacion']
    for col in columnas_numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.').str.strip(), errors='coerce').fillna(0)
        else:
            df[col] = 0.0
            
    return df

def generate_mock_data():
    np.random.seed(42)
    categorias = ["Pharma", "Automotive", "Groceries", "Apparel"]
    zonas = ["A", "B", "C", "D"]
    data = {
        "ID_Producto": [f"ITM100{i:02d}" for i in range(50)],
        "Nombre_Producto": [f"ITM100{i:02d}" for i in range(50)],
        "Categoria": np.random.choice(categorias, 50),
        "Zona": np.random.choice(zonas, 50),
        "Stock_Actual": np.random.randint(10, 450, 50),
        "Stock_Seguridad": np.random.randint(20, 80, 50),
        "Demanda_Promedio_Diaria": np.random.uniform(5, 50, 50).round(2),
        "Tiempo_Entrega_Dias": np.random.randint(3, 12, 50),
        "Rotacion": np.random.uniform(0.5, 12.0, 50).round(2)
    }
    return pd.DataFrame(data)

if uploaded_file is not None:
    df_raw = load_data(uploaded_file)
    st.sidebar.success("¡Dataset de Colab cargado con éxito!")
else:
    df_raw = generate_mock_data()
    st.sidebar.info("Mostrando estructura simulada compatible.")

# =============================================================================
# FILTROS
# =============================================================================
with st.sidebar:
    list_categorias = ["Todos"] + sorted(df_raw["Categoria"].dropna().unique().tolist())
    selected_cat = st.selectbox("Selecciona Categoría", list_categorias)
    
    list_zonas = ["Todos"] + sorted(df_raw["Zona"].dropna().unique().tolist())
    selected_zona = st.selectbox("Selecciona Zona", list_zonas)

df_filtered = df_raw.copy()
if selected_cat != "Todos":
    df_filtered = df_filtered[df_filtered["Categoria"] == selected_cat]
if selected_zona != "Todos":
    df_filtered = df_filtered[df_filtered["Zona"] == selected_zona]

# =============================================================================
# CUERPO PRINCIPAL
# =============================================================================
st.title("📦 Sistema de Prevención de Quiebres de Stock")
st.markdown("Herramienta analítica para el control de inventarios, predicción de desabastecimiento y cálculo de pedidos óptimos.")

# --- SECCIÓN 1: KPIs ---
st.subheader("📊 Indicadores Clave de Rendimiento")

quiebre_real = df_filtered[df_filtered["Stock_Actual"] == 0].shape[0]
# Un producto está en riesgo si el stock disponible es menor o igual al punto de reorden
riesgo_quiebre = df_filtered[df_filtered["Stock_Actual"] <= df_filtered["Stock_Seguridad"]].shape[0]
rotacion_promedio = df_filtered["Rotacion"].mean()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric(label="Total Items Filtrados", value=df_filtered.shape[0])
kpi2.metric(label="En Quiebre Absoluto (Stock 0)", value=quiebre_real, delta=f"{quiebre_real} SKUs", delta_color="inverse")
kpi3.metric(label="Por debajo de Alerta Reorder", value=riesgo_quiebre, delta="Acción requerida", delta_color="off")
kpi4.metric(label="Rotación de Inventario Promedio", value=f"{rotacion_promedio:.2f}x")

st.markdown("---")

# --- SECCIÓN 2: GRÁFICOS INTERACTIVOS ---
st.subheader("📈 Análisis de Distribución y Rotación")
col_graf1, col_graf2 = st.columns(2)

with col_graf1:
    st.markdown("#### Distribución de Rotación (Turnover Ratio) por Categoría")
    fig_rot = px.box(
        df_filtered, 
        x="Categoria", 
        y="Rotacion", 
        color="Categoria",
        points="all",
        labels={"Rotacion": "Turnover Ratio", "Categoria": "Categoría"}
    )
    st.plotly_chart(fig_rot, use_container_width=True)

with col_graf2:
    st.markdown("#### Top 10 Artículos Críticos (Menor Cobertura de Existencias)")
    df_filtered["Margen_Seguridad"] = df_filtered["Stock_Actual"] / (df_filtered["Stock_Seguridad"] + 1)
    top_quiebres = df_filtered.sort_values(by="Margen_Seguridad").head(10)
    
    fig_bar = px.bar(
        top_quiebres,
        x="Stock_Actual",
        y="ID_Producto",
        orientation='h',
        color="Zona",
        labels={"Stock_Actual": "Stock Actual (stock_level)", "ID_Producto": "Item ID"}
    )
    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# --- SECCIÓN 3: PREDICCIÓN TEMPRANA DE MACHINE LEARNING ---
st.subheader("🔮 Alertas Tempranas de Machine Learning")
st.markdown("Tabla analítica con clasificación automática del nivel de riesgo de desabastecimiento.")

def ejecutar_inferencia_ml(data):
    nombre_modelo = 'modelo_quiebre.pkl'
    
    # Cálculo real de días de cobertura basados en demanda diaria real
    data['Dias_Para_Quiebre'] = np.where(
        data['Demanda_Promedio_Diaria'] > 0,
        (data['Stock_Actual'] / data['Demanda_Promedio_Diaria']).round(1),
        999
    )
    
    if os.path.exists(nombre_modelo):
        try:
            model = joblib.load(nombre_modelo)
            # Asegura pasar el orden idéntico de features usado en el entrenamiento de Colab
            X = data[['Stock_Actual', 'Stock_Seguridad', 'Demanda_Promedio_Diaria', 'Tiempo_Entrega_Dias']]
            data['Riesgo_Predicho'] = model.predict(X)
        except Exception as e:
            st.warning(f"Aviso: Usando clasificador lógico de respaldo ({e})")
            condiciones = [
                (data['Stock_Actual'] == 0) | (data['Dias_Para_Quiebre'] <= data['Tiempo_Entrega_Dias']),
                (data['Stock_Actual'] <= data['Stock_Seguridad']),
                (data['Stock_Actual'] > data['Stock_Seguridad'])
            ]
            data['Riesgo_Predicho'] = np.select(condiciones, ["CRÍTICO", "ALTO", "NORMAL"], default="NORMAL")
    else:
        # Árbol lógico determinista de respaldo matemático exacto
        condiciones = [
            (data['Stock_Actual'] == 0) | (data['Dias_Para_Quiebre'] <= data['Tiempo_Entrega_Dias']),
            (data['Stock_Actual'] <= data['Stock_Seguridad']),
            (data['Stock_Actual'] > data['Stock_Seguridad'])
        ]
        data['Riesgo_Predicho'] = np.select(condiciones, ["CRÍTICO", "ALTO", "NORMAL"], default="NORMAL")
        
    return data

df_predicciones = ejecutar_inferencia_ml(df_filtered)

columnas_visibles = [
    "ID_Producto", "Categoria", "Zona", "Stock_Actual", 
    "Stock_Seguridad", "Demanda_Promedio_Diaria", "Dias_Para_Quiebre", "Riesgo_Predicho"
]

st.dataframe(
    df_predicciones[columnas_visibles].style.map(color_riesgo, subset=['Riesgo_Predicho']),
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

# --- SECCIÓN 4: SIMULADOR INTELIGENTE DE PEDIDOS ---
st.subheader("🤖 Simulador de Abastecimiento Óptimo")
st.markdown("Optimización matemática de reabastecimiento en base al Lead Time y Demanda.")

producto_seleccionado = st.selectbox(
    "Selecciona el Item ID para simular escenario:",
    df_filtered["ID_Producto"].unique()
)

if producto_seleccionado:
    row_prod = df_filtered[df_filtered["ID_Producto"] == producto_seleccionado].iloc[0]
    col_sim1, col_sim2 = st.columns(2)
    
    with col_sim1:
        st.markdown("#### 📥 Variables Logísticas del Item")
        stock_actual_input = st.number_input("Stock Actual en Almacén (stock_level)", value=int(row_prod["Stock_Actual"]), min_value=0)
        demanda_input = st.number_input("Demanda Promedio Diaria (daily_demand)", value=float(row_prod["Demanda_Promedio_Diaria"]), min_value=0.0, step=0.5)
        lead_time_input = st.number_input("Tiempo de Entrega (lead_time_days)", value=int(row_prod["Tiempo_Entrega_Dias"]), min_value=0)
        # El stock de seguridad dinámico sugerido para mitigar variabilidad durante el lead time
        stock_seg_input = st.number_input("Stock de Seguridad Base (reorder_point)", value=int(row_prod["Stock_Seguridad"]), min_value=0)

    with col_sim2:
        st.markdown("#### 🎯 Solución de Compra Sugerida")
        
        # El Punto de Reposición (ROP) estándar técnico
        punto_de_pedido = int((demanda_input * lead_time_input) + stock_seg_input)
        
        if stock_actual_input <= punto_de_pedido:
            # Orden de compra recomendada para cubrir 30 días de inventario cíclico
            cantidad_sugerida = int((demanda_input * 30) + stock_seg_input - stock_actual_input)
            if cantidad_sugerida < 0:
                cantidad_sugerida = 0
            necesita_pedido = "⚠️ SÍ - Generar Orden Urgente al Proveedor"
            color_alert = st.error
        else:
            cantidad_sugerida = 0
            necesita_pedido = "✅ NO - Cobertura de Stock Estable"
            color_alert = st.success
            
        st.metric(label="Punto de Reposición Calculado (ROP)", value=f"{punto_de_pedido} unidades")
        color_alert(f"**Estado de Alerta:** {necesita_pedido}")
        
        st.info(
            f"**Lógica de Control:** Este producto requiere mantener un mínimo de **{punto_de_pedido}** unidades "
            f"para no quedar desabastecido durante los {lead_time_input} días que tarda el proveedor. "
            f"Como cuentas con **{stock_actual_input}** unidades, la app ha calculado su estado de reaprovisionamiento."
        )
        st.subheader(f"Pedido Óptimo Recomendado: {cantidad_sugerida} uds.")
