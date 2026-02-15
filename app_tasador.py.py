import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- CONFIGURACIÓN VISUAL ---
st.set_page_config(page_title="Tasador IA - MDP", page_icon="🏢", layout="centered")

# CSS para darle un toque más pro
st.markdown("""
    <style>
    .stButton>button {width: 100%; background-color: #FF4B4B; color: white;}
    .big-font {font-size:24px !important; font-weight: bold;}
    </style>
    """, unsafe_allow_html=True)

# --- 1. CARGA DEL CEREBRO ---
@st.cache_resource
def cargar_artefactos():
    # Asegurate que el nombre coincida con como lo guardaste
    try:
        ruta = "modeloML.pkl" # O la ruta completa si da error
        artefactos = joblib.load(ruta)
        return artefactos
    except FileNotFoundError:
        st.error(f"No encuentro el archivo '{ruta}'. Verificá la ruta.")
        return None

artefactos = cargar_artefactos()

if artefactos:
    modelo = artefactos['modelo_precio']
    kmeans = artefactos['modelo_zonas']
    columnas_modelo = artefactos['columnas']
else:
    st.stop()

# --- 2. INTERFAZ DE USUARIO ---
st.title("🤖 Tasador Inmobiliario IA")
st.markdown("##### Mar del Plata | Algoritmo Random Forest")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Ubicación")
    # Zonas predefinidas para facilitar la prueba
    barrios = {
        "Playa Grande (Costa)": (-38.0169, -57.5309),
        "Varese (Torreón)": (-38.0120, -57.5350),
        "Güemes (Comercial)": (-38.0122, -57.5388),
        "Centro (Casino)": (-38.0055, -57.5427),
        "La Perla (Plaza España)": (-37.9926, -57.5492),
        "Constitución": (-37.9754, -57.5583),
        "Puerto": (-38.0357, -57.5392),
        "Mogotes (Punta)": (-38.0583, -57.5519),
        "Manual (Elegir en mapa)": (None, None)
    }
    
    seleccion_barrio = st.selectbox("Seleccionar Zona Referencia", list(barrios.keys()))
    
    if seleccion_barrio == "Manual (Elegir en mapa)":
        st.info("Ajustá las coordenadas manualmente abajo 👇")
        lat_input = st.number_input("Latitud", value=-38.000, format="%.5f")
        lon_input = st.number_input("Longitud", value=-57.550, format="%.5f")
    else:
        lat_input, lon_input = barrios[seleccion_barrio]
        # Mostramos las coordenadas pero deshabilitadas si elige un preset
        st.text(f"Lat: {lat_input} | Lon: {lon_input}")

with col2:
    st.subheader("🏠 Características")
    tipo = st.selectbox("Tipo de Propiedad", ["Departamentos", "Casas", "Ph", "Oficinas", "Locales", "Terrenos"])
    metros = st.number_input("Metros Totales (m²)", min_value=15, max_value=600, value=60)
    ambientes = st.slider("Ambientes", 1, 6, 2)
    banos = st.slider("Baños", 1, 5, 1)
    cochera = st.checkbox("Incluye Cochera", value=False)

# Mapa para visualizar dónde estamos tasando
map_df = pd.DataFrame({'lat': [lat_input], 'lon': [lon_input]})
st.map(map_df, zoom=14)

# --- 3. LÓGICA DE PREDICCIÓN ---
if st.button("CALCULAR PRECIO DE MERCADO"):
    
    # A) Armar el DataFrame vacío con las columnas EXACTAS del entrenamiento
    input_data = pd.DataFrame(0, index=[0], columns=columnas_modelo)
    
    # B) Llenar datos básicos
    input_data['metros'] = metros
    input_data['lat'] = lat_input
    input_data['lon'] = lon_input
    input_data['ambientes'] = ambientes
    input_data['banos'] = banos
    input_data['cochera'] = 1 if cochera else 0
    
    # C) Calcular el Cluster Automáticamente
    # Aquí está la magia: Usamos el KMeans guardado para saber qué zona es
    cluster_predicho = kmeans.predict([[lat_input, lon_input]])[0]
    input_data['cluster_ubicacion'] = cluster_predicho
    
    # D) One-Hot Encoding del Tipo
    col_tipo = f"tipo_{tipo}"
    if col_tipo in input_data.columns:
        input_data[col_tipo] = 1
    
    # E) Predicción final
    try:
        precio_estimado = modelo.predict(input_data)[0]
        
        st.markdown("---")
        res_col1, res_col2, res_col3 = st.columns(3)
        
        with res_col1:
            st.metric("Precio Estimado", f"U$S {precio_estimado:,.0f}")
        with res_col2:
            m2_val = precio_estimado / metros
            st.metric("Valor m²", f"U$S {m2_val:,.0f}")
        with res_col3:
            st.markdown(f"**Zona Detectada:**\nCluster #{cluster_predicho}")
            
        # Un mensaje de contexto según el valor
        if precio_estimado > 200000:
            st.success("💎 Propiedad de alto valor (Premium)")
        elif precio_estimado < 50000:
            st.warning("📉 Oportunidad / Entrada de gama")
            
    except Exception as e:
        st.error(f"Error en el cálculo: {e}")