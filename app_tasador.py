import streamlit as st
import pandas as pd
import joblib
import numpy as np
import folium
from streamlit_folium import st_folium

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Tasador Inmobiliario MDP", page_icon="🏢", layout="wide")

# --- ESTILOS CSS ---
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        height: 3em;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CARGA DEL MODELO ---
@st.cache_resource
def cargar_modelo():
    try:
        artefactos = joblib.load('modeloML.pkl')
        return artefactos
    except FileNotFoundError:
        st.error("⚠️ No se encuentra el archivo del modelo.")
        st.stop()

artefactos = cargar_modelo()
modelo = artefactos['modelo_precio']
kmeans = artefactos['modelo_zonas']
cols_entrenamiento = artefactos['columnas']

# --- LÓGICA DE UBICACIÓN ---
# Inicializamos coordenadas por defecto (Centro de MDP) si no existen
if 'lat' not in st.session_state:
    st.session_state['lat'] = -38.0000
if 'lon' not in st.session_state:
    st.session_state['lon'] = -57.5500

st.title("🏡 Tasador Inteligente: Mar del Plata")

col_mapa, col_datos = st.columns([3, 2], gap="medium")

with col_mapa:
    st.subheader("1. Ubicación Exacta")
    st.caption("Hacé clic en el mapa para marcar la ubicación de la propiedad.")

    # Selector rápido de zonas (ayuda a navegar, pero el click manda)
    barrios = {
        "Seleccionar en Mapa": (None, None),
        "Playa Grande": (-38.0169, -57.5309),
        "Varese": (-38.0120, -57.5350),
        "Güemes": (-38.0122, -57.5388),
        "Centro": (-38.0055, -57.5427),
        "La Perla": (-37.9926, -57.5492),
        "Constitución": (-37.9754, -57.5583),
        "Puerto": (-38.0357, -57.5392),
        "Mogotes": (-38.0583, -57.5519)
    }
    
    zona_elegida = st.selectbox("Ir a zona (Opcional)", list(barrios.keys()))

    # Si el usuario elige una zona del menú, actualizamos el centro del mapa
    start_lat = st.session_state['lat']
    start_lon = st.session_state['lon']
    zoom_level = 14

    if zona_elegida != "Seleccionar en Mapa":
        nueva_lat, nueva_lon = barrios[zona_elegida]
        if nueva_lat: # Si no es None
            start_lat = nueva_lat
            start_lon = nueva_lon
            # Actualizamos el session state para que el marcador se mueva ahí también
            st.session_state['lat'] = nueva_lat
            st.session_state['lon'] = nueva_lon

    # CREACIÓN DEL MAPA INTERACTIVO
    m = folium.Map(location=[start_lat, start_lon], zoom_start=zoom_level)
    
    # Agregamos el marcador en la posición actual guardada
    folium.Marker(
        [st.session_state['lat'], st.session_state['lon']],
        popup="Ubicación Seleccionada",
        tooltip="Propiedad",
        icon=folium.Icon(color="red", icon="home")
    ).add_to(m)

    # Capturamos el evento de clic
    mapa_output = st_folium(m, height=500, use_container_width=True)

    # Si el usuario hace clic, actualizamos las coordenadas
    if mapa_output['last_clicked']:
        st.session_state['lat'] = mapa_output['last_clicked']['lat']
        st.session_state['lon'] = mapa_output['last_clicked']['lng']
        # Recargamos para que el marcador se mueva al nuevo clic
        if st.button("Confirmar Ubicación Marcada"):
             st.rerun()

    st.info(f"📍 Coordenadas detectadas: {st.session_state['lat']:.5f}, {st.session_state['lon']:.5f}")

with col_datos:
    st.subheader("2. Características")
    with st.container(border=True):
        tipo = st.selectbox("Tipo", ["Departamentos", "Casas", "Ph", "Locales", "Oficinas"])
        metros = st.number_input("Metros Totales (m²)", 20, 600, 60)
        ambientes = st.slider("Ambientes", 1, 6, 2)
        banos = st.slider("Baños", 1, 4, 1)
        cochera = st.checkbox("Tiene Cochera")

    st.markdown("###") # Espacio
    
    if st.button("CALCULAR VALOR"):
        
        # Armado del DataFrame
        input_data = pd.DataFrame(0, index=[0], columns=cols_entrenamiento)
        input_data['metros'] = metros
        input_data['lat'] = st.session_state['lat']
        input_data['lon'] = st.session_state['lon']
        input_data['ambientes'] = ambientes
        input_data['banos'] = banos
        input_data['cochera'] = 1 if cochera else 0
        
        # Cluster automático
        input_data['cluster_ubicacion'] = kmeans.predict([[st.session_state['lat'], st.session_state['lon']]])[0]
        
        # One-Hot Encoding
        col_tipo = f"tipo_{tipo}"
        if col_tipo in input_data.columns:
            input_data[col_tipo] = 1
            
        # Predicción
        precio = modelo.predict(input_data)[0]
        m2 = precio / metros
        
        st.success("¡Tasación Exitosa!")
        st.metric("Valor de Mercado Estimado", f"U$S {precio:,.0f}")
        st.metric("Valor por m²", f"U$S {m2:,.0f}")
        st.caption("Margen de error promedio del modelo: ± U$S 25.000")