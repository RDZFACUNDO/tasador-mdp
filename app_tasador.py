import streamlit as st
import pandas as pd
import joblib
import numpy as np
import folium
from streamlit_folium import st_folium

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Tasador Inmobiliario MDP", page_icon="🏢", layout="wide")

# --- ESTILOS CSS (FONDO BLANCO Y AJUSTES) ---
st.markdown("""
    <style>
    /* Forzar fondo blanco en toda la app */
    [data-testid="stAppViewContainer"] {
        background-color: #ffffff;
    }
    [data-testid="stHeader"] {
        background-color: #ffffff;
    }
    
    /* Textos en color oscuro para contraste */
    h1, h2, h3, p, label, div, span {
        color: #212529 !important;
    }
    
    /* Botón rojo estilizado */
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white !important; /* Texto blanco forzado en botón */
        height: 3em;
        border-radius: 8px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #E04444;
        color: white !important;
    }

    /* Ajuste para subir el contenido de la derecha */
    .block-container {
        padding-top: 2rem; /* Menos espacio arriba del todo */
    }
    
    /* Recuadro de resultados */
    .resultado-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #FF4B4B;
        margin-top: 10px;
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
if 'lat' not in st.session_state:
    st.session_state['lat'] = -38.0000
if 'lon' not in st.session_state:
    st.session_state['lon'] = -57.5500

# Título más compacto
st.markdown("## 🏡 Tasador Inteligente: Mar del Plata")

col_mapa, col_datos = st.columns([3, 1.8], gap="large") # Ajusté la proporción para dar aire

with col_mapa:
    # Selector de estilo de mapa y zona en la misma línea visual
    c1, c2 = st.columns([1, 1])
    with c1:
        estilo_mapa = st.radio("Estilo de Mapa", ["Calles (Estándar)", "Claro (Minimalista)"], horizontal=True)
    with c2:
        # Selector de zonas simplificado
        barrios = {
            "Centrar en...": (None, None),
            "Playa Grande": (-38.0169, -57.5309),
            "Varese": (-38.0120, -57.5350),
            "Güemes": (-38.0122, -57.5388),
            "Centro": (-38.0055, -57.5427),
            "La Perla": (-37.9926, -57.5492),
            "Constitución": (-37.9754, -57.5583),
        }
        zona_elegida = st.selectbox("Ir a Zona", list(barrios.keys()), label_visibility="collapsed")

    # Lógica de movimiento del mapa
    start_lat = st.session_state['lat']
    start_lon = st.session_state['lon']
    
    if zona_elegida != "Centrar en...":
        nueva_lat, nueva_lon = barrios[zona_elegida]
        if nueva_lat:
            start_lat = nueva_lat
            start_lon = nueva_lon
            st.session_state['lat'] = nueva_lat
            st.session_state['lon'] = nueva_lon

    # Configuración del tile (capa) del mapa
    tile_layer = "CartoDB positron" if estilo_mapa == "Claro (Minimalista)" else "OpenStreetMap"

    m = folium.Map(location=[start_lat, start_lon], zoom_start=14, tiles=tile_layer)
    
    folium.Marker(
        [st.session_state['lat'], st.session_state['lon']],
        popup="Propiedad",
        icon=folium.Icon(color="red", icon="home")
    ).add_to(m)

    # Mapa un poco más alto para compensar el formulario
    mapa_output = st_folium(m, height=550, use_container_width=True)

    if mapa_output['last_clicked']:
        st.session_state['lat'] = mapa_output['last_clicked']['lat']
        st.session_state['lon'] = mapa_output['last_clicked']['lng']
        if st.button("📍 Confirmar nueva ubicación", key="btn_confirm"):
             st.rerun()

with col_datos:
    # Eliminamos el subheader grande para ganar espacio vertical
    st.markdown("#### Características")
    
    # Inputs más compactos
    tipo = st.selectbox("Tipo de Propiedad", ["Departamentos", "Casas", "Ph", "Locales", "Oficinas"])
    
    c_metros, c_cochera = st.columns([2, 1])
    with c_metros:
        metros = st.number_input("Metros (m²)", 20, 600, 60)
    with c_cochera:
        st.write("") # Espaciador para alinear checkbox
        st.write("") 
        cochera = st.checkbox("Cochera")

    ambientes = st.slider("Ambientes", 1, 6, 2)
    banos = st.slider("Baños", 1, 4, 1)

    st.markdown("---") # Separador sutil

    if st.button("CALCULAR VALOR", use_container_width=True):
        
        # --- CÁLCULO ---
        input_data = pd.DataFrame(0, index=[0], columns=cols_entrenamiento)
        input_data['metros'] = metros
        input_data['lat'] = st.session_state['lat']
        input_data['lon'] = st.session_state['lon']
        input_data['ambientes'] = ambientes
        input_data['banos'] = banos
        input_data['cochera'] = 1 if cochera else 0
        
        input_data['cluster_ubicacion'] = kmeans.predict([[st.session_state['lat'], st.session_state['lon']]])[0]
        
        col_tipo = f"tipo_{tipo}"
        if col_tipo in input_data.columns:
            input_data[col_tipo] = 1
            
        precio = modelo.predict(input_data)[0]
        m2 = precio / metros
        
        # --- RESULTADO COMPACTO ---
        # Usamos HTML para forzar el diseño exacto que querés al final del mapa
        st.markdown(f"""
        <div class="resultado-box">
            <h3 style="margin-bottom: 0px; color: #333;">U$S {precio:,.0f}</h3>
            <p style="color: #666; margin-bottom: 5px;">Precio Estimado de Mercado</p>
            <hr style="margin: 5px 0;">
            <p style="font-size: 14px; margin-bottom: 0;">Valor por m²: <b>U$S {m2:,.0f}</b></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Espacio vacío para mantener estructura si no hay cálculo
        st.info("👆 Ajustá la ubicación y características para tazar.")