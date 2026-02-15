import streamlit as st
import pandas as pd
import joblib
import numpy as np
import folium
from streamlit_folium import st_folium

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Tasador Inmobiliario MDP", page_icon="🏢", layout="wide")

# --- ESTILOS CSS PERSONALIZADOS ---
st.markdown("""
    <style>
    /* 1. Fondo blanco general */
    [data-testid="stAppViewContainer"] {
        background-color: #ffffff;
    }
    [data-testid="stHeader"] {
        background-color: #ffffff;
    }
    
    /* 2. Títulos y Textos principales en VERDE (#1d6e5d) */
    h2, h3 {
        color: #1d6e5d !important;
    }
    
    /* Etiquetas de los inputs (Selectbox, Slider, etc.) */
    .stSelectbox label, .stSlider label, .stNumberInput label, .stCheckbox label, .stRadio label {
        color: #1d6e5d !important;
        font-weight: bold;
    }
    
    /* 3. Personalización de Sliders (Slicers) al color verde */
    div[data-baseweb="slider"] > div > div > div > div {
        background-color: #1d6e5d !important;
    }
    div[data-baseweb="slider"] > div > div > div[role="slider"] {
        background-color: #1d6e5d !important;
        box-shadow: 0 0 5px rgba(29, 110, 93, 0.5);
    }
    
    /* 4. Botón de Calcular */
    .stButton>button {
        width: 100%;
        background-color: #1d6e5d; /* Tu color verde */
        color: white !important;
        height: 3em;
        border-radius: 8px;
        border: none;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #145244; /* Un verde un poco más oscuro para el efecto hover */
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        color: white !important;
    }

    /* Ajustes de espaciado */
    .block-container {
        padding-top: 2rem;
    }
    
    /* Caja de Resultados */
    .resultado-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1d6e5d; /* Borde verde también */
        margin-top: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    /* Color de textos generales dentro de la caja de resultado */
    .resultado-box h3, .resultado-box p, .resultado-box b {
        color: #212529 !important; /* Mantenemos gris oscuro para lectura, verde para el borde */
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

# Título Principal
st.markdown("## 🏡 Tasador Inteligente: Mar del Plata")

col_mapa, col_datos = st.columns([3, 1.8], gap="large")

with col_mapa:
    # Selector de estilo y zona
    c1, c2 = st.columns([1, 1])
    with c1:
        # Usamos label_visibility="collapsed" en el radio para que quede más limpio
        estilo_mapa = st.radio("Estilo de Mapa", ["Calles", "Claro"], horizontal=True, label_visibility="collapsed")
    with c2:
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

    # Movimiento del mapa
    start_lat = st.session_state['lat']
    start_lon = st.session_state['lon']
    
    if zona_elegida != "Centrar en...":
        nueva_lat, nueva_lon = barrios[zona_elegida]
        if nueva_lat:
            start_lat = nueva_lat
            start_lon = nueva_lon
            st.session_state['lat'] = nueva_lat
            st.session_state['lon'] = nueva_lon

    tile_layer = "CartoDB positron" if estilo_mapa == "Claro" else "OpenStreetMap"

    m = folium.Map(location=[start_lat, start_lon], zoom_start=14, tiles=tile_layer)
    
    # Marcador rojo (para que contraste bien con el verde de la app)
    folium.Marker(
        [st.session_state['lat'], st.session_state['lon']],
        popup="Propiedad",
        icon=folium.Icon(color="red", icon="home")
    ).add_to(m)

    mapa_output = st_folium(m, height=480, use_container_width=True)

    if mapa_output['last_clicked']:
        st.session_state['lat'] = mapa_output['last_clicked']['lat']
        st.session_state['lon'] = mapa_output['last_clicked']['lng']
        if st.button("📍 Confirmar ubicación", key="btn_confirm"):
             st.rerun()

with col_datos:
    st.markdown("### Características")
    
    tipo = st.selectbox("Tipo de Propiedad", ["Departamentos", "Casas", "Ph", "Locales", "Oficinas"])
    
    c_metros, c_cochera = st.columns([2, 1])
    with c_metros:
        metros = st.number_input("Metros (m²)", 20, 600, 60)
    with c_cochera:
        st.write("") 
        st.write("") 
        cochera = st.checkbox("Cochera")

    ambientes = st.slider("Ambientes", 1, 6, 2)
    banos = st.slider("Baños", 1, 4, 1)

    st.markdown("---")

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
        
        # --- RESULTADO ---
        st.markdown(f"""
        <div class="resultado-box">
            <h3 style="margin-bottom: 0px; color: #333 !important;">U$S {precio:,.0f}</h3>
            <p style="color: #666 !important; margin-bottom: 5px;">Precio Estimado de Mercado</p>
            <hr style="margin: 5px 0; border-top: 1px solid #ddd;">
            <p style="font-size: 14px; margin-bottom: 0; color: #666 !important;">Valor por m²: <b>U$S {m2:,.0f}</b></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("👆 Ajustá la ubicación y características para tazar.")