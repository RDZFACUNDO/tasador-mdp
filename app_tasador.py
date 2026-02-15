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
    /* 1. FORZAR COLOR DE TEXTOS A VERDE */
    h2, h3, h4, strong, label, p {
        color: #1d6e5d !important;
    }
    
    /* 2. ARREGLO DE COCHERA (Checkbox) */
    /* Aseguramos que el texto se vea */
    label[data-baseweb="checkbox"] {
        color: #1d6e5d !important;
    }
    /* El cuadradito del checkbox */
    span[data-baseweb="checkbox"] div {
        background-color: #1d6e5d !important;
        border-color: #1d6e5d !important;
    }
    /* El tilde interno */
    span[data-baseweb="checkbox"] svg {
        fill: white !important;
    }

    /* 3. INPUT NUMÉRICO (El bloque negro de +/-) */
    /* Fondo del input completo */
    div[data-baseweb="input"] {
        background-color: #1d6e5d !important;
        border-color: #1d6e5d !important;
        border-radius: 5px;
        color: white !important;
    }
    /* El número escrito */
    input[data-baseweb="input"] {
        background-color: #1d6e5d !important;
        color: white !important;
    }
    /* Los botones de +/- (que eran negros) */
    div[data-baseweb="base-input"] button {
        background-color: #145244 !important; /* Un verde un poquito mas oscuro para diferenciar */
        color: white !important;
    }
    div[data-baseweb="base-input"] button svg {
        fill: white !important;
    }

    /* 4. MENÚS DESPLEGABLES (Selectbox) */
    div[data-baseweb="select"] > div {
        background-color: #1d6e5d !important;
        color: white !important;
        border-color: #1d6e5d !important;
    }
    div[data-baseweb="select"] span {
        color: white !important;
    }
    div[data-baseweb="select"] svg {
        fill: white !important;
    }
    
    /* Arreglo para que las opciones de la lista sean legibles (fondo blanco, letra negra) */
    ul[data-baseweb="menu"] {
        background-color: white !important;
    }
    ul[data-baseweb="menu"] li span {
        color: #333 !important;
    }

    /* 5. BOTÓN CALCULAR */
    .stButton>button {
        width: 100%;
        background-color: #1d6e5d;
        color: white !important;
        border: none;
        height: 3em;
        font-weight: bold;
        margin-top: 15px;
    }
    .stButton>button:hover {
        background-color: #145244;
    }

    /* 6. RESULTADOS */
    .resultado-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1d6e5d;
        margin-top: 10px;
    }
    /* Forzamos negro para el texto del resultado para que se lea bien sobre gris claro */
    .resultado-box h3, .resultado-box p, .resultado-box b {
        color: #333 !important;
    }

    /* Ajuste de márgenes */
    .block-container {
        padding-top: 2rem;
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

st.markdown("## 🏡 Tasador Inteligente: Mar del Plata")

col_mapa, col_datos = st.columns([3, 1.8], gap="large")

with col_mapa:
    c1, c2 = st.columns([1, 1])
    with c1:
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
    
    st.info("👆 Hacé clic en el mapa para ajustar la ubicación exacta antes de tasar.")


with col_datos:
    st.markdown("### Características")
    
    tipo = st.selectbox("Tipo de Propiedad", ["Departamentos", "Casas", "Ph", "Locales", "Oficinas"])
    
    # Input de metros y Checkbox cochera
    c_metros, c_cochera = st.columns([2, 1])
    with c_metros:
        metros = st.number_input("Metros (m²)", 20, 600, 60)
    with c_cochera:
        st.write("") 
        st.write("") 
        cochera = st.checkbox("Cochera") # Ahora debería ser visible

    ambientes = st.slider("Ambientes", 1, 6, 2)
    banos = st.slider("Baños", 1, 4, 1)

    st.markdown("---")

    if st.button("CALCULAR VALOR", use_container_width=True):
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
        
        st.markdown(f"""
        <div class="resultado-box">
            <h3 style="margin-bottom: 0px;">U$S {precio:,.0f}</h3>
            <p style="margin-bottom: 5px;">Precio Estimado de Mercado</p>
            <hr style="margin: 5px 0; border-top: 1px solid #ccc;">
            <p style="font-size: 14px; margin-bottom: 0;">Valor por m²: <b>U$S {m2:,.0f}</b></p>
        </div>
        """, unsafe_allow_html=True)