import streamlit as st
import pandas as pd
import joblib
import numpy as np
import folium
from streamlit_folium import st_folium

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Tasador Inmobiliario MDP", page_icon="🏢", layout="wide")

# --- ESTILOS CSS (Mantenemos tu estilo verde #1d6e5d) ---
st.markdown("""
    <style>
    /* 1. TEXTOS GENERALES EN VERDE */
    h1, h2, h3, h4, h5, h6, strong, label {
        color: #1d6e5d !important;
    }
    p:not(.stButton p) {
        color: #1d6e5d !important;
    }
    
    /* 2. BOTÓN CALCULAR */
    div[data-testid="stButton"] button {
        width: 100%;
        background-color: #1d6e5d !important;
        border: none !important;
        height: 3em;
        margin-top: 15px;
    }
    div[data-testid="stButton"] button p {
        color: white !important;
        font-weight: bold !important;
        font-size: 16px !important;
    }
    div[data-testid="stButton"] button:hover {
        background-color: #145244 !important;
        color: white !important;
    }

    /* 3. CHECKBOX (Cochera) */
    label[data-baseweb="checkbox"] p { color: #1d6e5d !important; }
    span[data-baseweb="checkbox"][aria-checked="true"] div:first-child {
        background-color: #1d6e5d !important;
        border-color: #1d6e5d !important;
    }
    span[data-baseweb="checkbox"][aria-checked="false"] div:first-child {
        background-color: #ffffff !important;
        border-color: #1d6e5d !important;
    }
    span[data-baseweb="checkbox"] svg { fill: white !important; }

    /* 4. INPUT DE NÚMERO (Metros) */
    div[data-baseweb="input"] {
        background-color: #1d6e5d !important;
        border-color: #1d6e5d !important;
        border-radius: 5px;
    }
    input[data-baseweb="input"] {
        background-color: #1d6e5d !important;
        color: white !important; 
    }
    div[data-baseweb="base-input"] button {
        background-color: #145244 !important;
        color: white !important;
    }
    div[data-baseweb="base-input"] button svg { fill: white !important; }

    /* 5. SLIDERS */
    div[data-baseweb="slider"] div[role="slider"] {
        background-color: #1d6e5d !important;
        box-shadow: none !important;
    }
    div[data-baseweb="slider"] div[style*="background-color: rgb(255, 75, 75)"] {
        background-color: #1d6e5d !important;
    }
    div[data-baseweb="slider"] > div > div > div > div {
        background-color: #1d6e5d !important;
    }
    div[data-testid="stSliderTickBar"] + div { color: #1d6e5d !important; }

    /* 6. SELECTOR DE MAPA (Radio) */
    div[data-testid="stRadio"] label p { color: #1d6e5d !important; font-weight: bold; }
    div[data-baseweb="radio"] [aria-checked="true"] > div:first-child {
        background-color: #1d6e5d !important;
        border-color: #1d6e5d !important;
    }
    div[data-baseweb="radio"] > div:first-child { border-color: #1d6e5d !important; }

    /* 7. MENÚS DESPLEGABLES */
    div[data-baseweb="select"] > div {
        background-color: #1d6e5d !important;
        color: white !important;
        border-color: #1d6e5d !important;
    }
    div[data-baseweb="select"] span { color: white !important; }
    div[data-baseweb="select"] svg { fill: white !important; }
    ul[data-baseweb="menu"] { background-color: white !important; }
    ul[data-baseweb="menu"] li span { color: #333 !important; }

    /* 8. RESULTADOS */
    .resultado-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1d6e5d;
        margin-top: 10px;
    }
    .resultado-box h3, .resultado-box p, .resultado-box b {
        color: #333 !important; 
    }

    .block-container { padding-top: 2rem; }
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

# --- INICIALIZACIÓN DE VARIABLES DE ESTADO ---
if 'lat' not in st.session_state:
    st.session_state['lat'] = -38.0000
if 'lon' not in st.session_state:
    st.session_state['lon'] = -57.5500
# Variables para guardar el resultado y que no desaparezca
if 'precio_calculado' not in st.session_state:
    st.session_state['precio_calculado'] = None
if 'm2_calculado' not in st.session_state:
    st.session_state['m2_calculado'] = None

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

    # Si cambia el selector de zona, actualizamos
    if zona_elegida != "Centrar en...":
        nueva_lat, nueva_lon = barrios[zona_elegida]
        if nueva_lat:
             # Solo actualizamos si es diferente para no bloquear el movimiento manual
             if nueva_lat != st.session_state['lat'] or nueva_lon != st.session_state['lon']:
                st.session_state['lat'] = nueva_lat
                st.session_state['lon'] = nueva_lon
                st.rerun()

    tile_layer = "CartoDB positron" if estilo_mapa == "Claro" else "OpenStreetMap"

    m = folium.Map(location=[st.session_state['lat'], st.session_state['lon']], zoom_start=14, tiles=tile_layer)
    
    folium.Marker(
        [st.session_state['lat'], st.session_state['lon']],
        popup="Propiedad",
        icon=folium.Icon(color="red", icon="home")
    ).add_to(m)

    # El mapa devuelve datos cada vez que interactúas
    mapa_output = st_folium(m, height=480, use_container_width=True)

    # --- LÓGICA DE CLIC AUTOMÁTICO (Sin botón Confirmar) ---
    if mapa_output['last_clicked']:
        click_lat = mapa_output['last_clicked']['lat']
        click_lon = mapa_output['last_clicked']['lng']
        
        # Si el clic es diferente a lo que ya tenemos guardado, actualizamos y recargamos
        # Usamos una pequeña tolerancia para evitar recargas infinitas por decimales
        if abs(click_lat - st.session_state['lat']) > 0.0001 or abs(click_lon - st.session_state['lon']) > 0.0001:
            st.session_state['lat'] = click_lat
            st.session_state['lon'] = click_lon
            st.rerun() # Esto recarga la página solo para actualizar el marcador rojo
    
    st.info("👆 Hacé clic en el mapa para ajustar la ubicación exacta antes de tasar.")


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

    # BOTÓN DE CÁLCULO
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
        
        # GUARDAMOS EL RESULTADO EN LA MEMORIA DE LA SESIÓN
        st.session_state['precio_calculado'] = precio
        st.session_state['m2_calculado'] = m2

    # --- MOSTRAR RESULTADO (Si existe en memoria) ---
    if st.session_state['precio_calculado'] is not None:
        precio_final = st.session_state['precio_calculado']
        m2_final = st.session_state['m2_calculado']
        
        st.markdown(f"""
        <div class="resultado-box">
            <h3 style="margin-bottom: 0px;">U$S {precio_final:,.0f}</h3>
            <p style="margin-bottom: 5px;">Precio Estimado de Mercado</p>
            <hr style="margin: 5px 0; border-top: 1px solid #ccc;">
            <p style="font-size: 14px; margin-bottom: 0;">Valor por m²: <b>U$S {m2_final:,.0f}</b></p>
        </div>
        """, unsafe_allow_html=True)