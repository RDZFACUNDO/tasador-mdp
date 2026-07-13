"""
app_tasador.py
==============
Tasador Inmobiliario MDP — app Streamlit (v2).

- Modelo: LightGBM de cuantiles (P10/P50/P90) + calibración conforme,
  entrenado por `train_model.py` y cargado desde `modelo_tasador.pkl`.
- Map-first: la ubicación se fija clickeando el mapa (folium).
- Muestra: rango visual P10–P90, desglose SHAP en lenguaje simple,
  comparables reales y aviso de confianza por zona.
- Branding del sitio: #5B8B8A / Roboto / tarjetas blancas redondeadas.
- Preparada para correr embebida en iframe (?embedded=true).
"""

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Workaround defensivo: el backend de strings de pyarrow puede segfaultear al
# construir DataFrames dentro del hilo de script de Streamlit en algunos
# builds de pandas/pyarrow. El storage 'python' es igual de correcto y no
# tiene costo perceptible en frames de una fila.
pd.set_option("mode.string_storage", "python")

import streamlit as st
import folium
from streamlit_folium import st_folium

from preprocess_dataset import (
    ANTIGUEDAD_LABELS,
    antiguedad_desde_seleccion,
    calcular_banos_totales,
)

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Tasador Inmobiliario MDP",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

COLOR = "#5B8B8A"        # verde salvia del sitio
COLOR_HOVER = "#4a7a79"
COLOR_BG = "#f5f5f5"

EMBEDDED = st.query_params.get("embedded", "false").lower() == "true"

MIN_COMPARABLES_ZONA = 30  # umbral para el aviso de "pocos datos en esta zona"

# Ruta absoluta al artefacto, resuelta contra la ubicación del script.
# Así la app carga bien el modelo sin importar desde qué carpeta se
# ejecute `streamlit run` (evita el clásico FileNotFoundError por cwd).
ARTEFACTO_PATH = Path(__file__).resolve().parent / "modelo_tasador.pkl"

BARRIOS = {
    "Centrar en...": (None, None),
    "Playa Grande": (-38.0169, -57.5309),
    "Varese": (-38.0120, -57.5350),
    "Güemes": (-38.0122, -57.5388),
    "Centro": (-38.0055, -57.5427),
    "La Perla": (-37.9926, -57.5492),
    "Constitución": (-37.9754, -57.5583),
    "Puerto": (-38.0331, -57.5406),
    "Chauvin": (-38.0053, -57.5563),
}

# ---------------------------------------------------------------------------
# Estilos (branding del sitio, NO el verde viejo #1d6e5d)
# ---------------------------------------------------------------------------

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

html, body, [class*="css"], .stApp, p, label, span, div {{
    font-family: 'Roboto', sans-serif;
}}
.stApp {{ background: {COLOR_BG}; }}

/* Ocultar chrome de Streamlit (siempre: la app vive embebida) */
#MainMenu, header[data-testid="stHeader"], footer,
div[data-testid="stToolbar"], div[data-testid="stDecoration"],
div[data-testid="stStatusWidget"], a[data-testid="viewerBadge_link"],
div[class*="viewerBadge"] {{
    display: none !important;
    visibility: hidden !important;
}}

.block-container {{
    padding-top: {'0.8rem' if EMBEDDED else '1.6rem'} !important;
    padding-bottom: 1rem !important;
    max-width: 1300px;
}}

h1, h2, h3 {{ color: {COLOR} !important; font-weight: 700; }}

/* Tarjetas blancas estilo sitio */
.card {{
    background: white;
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    margin-bottom: 14px;
}}

/* Botón principal */
div[data-testid="stButton"] > button {{
    width: 100%;
    background-color: {COLOR} !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    height: 3em;
    transition: all .3s ease;
}}
div[data-testid="stButton"] > button:hover {{
    background-color: {COLOR_HOVER} !important;
    transform: translateY(-1px);
}}
div[data-testid="stButton"] > button p {{
    color: white !important; font-weight: 700 !important;
}}

/* Sliders y widgets en color de marca */
div[data-baseweb="slider"] div[role="slider"] {{ background-color: {COLOR} !important; box-shadow: none !important; }}
div[data-baseweb="slider"] > div > div > div > div {{ background-color: {COLOR} !important; }}

label, .stSelectbox label, .stNumberInput label, .stSlider label {{
    color: #444 !important; font-weight: 500 !important; font-size: 14px !important;
}}

/* Selectboxes (Ir a zona, Tipo de propiedad, Antigüedad): esta versión de
   Streamlit los arma con react-aria-ComboBox, no con BaseWeb -> se apunta
   a esa estructura real en vez de data-baseweb="select". */
div[data-testid="stSelectbox"] .react-aria-ComboBox,
div[data-testid="stSelectbox"] .react-aria-ComboBox [role="group"] {{
    background-color: {COLOR} !important;
    border-color: {COLOR} !important;
    border-radius: 6px !important;
}}
div[data-testid="stSelectbox"] input[role="combobox"] {{
    background-color: {COLOR} !important;
    color: white !important;
}}
div[data-testid="stSelectbox"] input[role="combobox"]::placeholder {{
    color: rgba(255,255,255,0.85) !important;
}}
div[data-testid="stSelectbox"] button {{
    background-color: {COLOR} !important;
}}
div[data-testid="stSelectbox"] svg {{ fill: white !important; }}

/* Menú desplegable emergente: blanco con texto oscuro para legibilidad,
   sea cual sea el widget que lo abre (selectbox usa role=listbox/option) */
[role="listbox"] {{ background-color: white !important; }}
[role="listbox"] [role="option"] {{ color: #333 !important; }}
[role="listbox"] [role="option"]:hover,
[role="listbox"] [role="option"][aria-selected="true"] {{
    background-color: #f0f4f4 !important;
}}
ul[data-baseweb="menu"] {{ background-color: white !important; }}
ul[data-baseweb="menu"] li {{ color: #333 !important; }}

/* Number input (Metros cubiertos): mismo verde, texto blanco, botones +/- */
div[data-testid="stNumberInput"] div[data-baseweb="input"] {{
    background-color: {COLOR} !important;
    border-color: {COLOR} !important;
    border-radius: 6px !important;
}}
div[data-testid="stNumberInput"] input {{
    background-color: {COLOR} !important;
    color: white !important;
}}
div[data-testid="stNumberInput"] button {{
    background-color: {COLOR_HOVER} !important;
}}
div[data-testid="stNumberInput"] button svg {{ fill: white !important; }}

/* Radio "Estilo de mapa": el texto quedaba blanco sobre blanco, pasa a verde */
div[data-testid="stRadio"] label p {{
    color: {COLOR} !important; font-weight: 600 !important;
}}
div[data-baseweb="radio"] [aria-checked="true"] > div:first-child {{
    background-color: {COLOR} !important;
    border-color: {COLOR} !important;
}}
div[data-baseweb="radio"] > div:first-child {{ border-color: {COLOR} !important; }}

/* Expander */
details[data-testid="stExpander"] {{
    background: white; border-radius: 12px;
    border: 1px solid #e8e8e8;
}}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Carga de artefactos
# ---------------------------------------------------------------------------

@st.cache_resource
def cargar_artefactos():
    try:
        return joblib.load(ARTEFACTO_PATH)
    except FileNotFoundError:
        st.error(
            f"⚠️ No se encuentra `modelo_tasador.pkl` en `{ARTEFACTO_PATH}`. "
            f"Corré `train_model.py` primero, o verificá que el archivo esté "
            f"en la misma carpeta que `app_tasador.py`."
        )
        st.stop()


@st.cache_resource
def cargar_explainer(_modelo):
    import shap
    return shap.TreeExplainer(_modelo)


ART = cargar_artefactos()
MODELOS = ART["modelos_cuantiles"]
KMEANS = ART["kmeans"]
COLS = ART["columnas"]
Q_CONF = ART["q_conforme"]
PRECIO_ZONA = ART["precio_m2_zona_cluster"]
MEDIA_GLOBAL = ART["precio_m2_media_global"]
CONTEO_CLUSTER = ART["conteo_por_cluster"]
USAR_PRECIO_ZONA = ART["usar_precio_zona"]
TIPOS = ART["tipos_validos"]
COMPARABLES = ART["comparables"]
K_CLUSTERS = ART["metadata"]["k_clusters"]

CENTROIDES = KMEANS.cluster_centers_

TIPO_LABELS = {
    "Apartamento": "Departamento",
    "Casa": "Casa",
    "PH": "PH",
    "Local Comercial": "Local Comercial",
    "Oficina comercial": "Oficina",
}
TIPO_DESDE_LABEL = {v: k for k, v in TIPO_LABELS.items()}


def fmt_usd(x: float) -> str:
    """U$S 169.149 (convención es-AR)."""
    return f"U$S {x:,.0f}".replace(",", ".")


# ---------------------------------------------------------------------------
# Estado de sesión
# ---------------------------------------------------------------------------

DEFAULTS = {
    "lat": -38.0055, "lon": -57.5427,   # Centro MDP
    "resultado": None,                   # dict con p10/p50/p90/shap/etc.
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------------------------------------------------------------------
# Lógica de predicción
# ---------------------------------------------------------------------------

def armar_fila(lat, lon, tipo, metros, ambientes, dormitorios,
               banos, toilettes, cocheras, antiguedad_sel) -> pd.DataFrame:
    cluster = int(KMEANS.predict(np.array([[lat, lon]]))[0])
    en_constr, ant_cat = antiguedad_desde_seleccion(antiguedad_sel)
    banos_tot = float(calcular_banos_totales([banos], [toilettes]).iloc[0])

    fila = {
        "covered_area_m2": float(metros),
        "rooms": float(ambientes),
        "bedrooms": float(dormitorios),
        "banos_totales": banos_tot,
        "parking": float(cocheras),
        "en_construccion": en_constr,
        "antiguedad_categoria": ant_cat,
        "latitud": lat,
        "longitud": lon,
        "cluster_ubicacion": cluster,
        "real_estate_type": tipo,
    }
    if USAR_PRECIO_ZONA:
        fila["precio_m2_zona_cluster"] = PRECIO_ZONA.get(cluster, MEDIA_GLOBAL)

    X = pd.DataFrame([fila])[COLS]
    X["real_estate_type"] = pd.Categorical(X["real_estate_type"], categories=TIPOS)
    X["cluster_ubicacion"] = pd.Categorical(
        X["cluster_ubicacion"], categories=list(range(K_CLUSTERS))
    )
    return X, cluster


GRUPOS_SHAP = {
    "Ubicación": ["cluster_ubicacion", "precio_m2_zona_cluster", "latitud", "longitud"],
    "Superficie cubierta": ["covered_area_m2"],
    "Ambientes y dormitorios": ["rooms", "bedrooms"],
    "Baños": ["banos_totales"],
    "Cocheras": ["parking"],
    "Antigüedad": ["antiguedad_categoria", "en_construccion"],
    "Tipo de propiedad": ["real_estate_type"],
}


def desglose_shap(X: pd.DataFrame, pred_log: float) -> list[tuple[str, float]]:
    """Contribución en USD de cada grupo de features, ordenada por impacto."""
    explainer = cargar_explainer(MODELOS["p50"])
    sv = explainer.shap_values(X)[0]
    contrib_log = dict(zip(COLS, sv))
    precio = float(np.expm1(pred_log))

    resultado = []
    for grupo, cols_g in GRUPOS_SHAP.items():
        v = sum(contrib_log.get(c, 0.0) for c in cols_g)
        # USD que aporta el grupo: precio actual vs precio sin ese aporte
        delta = precio - float(np.expm1(pred_log - v))
        resultado.append((grupo, delta))
    return sorted(resultado, key=lambda t: -abs(t[1]))


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def buscar_comparables(lat, lon, cluster, tipo, metros, n=5) -> pd.DataFrame:
    """Comparables reales: mismo cluster (o vecinos), mismo tipo, superficie similar."""
    # Clusters vecinos por distancia entre centroides
    d_centroides = np.linalg.norm(CENTROIDES - np.array([lat, lon]), axis=1)
    clusters_cercanos = list(np.argsort(d_centroides)[:6])  # propio + 5 vecinos

    c = COMPARABLES
    cand = c[
        (c["real_estate_type"] == tipo)
        & (c["cluster_ubicacion"].isin(clusters_cercanos))
        & (c["covered_area_m2"].between(metros * 0.6, metros * 1.4))
    ].copy()

    if len(cand) < 3:  # relajar superficie si hay pocos
        cand = c[
            (c["real_estate_type"] == tipo)
            & (c["cluster_ubicacion"].isin(clusters_cercanos))
        ].copy()

    if cand.empty:
        return cand

    cand["dist_m"] = cand.apply(
        lambda r: haversine_m(lat, lon, r["latitud"], r["longitud"]), axis=1
    )
    return cand.sort_values("dist_m").head(n)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.markdown(
    f"<h2 style='margin-bottom:4px;'>🏠 Tasador Online · Mar del Plata</h2>"
    f"<p style='color:#666; margin-top:0; font-size:14px;'>"
    f"Hacé clic en el mapa para fijar la ubicación, completá las características "
    f"y calculá el valor de mercado estimado.</p>",
    unsafe_allow_html=True,
)

col_mapa, col_datos = st.columns([3, 1.9], gap="medium")

# ----------------------------- Columna mapa -------------------------------
with col_mapa:
    c1, c2 = st.columns([1, 1])
    with c1:
        estilo_mapa = st.radio(
            "Estilo de mapa", ["Calles", "Claro"],
            horizontal=True, label_visibility="collapsed",
        )
    with c2:
        zona_elegida = st.selectbox(
            "Ir a zona", list(BARRIOS.keys()), label_visibility="collapsed",
        )

    if zona_elegida != "Centrar en...":
        nlat, nlon = BARRIOS[zona_elegida]
        if abs(nlat - st.session_state["lat"]) > 1e-6 or abs(nlon - st.session_state["lon"]) > 1e-6:
            st.session_state["lat"], st.session_state["lon"] = nlat, nlon
            st.rerun()

    tiles = "CartoDB positron" if estilo_mapa == "Claro" else "OpenStreetMap"
    m = folium.Map(
        location=[st.session_state["lat"], st.session_state["lon"]],
        zoom_start=14, min_zoom=12, tiles=tiles,
    )
    folium.Marker(
        [st.session_state["lat"], st.session_state["lon"]],
        popup="Propiedad",
        icon=folium.Icon(color="red", icon="home"),
    ).add_to(m)

    salida_mapa = st_folium(
        m,
        height=640 if not EMBEDDED else 490,
        use_container_width=True,
        returned_objects=["last_clicked"],
    )

    if salida_mapa.get("last_clicked"):
        clat = salida_mapa["last_clicked"]["lat"]
        clon = salida_mapa["last_clicked"]["lng"]
        if (abs(clat - st.session_state["lat"]) > 1e-4
                or abs(clon - st.session_state["lon"]) > 1e-4):
            st.session_state["lat"], st.session_state["lon"] = clat, clon
            st.rerun()

    st.markdown(
        f"<p style='font-size:13px; color:#666; margin-top:2px;'>"
        f"📍 Ubicación: {st.session_state['lat']:.5f}, {st.session_state['lon']:.5f} "
        f"&nbsp;·&nbsp; clic en el mapa para ajustar</p>",
        unsafe_allow_html=True,
    )

# --------------------------- Columna formulario ---------------------------
with col_datos:
    tipo_label = st.selectbox("Tipo de propiedad", list(TIPO_DESDE_LABEL.keys()))
    tipo = TIPO_DESDE_LABEL[tipo_label]

    label_metros = ("Metros cubiertos (m²)" if tipo != "Casa"
                    else "Metros cubiertos (m², sin patio/jardín)")
    metros = st.number_input(label_metros, min_value=15, max_value=1000, value=60)

    ambientes = st.slider("Ambientes", 1, 8, 2)
    dormitorios = st.slider("Dormitorios", 0, 6, 1)

    cb1, cb2 = st.columns(2)
    with cb1:
        banos = st.slider("Baños", 1, 5, 1)
    with cb2:
        toilettes = st.slider("Toilettes", 0, 2, 0)

    cocheras = st.slider("Cocheras", 0, 4, 0)

    antiguedad_sel = st.selectbox(
        "Antigüedad", ANTIGUEDAD_LABELS + ["En construcción"],
    )

    calcular = st.button("CALCULAR VALOR", width="stretch")

# ---------------------------------------------------------------------------
# Cálculo
# ---------------------------------------------------------------------------

if calcular:
    lat, lon = st.session_state["lat"], st.session_state["lon"]
    X, cluster = armar_fila(
        lat, lon, tipo, metros, ambientes, dormitorios,
        banos, toilettes, cocheras, antiguedad_sel,
    )

    log_p10 = float(MODELOS["p10"].predict(X)[0]) - Q_CONF
    log_p50 = float(MODELOS["p50"].predict(X)[0])
    log_p90 = float(MODELOS["p90"].predict(X)[0]) + Q_CONF

    p50 = float(np.expm1(log_p50))
    p10 = min(float(np.expm1(log_p10)), p50)
    p90 = max(float(np.expm1(log_p90)), p50)

    comparables = buscar_comparables(lat, lon, cluster, tipo, metros)
    n_zona = CONTEO_CLUSTER.get(cluster, 0)

    st.session_state["resultado"] = {
        "p10": p10, "p50": p50, "p90": p90,
        "m2": p50 / metros,
        "shap": desglose_shap(X, log_p50),
        "comparables": comparables,
        "n_zona": n_zona,
        "tipo_label": tipo_label,
        "metros": metros,
        "lat": lat, "lon": lon,
    }

# ---------------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------------

res = st.session_state["resultado"]
if res:
    p10, p50, p90 = res["p10"], res["p50"], res["p90"]

    # Posición del P50 dentro del rango, para el marcador de la barra
    pos = 50.0 if p90 <= p10 else (p50 - p10) / (p90 - p10) * 100
    pos = min(max(pos, 2), 98)

    aviso_zona = ""
    if res["n_zona"] < MIN_COMPARABLES_ZONA:
        aviso_zona = (
            f"<div style='background:#fff8e6; border-left:4px solid #e6a817; "
            f"border-radius:6px; padding:8px 12px; margin-top:10px; font-size:13px; color:#7a5c00;'>"
            f"⚠️ <b>Estimación con menor precisión:</b> hay pocos datos en esta zona "
            f"({res['n_zona']} propiedades en la base). El rango puede ser menos confiable."
            f"</div>"
        )

    st.markdown(f"""
<div class="card" style="text-align:center;">
  <p style="margin:0; font-size:13px; color:#888; letter-spacing:1px;">VALOR DE MERCADO ESTIMADO — {res['tipo_label'].upper()} · {res['metros']} m²</p>
  <h1 style="margin:2px 0 0 0; font-size:38px; color:{COLOR} !important;">{fmt_usd(p50)}</h1>
  <p style="margin:0 0 14px 0; font-size:14px; color:#888;">~ {fmt_usd(res['m2'])} / m² cubierto</p>

  <!-- Barra de rango P10–P90 -->
  <div style="position:relative; height:14px; border-radius:7px; margin:6px 8px 2px 8px;
              background:linear-gradient(90deg, #e8b04b 0%, {COLOR} 50%, #67b26f 100%); opacity:.92;">
    <div style="position:absolute; left:{pos:.1f}%; top:-5px; transform:translateX(-50%);
                width:4px; height:24px; background:#2c3e50; border-radius:2px;"></div>
  </div>
  <div style="display:flex; justify-content:space-between; padding:4px 8px 0 8px;">
    <div style="text-align:left;">
      <span style="font-size:11px; color:#b07d1a; font-weight:700;">MÍNIMO (P10)</span><br>
      <span style="font-size:16px; color:#333; font-weight:700;">{fmt_usd(p10)}</span>
    </div>
    <div style="text-align:center;">
      <span style="font-size:11px; color:#888;">80% de las propiedades comparables<br>se venden dentro de este rango</span>
    </div>
    <div style="text-align:right;">
      <span style="font-size:11px; color:#3e8e4d; font-weight:700;">MÁXIMO (P90)</span><br>
      <span style="font-size:16px; color:#333; font-weight:700;">{fmt_usd(p90)}</span>
    </div>
  </div>
  {aviso_zona}
</div>
""", unsafe_allow_html=True)

    col_shap, col_comp = st.columns(2, gap="medium")

    # ------------------------- Por qué este precio -----------------------
    with col_shap:
        with st.expander("💡 ¿Por qué este precio?", expanded=not EMBEDDED):
            st.markdown(
                "<p style='font-size:13px; color:#666;'>Aporte de cada característica "
                "al valor estimado, respecto de una propiedad promedio de la base:</p>",
                unsafe_allow_html=True,
            )
            max_abs = max(abs(d) for _, d in res["shap"]) or 1.0
            filas_html = ""
            for grupo, delta in res["shap"]:
                if abs(delta) < 500:
                    continue
                ancho = abs(delta) / max_abs * 100
                color_barra = "#67b26f" if delta >= 0 else "#e8875b"
                signo = "+" if delta >= 0 else "−"
                filas_html += f"""
<div style="display:flex; align-items:center; margin-bottom:7px; font-size:13px;">
  <div style="width:44%; color:#444;">{grupo}</div>
  <div style="width:34%; background:#f0f0f0; border-radius:4px; height:10px;">
    <div style="width:{ancho:.0f}%; background:{color_barra}; height:10px; border-radius:4px;"></div>
  </div>
  <div style="width:22%; text-align:right; font-weight:700; color:{'#3e8e4d' if delta>=0 else '#c0603a'};">
    {signo}{fmt_usd(abs(delta)).replace('U$S ','US$')}
  </div>
</div>"""
            st.markdown(filas_html or "<p style='font-size:13px;'>Sin aportes relevantes.</p>",
                        unsafe_allow_html=True)

    # --------------------------- Comparables -----------------------------
    with col_comp:
        with st.expander("🏘️ Propiedades comparables reales", expanded=not EMBEDDED):
            comp = res["comparables"]
            if comp.empty:
                st.markdown(
                    "<p style='font-size:13px; color:#666;'>No encontramos propiedades "
                    "similares publicadas cerca de esta ubicación.</p>",
                    unsafe_allow_html=True,
                )
            else:
                filas_html = ""
                for _, r in comp.iterrows():
                    dist = (f"{r['dist_m']:.0f} m" if r["dist_m"] < 1000
                            else f"{r['dist_m']/1000:.1f} km")
                    detalles = f"{r['covered_area_m2']:.0f} m²"
                    if pd.notna(r["rooms"]):
                        detalles += f" · {int(r['rooms'])} amb"
                    filas_html += f"""
<div style="display:flex; justify-content:space-between; align-items:center;
            border-bottom:1px solid #f0f0f0; padding:7px 0; font-size:13px;">
  <div>
    <span style="color:#333; font-weight:500;">{TIPO_LABELS.get(r['real_estate_type'], r['real_estate_type'])} · {detalles}</span><br>
    <span style="color:#999; font-size:12px;">a {dist} · {fmt_usd(r['precio_m2']).replace('U$S ','US$')}/m²</span>
  </div>
  <div style="font-weight:700; color:{COLOR};">{fmt_usd(r['price_value'])}</div>
</div>"""
                st.markdown(filas_html, unsafe_allow_html=True)
                st.markdown(
                    "<p style='font-size:11px; color:#aaa; margin-top:6px;'>"
                    "Precios de publicación reales en Zonaprop, cercanos a la ubicación elegida.</p>",
                    unsafe_allow_html=True,
                )

# ---------------------------------------------------------------------------
# Pie: transparencia del modelo
# ---------------------------------------------------------------------------

meta = ART["metadata"]
with st.expander("ℹ️ Sobre este modelo"):
    mets = pd.DataFrame(meta["metricas_por_tipo"])
    mets["MAPE"] = (mets["MAPE"] * 100).round(1).astype(str) + "%"
    mets["cobertura_P10_P90"] = (mets["cobertura_P10_P90"] * 100).round(0).astype(int).astype(str) + "%"
    mets["MAE"] = mets["MAE"].round(0).map(lambda v: fmt_usd(v))
    st.markdown(
        f"<p style='font-size:13px; color:#666;'>Modelo LightGBM de regresión por cuantiles "
        f"con intervalos calibrados, entrenado el {meta['fecha_entrenamiento'][:10]} sobre "
        f"<b>{meta['n_dataset']:,}</b> propiedades publicadas en Mar del Plata. "
        f"Error típico (MAPE) y cobertura del rango por tipo de propiedad:</p>".replace(",", "."),
        unsafe_allow_html=True,
    )
    st.dataframe(
        mets[["tipo", "n_test", "MAE", "MAPE", "cobertura_P10_P90"]],
        hide_index=True, width="stretch",
    )
    st.markdown(
        "<p style='font-size:12px; color:#aaa;'>Los valores son estimaciones estadísticas "
        "sobre precios de publicación y no reemplazan una tasación profesional.</p>",
        unsafe_allow_html=True,
    )
