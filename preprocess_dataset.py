"""
preprocess_dataset.py
=====================
Carga de `inmuebles.db`, filtros de filas, limpieza de outliers y feature
engineering para el Tasador Inmobiliario MDP.

Este módulo es la ÚNICA fuente de verdad de la lógica de features:
lo importa `train_model.py` para entrenar y `app_tasador.py` para inferencia,
de modo que `banos_totales`, `antiguedad_categoria`, etc. se calculan
exactamente igual en ambos lados.

Feature set final (decisión cerrada — ver brief):
    covered_area_m2, rooms, bedrooms, banos_totales, parking,
    en_construccion, antiguedad_categoria (ordinal 0-5, NaN si en construcción),
    real_estate_type (categórica), latitud, longitud, cluster_ubicacion
    [+ precio_m2_zona_cluster si USAR_PRECIO_ZONA]
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ruta por defecto de la base, resuelta contra la estructura real de carpetas:
#   Zonaprop/
#     Scraper/data/inmuebles.db
#     Tasador/preprocess_dataset.py   <- este archivo
# Si algún día movés las carpetas, pasá la ruta explícita a cargar_dataset()
# o corré train_model.py con --db "ruta\a\inmuebles.db".
# ---------------------------------------------------------------------------
DB_PATH_DEFAULT = Path(__file__).resolve().parent.parent / "Scraper" / "data" / "inmuebles.db"

# ---------------------------------------------------------------------------
# Constantes configurables
# ---------------------------------------------------------------------------

# Peso de un toilette relativo a un baño completo al construir `banos_totales`.
# Cambiar a 1.0 si se decide que un toilette cuenta como baño entero.
PESO_TOILETTE: float = 0.5

# Si True, se agrega la feature `precio_m2_zona_cluster` (precio/m² promedio
# del cluster de ubicación, calculado SOLO sobre el set de entrenamiento).
USAR_PRECIO_ZONA: bool = True

# Bounding box razonable para Mar del Plata y alrededores (General Pueyrredón).
# Filas con coordenadas fuera de esto son errores de geocoding del aviso.
LAT_MIN, LAT_MAX = -38.30, -37.70
LON_MIN, LON_MAX = -58.00, -57.30

# Límites duros de sanidad de precio
PRECIO_MINIMO_USD = 5_000

# Percentiles para recorte de precio/m² (sobre covered_area_m2).
# El rango resultante en esta base queda aproximadamente en la referencia
# USD 400–4.500/m² mencionada en el brief.
PCTL_M2_INF = 0.005
PCTL_M2_SUP = 0.995

# Categorías de antigüedad (ordinal). El orden del listado ES el encoding.
ANTIGUEDAD_LABELS = [
    "A estrenar",        # 0
    "Hasta 5 años",      # 1
    "6 a 15 años",       # 2
    "16 a 30 años",      # 3
    "31 a 50 años",      # 4
    "Más de 50 años",    # 5
]

# Columnas numéricas base (sin geo ni derivadas de cluster)
FEATURES_NUMERICAS = [
    "covered_area_m2",
    "rooms",
    "bedrooms",
    "banos_totales",
    "parking",
    "en_construccion",
    "antiguedad_categoria",
]

FEATURES_GEO = ["latitud", "longitud", "cluster_ubicacion"]

TIPOS_VALIDOS = ["Apartamento", "Casa", "PH", "Local Comercial", "Oficina comercial"]


# ---------------------------------------------------------------------------
# Feature engineering elemental (reutilizado en inferencia)
# ---------------------------------------------------------------------------

def calcular_banos_totales(bathrooms, toilette, peso_toilette: float = PESO_TOILETTE):
    """banos_totales = bathrooms + peso_toilette * toilette (NaN-safe)."""
    b = pd.to_numeric(pd.Series(bathrooms) if not isinstance(bathrooms, pd.Series) else bathrooms,
                      errors="coerce")
    t = pd.to_numeric(pd.Series(toilette) if not isinstance(toilette, pd.Series) else toilette,
                      errors="coerce").fillna(0)
    return b + peso_toilette * t


def categorizar_antiguedad(antiquity: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Devuelve (en_construccion, antiguedad_categoria).

    - `en_construccion`: 1 si antiquity == 'En construcción', 0 en otro caso.
    - `antiguedad_categoria`: ordinal 0-5 según los cortes del brief.
      NaN cuando en_construccion == 1 (no aplica) o cuando antiquity es nula.
    """
    s = antiquity.astype("string").str.strip()

    en_constr = (s == "En construcción").fillna(False).astype(int)

    anios = pd.to_numeric(s, errors="coerce")  # 'A estrenar' / 'En construcción' -> NaN

    cat = pd.Series(np.nan, index=s.index, dtype="float64")
    cat[s == "A estrenar"] = 0
    cat[(anios >= 1) & (anios <= 5)] = 1
    cat[(anios >= 6) & (anios <= 15)] = 2
    cat[(anios >= 16) & (anios <= 30)] = 3
    cat[(anios >= 31) & (anios <= 50)] = 4
    cat[anios > 50] = 5
    # antiquity == 0 explícito lo tratamos como "A estrenar"
    cat[anios == 0] = 0
    # En construcción: la categoría no aplica
    cat[en_constr == 1] = np.nan

    return en_constr, cat


def antiguedad_desde_seleccion(seleccion: str) -> tuple[int, float]:
    """
    Mapea la opción elegida en la app ('A estrenar', ..., 'En construcción')
    a (en_construccion, antiguedad_categoria) con la MISMA lógica que el
    entrenamiento.
    """
    if seleccion == "En construcción":
        return 1, np.nan
    return 0, float(ANTIGUEDAD_LABELS.index(seleccion))


# ---------------------------------------------------------------------------
# Carga + filtros + limpieza (solo entrenamiento)
# ---------------------------------------------------------------------------

def cargar_dataset(db_path: str | Path | None = None, verbose: bool = True) -> pd.DataFrame:
    """
    Carga `propiedades`, aplica los filtros de filas del brief, limpia
    outliers y construye todas las features (excepto cluster/precio de zona,
    que dependen de objetos entrenados y viven en train_model.py).

    Si `db_path` es None, usa `DB_PATH_DEFAULT` (la ubicación real de
    `inmuebles.db` en `Scraper/data/`, hermana de la carpeta `Tasador/`).
    """
    if db_path is None:
        db_path = DB_PATH_DEFAULT
    db_path = Path(db_path)

    # sqlite3.connect() NO tira error si el archivo no existe: crea una base
    # vacía nueva en silencio. Sin este chequeo, una ruta mal escrita
    # produce un dataset vacío en vez de un error claro.
    if not db_path.exists():
        raise FileNotFoundError(
            f"No se encuentra la base en: {db_path}\n"
            f"Pasá la ruta correcta con cargar_dataset(db_path=...) "
            f"o, si usás train_model.py, con --db \"ruta\\a\\inmuebles.db\"."
        )

    con = sqlite3.connect(db_path)
    query = """
        SELECT
            price_value,
            covered_area_m2,
            rooms,
            bedrooms,
            bathrooms,
            toilette,
            parking,
            antiquity,
            real_estate_type,
            latitud,
            longitud
        FROM propiedades
        WHERE operation_type = 'venta'
          AND price_type     = 'USD'
          AND status         = 'ONLINE'
          AND posting_type   = 'PROPERTY'
          AND real_estate_type != 'Terrenos'
    """
    df = pd.read_sql(query, con)
    con.close()
    n0 = len(df)

    # --- Sanidad de coordenadas (app map-first: sin coords no hay fila) ---
    df = df.dropna(subset=["latitud", "longitud"])
    df = df[df["latitud"].between(LAT_MIN, LAT_MAX) & df["longitud"].between(LON_MIN, LON_MAX)]
    n1 = len(df)

    # --- Superficie: covered_area_m2 obligatoria y razonable ---
    df = df.dropna(subset=["covered_area_m2"])
    df = df[df["covered_area_m2"].between(15, 2_000)]
    n2 = len(df)

    # --- Precio: mínimo duro + recorte por percentiles de precio/m² ---
    df = df[df["price_value"] >= PRECIO_MINIMO_USD]
    df["precio_m2"] = df["price_value"] / df["covered_area_m2"]
    lo = df["precio_m2"].quantile(PCTL_M2_INF)
    hi = df["precio_m2"].quantile(PCTL_M2_SUP)
    # Acotamos también con el rango de referencia del brief para MDP
    lo = max(lo, 400.0)
    hi = min(hi, 4_500.0)
    df = df[df["precio_m2"].between(lo, hi)]
    n3 = len(df)

    # --- Features ---
    df["banos_totales"] = calcular_banos_totales(df["bathrooms"], df["toilette"])
    df["en_construccion"], df["antiguedad_categoria"] = categorizar_antiguedad(df["antiquity"])
    df["real_estate_type"] = pd.Categorical(df["real_estate_type"], categories=TIPOS_VALIDOS)

    # rooms/bedrooms/banos/parking: NaN se deja como NaN — LightGBM lo maneja
    # nativamente y refleja la realidad del aviso (dato no informado).
    for col in ["rooms", "bedrooms", "parking"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sanidad: valores absurdos (rooms=46, banos=124...) son errores de carga
    # del aviso -> los tratamos como "no informado" (NaN), no como señal.
    caps = {"rooms": 15, "bedrooms": 12, "banos_totales": 10, "parking": 10}
    for col, cap in caps.items():
        df.loc[df[col] > cap, col] = np.nan

    df = df.drop(columns=["bathrooms", "toilette", "antiquity"]).reset_index(drop=True)

    if verbose:
        print(f"Filas post-filtro SQL:        {n0:>6,}")
        print(f"Post-limpieza coordenadas:    {n1:>6,}")
        print(f"Post-limpieza superficie:     {n2:>6,}")
        print(f"Post-limpieza precio/m²:      {n3:>6,}  (rango m²: {lo:,.0f}–{hi:,.0f} USD/m²)")

    return df


def construir_matriz_features(df: pd.DataFrame, usar_precio_zona: bool = USAR_PRECIO_ZONA) -> list[str]:
    """Devuelve la lista ordenada de columnas de entrenamiento."""
    cols = FEATURES_NUMERICAS + FEATURES_GEO + ["real_estate_type"]
    if usar_precio_zona:
        cols = cols + ["precio_m2_zona_cluster"]
    return cols


if __name__ == "__main__":
    d = cargar_dataset()
    print(d.head())
    print(d.describe(include="all").T)
