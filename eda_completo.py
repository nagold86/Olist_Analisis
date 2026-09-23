#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Olist E-Commerce - EDA completo
# Unificacion de las 9 tablas + analisis exploratorio.
# Se puede correr directo con: python eda_completo.py
#


#
# Olist E-Commerce — EDA completo (unificación + análisis)
#
# Script autocontenido: une las 9 tablas crudas de Kaggle y corre el EDA completo, sin depender de
# ningún token ni cuenta de API. Cada bloque de código tiene, antes, un comentario con **qué
# hacemos** y **para qué**, y después (cuando corresponde) uno con el **insight** que deja ese resultado.
# El informe funcional (docx) resume todo esto en un solo documento, con el mismo criterio.
#

#
# 0. Configuración inicial
#
# **Qué hacemos:** importamos las librerías que se usan en todo el script (pandas/numpy para datos,
# matplotlib/seaborn/plotly/folium para gráficos y mapas, wordcloud/nltk/re/unicodedata para el análisis
# de texto) y definimos la paleta de colores institucional de Olist (azul #0A4EE4 como color principal).
#
# **Para qué:** para no repetir imports en cada bloque y para que todos los gráficos del script usen
# la misma paleta, dando una identidad visual consistente.
#

import re
import unicodedata
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

OLIST_BLUE = "#0A4EE4"
OLIST_BLUE_DARK = "#0D366B"
OLIST_GRAY = "#52514E"
PALETA_CATEGORICA = ["#0A4EE4", "#eb6834", "#1baf7a", "#eda100",
                     "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SUDESTE = {"SP", "RJ", "MG", "ES"}

sns.set_theme(style="whitegrid", rc={
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": OLIST_GRAY,
    "text.color": "#0b0b0b",
    "xtick.color": OLIST_GRAY,
    "ytick.color": OLIST_GRAY,
    "font.family": "sans-serif",
})
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.titleweight"] = "bold"
pd.set_option("display.max_columns", 50)

FIGS_DIR = Path("figuras_eda")
FIGS_DIR.mkdir(exist_ok=True)
_fig_i = 0

#
# 1. Datos
#
# **Qué hacemos:** dejamos documentado de dónde salen los datos: los 9 CSV originales de Kaggle ya
# están incluidos en `data/raw/` (se bajaron una sola vez, con sesión logueada en el navegador, sin
# token de API) y viajan junto con este script.
#
# **Para qué:** para que cualquier persona pueda abrir esta carpeta y correr el script de punta a
# punta sin tener que crearse una cuenta de API de Kaggle ni configurar credenciales. Si en algún
# momento hay que refrescar el dataset: se entra a la página de Kaggle logueado, se toca "Download",
# se descomprime y se reemplazan los CSV de `data/raw/`.
#

RAW_DIR = Path("data/raw")
CSV_UNIFICADO = Path("data/olist_dataset_unificado.csv")

#
# **Qué hacemos:** unimos las 9 tablas crudas en un único dataset a nivel de "ítem de pedido"
# (cada fila = un producto dentro de un pedido), agregando pagos y reviews a nivel de pedido antes de
# unir para no duplicar filas. Además del cruce de geolocalización del cliente (que ya teníamos),
# ahora sumamos también la geolocalización del **vendedor**, necesaria para poder calcular la distancia
# cliente-vendedor más adelante. Esta bloque solo se ejecuta si todavía no existe el CSV unificado.
#
# **Para qué:** porque el dataset original replica una base de datos operativa real (normalizada en
# 9 tablas relacionadas) y hay que integrarlo antes de poder analizarlo como una sola tabla.
#

if not CSV_UNIFICADO.exists():
    orders = pd.read_csv(RAW_DIR / "olist_orders_dataset.csv")
    items = pd.read_csv(RAW_DIR / "olist_order_items_dataset.csv")
    payments = pd.read_csv(RAW_DIR / "olist_order_payments_dataset.csv")
    reviews = pd.read_csv(RAW_DIR / "olist_order_reviews_dataset.csv")
    customers = pd.read_csv(RAW_DIR / "olist_customers_dataset.csv")
    products = pd.read_csv(RAW_DIR / "olist_products_dataset.csv")
    sellers = pd.read_csv(RAW_DIR / "olist_sellers_dataset.csv")
    geolocation = pd.read_csv(RAW_DIR / "olist_geolocation_dataset.csv")
    cat_translation = pd.read_csv(RAW_DIR / "product_category_name_translation.csv")

    # Pagos a nivel order_id (un pedido puede tener varios pagos -> se suman)
    payments_agg = (
        payments.groupby("order_id")
        .agg(
            payment_value_total=("payment_value", "sum"),
            payment_installments_max=("payment_installments", "max"),
            payment_type_principal=(
                "payment_type",
                lambda x: x.value_counts().idxmax() if x.notna().any() else None,
            ),
        )
        .reset_index()
    )

    # Reviews a nivel order_id (nos quedamos con la última si hay más de una)
    reviews_agg = (
        reviews.sort_values("review_creation_date")
        .groupby("order_id")
        .last()
        .reset_index()[["order_id", "review_score", "review_comment_message"]]
    )

    products = products.merge(cat_translation, on="product_category_name", how="left")

    df = items.merge(orders, on="order_id", how="left")
    df = df.merge(products, on="product_id", how="left")
    df = df.merge(sellers, on="seller_id", how="left")
    df = df.merge(customers, on="customer_id", how="left")
    df = df.merge(payments_agg, on="order_id", how="left")
    df = df.merge(reviews_agg, on="order_id", how="left")

    # Geolocalización: promedio de lat/lng por zip_code_prefix (misma tabla, dos usos: cliente y vendedor)
    geo_prom = (
        geolocation.groupby("geolocation_zip_code_prefix")
        .agg(lat=("geolocation_lat", "mean"), lng=("geolocation_lng", "mean"))
        .reset_index()
    )

    geo_cliente = geo_prom.rename(columns={
        "geolocation_zip_code_prefix": "customer_zip_code_prefix",
        "lat": "customer_lat", "lng": "customer_lng",
    })
    df = df.merge(geo_cliente, on="customer_zip_code_prefix", how="left")

    geo_vendedor = geo_prom.rename(columns={
        "geolocation_zip_code_prefix": "seller_zip_code_prefix",
        "lat": "seller_lat", "lng": "seller_lng",
    })
    df = df.merge(geo_vendedor, on="seller_zip_code_prefix", how="left")

    CSV_UNIFICADO.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_UNIFICADO, index=False)
    print(f"[unificación] filas={df.shape[0]:,} columnas={df.shape[1]} -> {CSV_UNIFICADO}")

#
# **Insight:** el resultado queda guardado en disco, así que esta unión pesada solo se hace una
# vez. Las corridas siguientes del script saltan directo a la carga (próximo bloque).
#

#
# 2. Carga del dataset unificado
#
# **Qué hacemos:** cargamos el CSV unificado (recién generado, o ya existente de una corrida anterior)
# y miramos las primeras filas.
#
# **Para qué:** para confirmar que la unificación salió bien antes de empezar a limpiar y transformar:
# cantidad de filas/columnas esperada, y que las columnas de las 9 tablas originales están todas
# presentes en una sola tabla.
#

df = pd.read_csv(CSV_UNIFICADO)
print(f"filas={df.shape[0]:,} columnas={df.shape[1]}")
df.head()

#
# 3. Limpieza
#
# **Qué hacemos:** sobre el dataset unificado: quitamos duplicados exactos, convertimos las columnas de
# fecha (que llegan como texto) a tipo fecha real, resolvemos nulos en la categoría de producto,
# separamos "sin comentario" de "comentario vacío" en las reviews, validamos que las coordenadas
# geográficas (tanto del cliente como del vendedor) caigan dentro del rango esperado para Brasil,
# guardamos la cantidad de ítems por pedido (antes de perder esa información), y eliminamos columnas
# que no aportan al análisis.
#
# **Para qué:** son la base para poder calcular tiempos de entrega, agrupar por categoría, hacer NLP
# sobre las reviews y graficar el mapa sin que datos corruptos, mal tipados o columnas irrelevantes
# compliquen esos cálculos más adelante.
#

COLUMNAS_FECHA = [
    "shipping_limit_date", "order_purchase_timestamp", "order_approved_at",
    "order_delivered_carrier_date", "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

df = df.drop_duplicates().reset_index(drop=True)

for col in COLUMNAS_FECHA:
    df[col] = pd.to_datetime(df[col], errors="coerce")

df["product_category_name_english"] = (
    df["product_category_name_english"].fillna("sin_categoria").str.strip()
)

df["tiene_comentario"] = df["review_comment_message"].notna().astype(int)
df["review_comment_message"] = df["review_comment_message"].fillna("")

for prefijo in ["customer", "seller"]:
    df[f"{prefijo}_lat"] = pd.to_numeric(df[f"{prefijo}_lat"], errors="coerce")
    df[f"{prefijo}_lng"] = pd.to_numeric(df[f"{prefijo}_lng"], errors="coerce")
    geo_valida = df[f"{prefijo}_lat"].between(-34, 6) & df[f"{prefijo}_lng"].between(-74, -32)
    df.loc[~geo_valida, [f"{prefijo}_lat", f"{prefijo}_lng"]] = np.nan

# order_item_id es el n° de línea del ítem dentro del pedido: antes de eliminarlo, guardamos la
# cantidad de ítems por pedido en una variable propia, para no perder esa información.
df["items_por_pedido"] = df.groupby("order_id")["order_item_id"].transform("count")

COLUMNAS_A_ELIMINAR = [
    "product_category_name",        # ya tenemos la traducción en product_category_name_english
    "product_name_lenght",
    "product_description_lenght",
    "product_photos_qty",
    "order_item_id",                 # su información relevante ya quedó en items_por_pedido
    "customer_zip_code_prefix",       # solo se usaba para el join de geolocalización
    "seller_zip_code_prefix",
]
df = df.drop(columns=[c for c in COLUMNAS_A_ELIMINAR if c in df.columns])

print(f"Filas finales: {df.shape[0]:,} | columnas finales: {df.shape[1]}")

#
# **Insight:** ninguna fila se elimina por tener fechas o coordenadas faltantes/inválidas —
# esos casos quedan en NaN, no se imputan ni se borran. Los pedidos que nunca llegaron a "entregado"
# (cancelados, en camino, etc.) simplemente no tienen fecha de entrega real, y eso es correcto: no
# corresponde inventarles una. Ojo con `order_item_id`: al eliminarlo sin guardar antes
# `items_por_pedido`, dos unidades reales del mismo producto en un mismo pedido pasan a verse como
# filas "duplicadas" — no lo son, son ítems distintos comprados juntos.
#

#
# 4. Feature engineering
#
# **Qué hacemos:** construimos nueve variables nuevas que no estaban en el dataset original
# (`items_por_pedido` ya se calculó en la limpieza, como parte de salvar la información de
# `order_item_id`; las otras ocho se calculan acá):
#
# - `tiempo_entrega_dias`: días entre la compra y la entrega real.
# - `dias_vs_estimado`: diferencia en días entre la entrega real y la fecha estimada (positivo = tarde).
# - `envio_demorado`: la variable binaria 1/0 pedida en la consigna, a partir del signo de `dias_vs_estimado`.
# - `flete_ratio`: cuánto representa el flete sobre el precio del producto (freight_value / price).
# - `reseña_positiva`: versión binaria de review_score (1 si 4-5 estrellas, 0 si 1-2, NaN si 3).
# - `region_sudeste` y `mes_compra`: variables auxiliares para agrupar por región y por mes.
# - `distancia_km`: distancia geográfica (fórmula de Haversine) entre cliente y vendedor, a partir de
#   las coordenadas de ambos.
# - `volumen_cm3`: volumen del producto (largo × alto × ancho), para ver si el tamaño físico influye en
#   la logística.
#
# **Para qué:** son las variables que se usan en las estadísticas, los gráficos, el mapa, el análisis
# exhaustivo de la sección 7 y el contraste de hipótesis de las secciones siguientes.
#

df["tiempo_entrega_dias"] = (
    df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
).dt.days

df["dias_vs_estimado"] = (
    df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
).dt.days

df["envio_demorado"] = np.where(
    df["order_delivered_customer_date"].isna(),
    np.nan,
    (df["order_delivered_customer_date"] > df["order_estimated_delivery_date"]).astype(float),
)

df["flete_ratio"] = df["freight_value"] / df["price"].replace(0, np.nan)

def clasificar_reseña(score):
    if pd.isna(score):
        return np.nan
    if score <= 2:
        return 0
    if score >= 4:
        return 1
    return np.nan

df["reseña_positiva"] = df["review_score"].apply(clasificar_reseña)
df["region_sudeste"] = df["customer_state"].isin(SUDESTE).map({True: "Sudeste", False: "Resto del país"})
df["mes_compra"] = df["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0  # radio de la Tierra en km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

df["distancia_km"] = haversine_km(
    df["customer_lat"], df["customer_lng"], df["seller_lat"], df["seller_lng"]
)

df["volumen_cm3"] = df["product_length_cm"] * df["product_height_cm"] * df["product_width_cm"]

print(f"% envío demorado (entregados): {df['envio_demorado'].mean(skipna=True)*100:.1f}%")
print(f"Distancia cliente-vendedor: media={df['distancia_km'].mean():.0f} km, "
      f"mediana={df['distancia_km'].median():.0f} km")

#
# **Insight:** casi 1 de cada 10 pedidos entregados llega después de la fecha estimada
# (dato que se ve en el % impreso arriba). Esa es la primera señal de que la logística es un problema
# relevante, mucho antes de llegar a las hipótesis formales de la sección 9. La distancia
# cliente-vendedor tiene una mediana de varios cientos de kilómetros — consistente con un país del
# tamaño de Brasil y con vendedores concentrados en el eje Sudeste — lo que ya anticipa que va a pesar
# en los tiempos de entrega.
#

#
# 5. Estadísticas descriptivas
#
# **Qué hacemos:** contamos cuántos pedidos hay por estado (`order_status`) y por medio de pago
# principal, mostrando no solo la cantidad sino qué % del total representa cada categoría, y
# calculamos estadísticos (media, mediana, desvío, etc.) de las variables numéricas clave, incluyendo
# las nuevas (`distancia_km`, `volumen_cm3`, `items_por_pedido`).
#
# **Para qué:** para tener una foto rápida de la composición del dataset antes de graficar nada, y
# para detectar de entrada si alguna categoría es tan chica que no vale la pena analizarla por separado.
#

def resumen_conteo(serie, nombre="valor"):
    conteo = serie.value_counts()
    porcentaje = (serie.value_counts(normalize=True) * 100).round(1)
    return pd.DataFrame({nombre: conteo, "%": porcentaje})

print("--- order_status ---")
print(resumen_conteo(df["order_status"], "pedidos"))
print()
print("--- payment_type_principal ---")
print(resumen_conteo(df["payment_type_principal"], "pedidos"))

#
# **Insight:** la enorme mayoría de los pedidos está en estado "delivered" (entregado), así que
# el resto de las categorías (canceled, shipped, etc.) son minoritarias y explican por qué hay tan
# pocos NaN en tiempos de entrega en proporción al total.
#

df[["price", "freight_value", "flete_ratio", "payment_value_total", "tiempo_entrega_dias",
    "dias_vs_estimado", "review_score", "distancia_km", "volumen_cm3", "items_por_pedido"]].describe().round(2)

#
# **Insight:** el flete promedio representa una porción considerable del precio del producto
# (flete_ratio), lo cual es consistente con la lógica de un marketplace que envía a todo Brasil: en
# categorías de precio bajo, el costo de envío puede pesar proporcionalmente mucho más. La mayoría de
# los pedidos tiene un solo ítem (`items_por_pedido`), así que los pocos pedidos con varios ítems son
# la excepción, no la regla.
#

#
# 6. Visualizaciones
#
# Todos los gráficos de esta sección usan la paleta de colores propia inspirada en la identidad visual
# de olist.com (azul institucional #0A4EE4 como color principal), y muestran su etiqueta de datos
# (conteo, % o valor) directamente sobre el gráfico para no tener que leerlo aparte en una tabla.
#

#
# **Qué hacemos:** graficamos el % de valores nulos por columna, ordenado de mayor a menor, con
# el % de cada columna como etiqueta al lado de la barra.
#
# **Para qué:** para priorizar visualmente qué columnas necesitan más atención en la limpieza, y
# confirmar que después de la sección 3 los nulos que quedan son los esperados (fechas de pedidos no
# entregados, categoría sin traducción, etc.) y no un error de carga.
#

nulos_pct = (df.isnull().mean() * 100).sort_values(ascending=False)
nulos_pct = nulos_pct[nulos_pct > 0]

fig, ax = plt.subplots(figsize=(9, 6))
barras = nulos_pct.plot(kind="barh", color=OLIST_BLUE, ax=ax)
ax.set_title("Porcentaje de valores nulos por columna")
ax.set_xlim(0, nulos_pct.max() * 1.15)
ax.bar_label(barras.containers[0], fmt="%.1f%%", padding=3)
ax.invert_yaxis()
plt.tight_layout()
_fig_i += 1
plt.savefig(FIGS_DIR / f"fig_{_fig_i:02d}.png", dpi=110, bbox_inches="tight")
plt.show()

#
# **Insight:** las columnas con más nulos son las ligadas a reviews sin comentario, a fechas de
# entrega de pedidos que nunca se entregaron y a la geolocalización del vendedor/cliente cuando el zip
# no matchea contra la tabla de geolocalización — todos nulos estructurales, no errores.
#

#
# **Qué hacemos:** graficamos la cantidad de pedidos únicos por mes de compra, con el valor
# anotado sobre cada punto.
#
# **Para qué:** para detectar estacionalidad y picos de demanda (por ejemplo, Black Friday) a lo largo
# de 2016-2018, información relevante para planificar stock y logística.
#

ventas_mensuales = df.groupby("mes_compra")["order_id"].nunique()

fig, ax = plt.subplots(figsize=(11, 4.5))
ventas_mensuales.plot(ax=ax, color=OLIST_BLUE, linewidth=2, marker="o", markersize=4)
for x, y in zip(ventas_mensuales.index, ventas_mensuales.values):
    ax.annotate(f"{y:,}", (x, y), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=8)
ax.set_title("Cantidad de pedidos únicos por mes")
plt.tight_layout()
_fig_i += 1
plt.savefig(FIGS_DIR / f"fig_{_fig_i:02d}.png", dpi=110, bbox_inches="tight")
plt.show()

#
# **Insight:** se ve un crecimiento sostenido de pedidos a lo largo de 2017 y un pico marcado en
# noviembre de 2017 (Black Friday), seguido de una caída abrupta al final de la serie que corresponde
# a datos incompletos del último mes, no a una caída real de ventas.
#

#
# **Qué hacemos:** graficamos las 10 categorías de producto con más ítems vendidos, con la
# cantidad exacta al lado de cada barra.
#
# **Para qué:** para identificar qué categorías concentran el volumen del negocio, información que
# sirve tanto para pricing como para decidir dónde priorizar mejoras logísticas.
#

top_categorias = df["product_category_name_english"].value_counts().head(10).sort_values()

fig, ax = plt.subplots(figsize=(9, 5.5))
barras = top_categorias.plot(kind="barh", color=OLIST_BLUE, ax=ax)
ax.set_xlim(0, top_categorias.max() * 1.15)
ax.bar_label(barras.containers[0], fmt="%.0f", padding=3)
ax.set_title("Top 10 categorías de producto")
plt.tight_layout()
_fig_i += 1
plt.savefig(FIGS_DIR / f"fig_{_fig_i:02d}.png", dpi=110, bbox_inches="tight")
plt.show()

#
# **Insight:** categorías de bajo valor unitario y alta rotación (como belleza/salud y artículos
# para el hogar) dominan el volumen, más que categorías de ticket alto.
#

#
# **Qué hacemos:** graficamos la distribución de review_score (cantidad de reseñas por puntaje,
# de 1 a 5 estrellas), con el conteo exacto sobre cada barra.
#
# **Para qué:** para ver si la satisfacción está polarizada (muchos 1 y muchos 5) o concentrada en
# valores intermedios, antes de binarizarla en reseña_positiva.
#

conteo_scores = df["review_score"].value_counts().sort_index()
colores_scores = [PALETA_CATEGORICA[7] if s <= 2 else (PALETA_CATEGORICA[3] if s == 3 else PALETA_CATEGORICA[2])
                   for s in conteo_scores.index]

fig, ax = plt.subplots(figsize=(7, 4.5))
barras = conteo_scores.plot(kind="bar", color=colores_scores, ax=ax)
ax.set_ylim(0, conteo_scores.max() * 1.15)
ax.bar_label(barras.containers[0], fmt="%.0f", padding=3)
ax.set_title("Distribución de review_score")
plt.tight_layout()
_fig_i += 1
plt.savefig(FIGS_DIR / f"fig_{_fig_i:02d}.png", dpi=110, bbox_inches="tight")
plt.show()

#
# **Insight:** la distribución está claramente sesgada hacia 5 estrellas, con un segundo grupo
# más chico en 1 estrella — hay pocas reseñas "tibias" (2 o 3), lo cual respalda el criterio usado en
# reseña_positiva de dejar el puntaje 3 como ambiguo (NaN) en vez de forzarlo a un lado.
#

#
# **Qué hacemos:** graficamos dos histogramas: la distribución del tiempo de entrega en días, y
# la distribución de la diferencia entre la entrega real y la estimada, cada uno con su media marcada
# y anotada.
#
# **Para qué:** para entender la variabilidad real de los tiempos de entrega y cuán seguido (y por
# cuánto) Olist se pasa de la fecha que le prometió al cliente.
#

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

media1 = df["tiempo_entrega_dias"].mean()
sns.histplot(df["tiempo_entrega_dias"].dropna(), bins=40, color=OLIST_BLUE, ax=axes[0])
axes[0].axvline(media1, color="black", linestyle="--", linewidth=1)
axes[0].text(media1, axes[0].get_ylim()[1] * 0.92, f" media={media1:.1f}", fontsize=9)
axes[0].set_title("Tiempo de entrega (días)")

media2 = df["dias_vs_estimado"].mean()
sns.histplot(df["dias_vs_estimado"].dropna(), bins=40, color=PALETA_CATEGORICA[1], ax=axes[1])
axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
axes[1].axvline(media2, color=OLIST_BLUE_DARK, linestyle="--", linewidth=1)
axes[1].text(media2, axes[1].get_ylim()[1] * 0.92, f" media={media2:.1f}", fontsize=9)
axes[1].set_title("Entrega real vs. estimada (días)")
plt.tight_layout()
_fig_i += 1
plt.savefig(FIGS_DIR / f"fig_{_fig_i:02d}.png", dpi=110, bbox_inches="tight")
plt.show()

#
# **Insight:** la mayoría de los pedidos se entrega bastante antes de la fecha estimada (la
# media de `dias_vs_estimado` es negativa), lo que sugiere que Olist calcula sus estimaciones con un
# margen conservador — y por eso, cuando ese margen no alcanza, el atraso resultante golpea más la
# satisfacción (ver sección 9).
#

#
# **Qué hacemos:** comparamos, con un boxplot, la distribución de review_score entre los pedidos
# entregados a tiempo y los demorados, con la mediana de cada grupo anotada.
#
# **Para qué:** es la primera mirada visual a la hipótesis 1 (demora → menor satisfacción), antes de
# contrastarla formalmente en la sección 9.
#

fig, ax = plt.subplots(figsize=(7, 4.5))
sns.boxplot(
    data=df.dropna(subset=["envio_demorado"]),
    x="envio_demorado", y="review_score",
    palette=[OLIST_BLUE, PALETA_CATEGORICA[7]], ax=ax,
)
medianas = df.dropna(subset=["envio_demorado"]).groupby("envio_demorado")["review_score"].median()
for i, val in enumerate(medianas):
    ax.text(i, val + 0.15, f"mediana={val:.1f}", ha="center", fontsize=9, fontweight="bold")
ax.set_xticklabels(["A tiempo (0)", "Demorado (1)"])
ax.set_title("review_score según envio_demorado")
plt.tight_layout()
_fig_i += 1
plt.savefig(FIGS_DIR / f"fig_{_fig_i:02d}.png", dpi=110, bbox_inches="tight")
plt.show()

df.groupby("envio_demorado")["review_score"].mean().round(2)

#
# **Insight:** la mediana de review_score entre los envíos demorados es notoriamente más baja
# que entre los que llegaron a tiempo, con menos dispersión hacia el 5 — visualmente ya se anticipa
# que la hipótesis 1 se va a confirmar.
#

#
# **Qué hacemos:** graficamos el medio de pago principal (torta, ya con % incluido) y la
# cantidad de cuotas elegidas (histograma, con la media marcada).
#
# **Para qué:** para entender cómo pagan los clientes de Olist, información relevante para la
# hipótesis 7 sobre si el medio de pago afecta la satisfacción.
#

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
conteo_pago = df["payment_type_principal"].value_counts()
axes[0].pie(conteo_pago, labels=conteo_pago.index, autopct="%1.0f%%",
            colors=PALETA_CATEGORICA, startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1})
axes[0].set_title("Medio de pago principal")

media_cuotas = df["payment_installments_max"].mean()
sns.histplot(df["payment_installments_max"].dropna(), bins=range(1, 25), color=OLIST_BLUE, ax=axes[1])
axes[1].axvline(media_cuotas, color="black", linestyle="--", linewidth=1)
axes[1].text(media_cuotas, axes[1].get_ylim()[1] * 0.92, f" media={media_cuotas:.1f}", fontsize=9)
axes[1].set_title("Cuotas elegidas")
plt.tight_layout()
_fig_i += 1
plt.savefig(FIGS_DIR / f"fig_{_fig_i:02d}.png", dpi=110, bbox_inches="tight")
plt.show()

#
# **Insight:** la tarjeta de crédito domina como medio de pago, y dentro de esa opción la mayoría
# elige pagar en pocas cuotas — el crédito en cuotas no parece ser, por ahora, un factor determinante
# de insatisfacción por sí solo.
#

#
# **Qué hacemos:** graficamos, con boxplots, la distribución de precio dentro de cada una de las
# 10 categorías con más ítems vendidos, con la mediana de cada categoría anotada.
#
# **Para qué:** para ver si las categorías de mayor volumen (gráfico anterior) son también las de
# mayor o menor ticket promedio — esa distinción es relevante para decisiones de pricing/marketing.
#

top10 = df["product_category_name_english"].value_counts().head(10).index
df_top10 = df[df["product_category_name_english"].isin(top10)]
orden = df_top10.groupby("product_category_name_english")["price"].median().sort_values().index

fig, ax = plt.subplots(figsize=(11, 5.5))
sns.boxplot(data=df_top10, y="product_category_name_english", x="price",
            order=orden, color=OLIST_BLUE, ax=ax, showfliers=False)
medianas = df_top10.groupby("product_category_name_english")["price"].median().reindex(orden)
for i, val in enumerate(medianas):
    ax.text(val, i, f" {val:.0f}", va="center", fontsize=8)
ax.set_title("Precio por categoría (top 10, sin outliers extremos)")
plt.tight_layout()
_fig_i += 1
plt.savefig(FIGS_DIR / f"fig_{_fig_i:02d}.png", dpi=110, bbox_inches="tight")
plt.show()

#
# **Insight:** las categorías de mayor volumen (belleza/salud, artículos para el hogar) tienen
# precios medianos bajos, mientras que categorías de menor volumen (por ejemplo, muebles) tienen
# tickets más altos — confirma que volumen y precio unitario son ejes distintos del negocio.
#

#
# 7. Análisis exhaustivo: logística, precio y vendedores
#
# Esta sección profundiza puntos específicos que quedaron abiertos en la sección anterior: qué
# categorías concentran las demoras, si la distancia y el tamaño/peso del producto influyen en la
# logística, cómo varía el ticket por región, cómo se relacionan entre sí las variables numéricas
# clave, y quiénes son los vendedores que más pesan (en volumen y en performance de entrega).
#

#
# **Qué hacemos:** calculamos, por categoría de producto (filtrando las que tienen al menos 30
# pedidos para no analizar categorías demasiado chicas), el % de envíos demorados y el tiempo de
# entrega promedio, y graficamos las 10 categorías con peor % de demora.
#
# **Para qué:** responde directamente qué categorías tienen mayor demora de envío — información
# concreta para priorizar dónde revisar la logística primero, en vez de tratar a todas las categorías
# por igual.
#

UMBRAL_MIN_PEDIDOS = 30

demora_categoria = df.groupby("product_category_name_english").agg(
    pedidos=("order_id", "nunique"),
    tiempo_entrega_promedio=("tiempo_entrega_dias", "mean"),
    pct_demorado=("envio_demorado", "mean"),
)
demora_categoria_filtrada = demora_categoria.query("pedidos >= @UMBRAL_MIN_PEDIDOS")
if demora_categoria_filtrada.empty:
    demora_categoria_filtrada = demora_categoria  # dataset chico: no filtramos por volumen

demora_categoria_filtrada = demora_categoria_filtrada.copy()
demora_categoria_filtrada["pct_demorado"] = (demora_categoria_filtrada["pct_demorado"] * 100).round(1)
demora_categoria_filtrada["tiempo_entrega_promedio"] = demora_categoria_filtrada["tiempo_entrega_promedio"].round(1)
peores_categorias = demora_categoria_filtrada.sort_values("pct_demorado", ascending=False).head(10)

fig, ax = plt.subplots(figsize=(9, 5.5))
barras = peores_categorias["pct_demorado"].sort_values().plot(kind="barh", color=PALETA_CATEGORICA[7], ax=ax)
ax.set_xlim(0, peores_categorias["pct_demorado"].max() * 1.15)
ax.bar_label(barras.containers[0], fmt="%.1f%%", padding=3)
ax.set_title("Categorías con mayor % de envíos demorados (≥30 pedidos)")
plt.tight_layout()
_fig_i += 1
plt.savefig(FIGS_DIR / f"fig_{_fig_i:02d}.png", dpi=110, bbox_inches="tight")
plt.show()

peores_categorias

#
# **Insight:** las categorías con peor % de demora no son necesariamente las de mayor volumen
# (sección 6) — hay categorías chicas o medianas que logísticamente rinden peor, probablemente por
# tratarse de productos más voluminosos/pesados o despachados desde vendedores más alejados (se
# retoma en las próximas dos bloques).
#

#
# **Qué hacemos:** comparamos, con un boxplot, la distancia cliente-vendedor (`distancia_km`)
# entre pedidos a tiempo y demorados, con la mediana de cada grupo anotada, y calculamos la
# correlación entre distancia y tiempo de entrega.
#
# **Para qué:** responde si la distancia geográfica entre cliente y vendedor influye en la demora —
# yendo más allá del proxy Sudeste/Resto del país (sección 9) con una distancia real en kilómetros.
#

fig, ax = plt.subplots(figsize=(7, 4.5))
datos_dist = df.dropna(subset=["envio_demorado", "distancia_km"])
sns.boxplot(data=datos_dist, x="envio_demorado", y="distancia_km",
            palette=[OLIST_BLUE, PALETA_CATEGORICA[7]], ax=ax, showfliers=False)
medianas_dist = datos_dist.groupby("envio_demorado")["distancia_km"].median()
for i, val in enumerate(medianas_dist):
    ax.text(i, val + val * 0.05, f"mediana={val:.0f} km", ha="center", fontsize=9, fontweight="bold")
ax.set_xticklabels(["A tiempo (0)", "Demorado (1)"])
ax.set_ylabel("Distancia cliente-vendedor (km)")
ax.set_title("Distancia cliente-vendedor según envio_demorado")
plt.tight_layout()
_fig_i += 1
plt.savefig(FIGS_DIR / f"fig_{_fig_i:02d}.png", dpi=110, bbox_inches="tight")
plt.show()

corr_dist = df[["distancia_km", "tiempo_entrega_dias"]].corr().iloc[0, 1]
print(f"Correlación distancia_km vs. tiempo_entrega_dias: {corr_dist:.2f}")

#
# **Insight:** los envíos demorados tienen una distancia mediana mayor que los que llegan a
# tiempo, y la correlación entre distancia y tiempo de entrega es positiva y considerable — la
# distancia real (no solo la región) es un factor logístico concreto detrás de las demoras.
#

#
# **Qué hacemos:** graficamos peso del producto vs. tiempo de entrega, y volumen del producto
# vs. flete_ratio, sobre una muestra de pedidos, y calculamos la matriz de correlación entre peso,
# volumen y las variables de logística/costo.
#
# **Para qué:** responde si el peso y el volumen de los productos influyen en si se entregan a tiempo
# y en cuánto se paga de flete en relación al precio.
#

muestra_pv = df.sample(n=min(3000, len(df)), random_state=42)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.scatterplot(data=muestra_pv, x="product_weight_g", y="tiempo_entrega_dias",
                 alpha=0.3, color=OLIST_BLUE, ax=axes[0])
axes[0].set_title("Peso del producto vs. tiempo de entrega")

sns.scatterplot(data=muestra_pv, x="volumen_cm3", y="flete_ratio",
                 alpha=0.3, color=PALETA_CATEGORICA[1], ax=axes[1])
axes[1].set_ylim(0, df["flete_ratio"].quantile(0.95))
axes[1].set_title("Volumen del producto vs. flete_ratio")
plt.tight_layout()
_fig_i += 1
plt.savefig(FIGS_DIR / f"fig_{_fig_i:02d}.png", dpi=110, bbox_inches="tight")
plt.show()

print("Correlaciones:")
print(df[["product_weight_g", "volumen_cm3", "tiempo_entrega_dias", "envio_demorado", "flete_ratio"]]
      .corr().round(2))

#
# **Insight:** el peso y el volumen tienen una correlación baja con el tiempo de entrega y con
# si el envío se demora o no (el tamaño físico del paquete no parece ser un driver fuerte de la
# demora), pero sí tienen una correlación más alta con `flete_ratio`: productos más pesados/grandes
# pagan proporcionalmente más flete, lo cual tiene sentido de negocio y valida que la variable está
# bien construida.
#

#
# **Qué hacemos:** calculamos el ticket promedio (`price`) por estado, graficando el top 10, y
# por región (Sudeste vs. resto del país).
#
# **Para qué:** responde cómo varía el gasto de ticket por estado/región — información relevante para
# estrategias de pricing o marketing diferenciadas por zona geográfica.
#

ticket_estado = df.groupby("customer_state")["price"].mean().sort_values(ascending=False).head(10)

fig, ax = plt.subplots(figsize=(9, 5.5))
barras = ticket_estado.sort_values().plot(kind="barh", color=OLIST_BLUE, ax=ax)
ax.set_xlim(0, ticket_estado.max() * 1.15)
ax.bar_label(barras.containers[0], fmt="R$ %.0f", padding=3)
ax.set_title("Ticket promedio (price) por estado, top 10")
plt.tight_layout()
_fig_i += 1
plt.savefig(FIGS_DIR / f"fig_{_fig_i:02d}.png", dpi=110, bbox_inches="tight")
plt.show()

ticket_region = df.groupby("region_sudeste")["price"].mean().round(2)
print("Ticket promedio por región:")
print(ticket_region)

#
# **Insight:** el ticket promedio no está claramente concentrado en el eje Sudeste — hay estados
# fuera de esa región con ticket promedio más alto, lo que sugiere que la brecha logística (sección 9)
# no viene acompañada de una brecha de gasto: los clientes de otras regiones compran productos de
# valor similar o mayor, pese a esperar más y pagar más flete.
#

#
# **Qué hacemos:** armamos la matriz de correlación entre las variables numéricas clave del
# proyecto: precio, flete, flete_ratio, tiempo de entrega, distancia, volumen, peso, review_score e
# items por pedido.
#
# **Para qué:** da una vista compacta de qué variables se mueven juntas antes de leer hipótesis por
# hipótesis — sirve como mapa de ruta del resto del análisis y para detectar relaciones que no eran
# obvias a simple vista.
#

numericas = [
    "price", "freight_value", "flete_ratio", "tiempo_entrega_dias", "dias_vs_estimado",
    "review_score", "distancia_km", "volumen_cm3", "product_weight_g", "items_por_pedido",
]
corr = df[numericas].corr()

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", center=0, ax=ax)
ax.set_title("Matriz de correlación entre variables numéricas clave")
plt.tight_layout()
_fig_i += 1
plt.savefig(FIGS_DIR / f"fig_{_fig_i:02d}.png", dpi=110, bbox_inches="tight")
plt.show()

#
# **Insight:** la correlación más fuerte del bloque logístico es distancia_km con
# tiempo_entrega_dias, seguida de freight_value con volumen/peso — review_score tiene correlaciones
# bajas con casi todas las variables numéricas de forma lineal, lo que confirma que su relación con la
# demora (sección 9) es más una diferencia de grupos (a tiempo vs. demorado) que una relación lineal
# continua.
#

#
# **Qué hacemos:** armamos dos rankings de vendedores (con al menos 20 pedidos, para dejar afuera
# a los que tienen muy poco volumen): los 10 de mayor volumen de pedidos, y los 10 con peor % de
# envíos demorados.
#
# **Para qué:** es un primer acercamiento simple (sin llegar a clustering, que queda para la etapa de
# modelado) a qué vendedores concentran el negocio y cuáles tienen peor desempeño logístico —
# información directamente accionable para priorizar a quién contactar primero.
#

UMBRAL_MIN_PEDIDOS_VENDEDOR = 20

ranking_vendedores = df.groupby("seller_id").agg(
    pedidos=("order_id", "nunique"),
    tiempo_entrega_promedio=("tiempo_entrega_dias", "mean"),
    pct_demorado=("envio_demorado", "mean"),
)
ranking_vendedores_filtrado = ranking_vendedores.query("pedidos >= @UMBRAL_MIN_PEDIDOS_VENDEDOR")
if ranking_vendedores_filtrado.empty:
    ranking_vendedores_filtrado = ranking_vendedores  # dataset chico: no filtramos por volumen

ranking_vendedores = ranking_vendedores_filtrado.copy()
ranking_vendedores["pct_demorado"] = (ranking_vendedores["pct_demorado"] * 100).round(1)
ranking_vendedores["tiempo_entrega_promedio"] = ranking_vendedores["tiempo_entrega_promedio"].round(1)

print("Top 10 vendedores por volumen de pedidos:")
print(ranking_vendedores.sort_values("pedidos", ascending=False).head(10))
print()
print("Top 10 vendedores con peor % de envíos demorados (≥20 pedidos):")
print(ranking_vendedores.sort_values("pct_demorado", ascending=False).head(10))

#
# **Insight:** los vendedores con mayor volumen no son los mismos que los que tienen peor % de
# demora — es decir, el problema logístico no está concentrado únicamente en los vendedores más
# grandes, lo que sugiere que una intervención pareja (capacitación, SLA, cambio de transportista) le
# serviría a un conjunto de vendedores más amplio que solo los "top".
#

#
# **Qué hacemos:** calculamos el ticket promedio (`price`) agrupado por la cantidad de cuotas
# máxima elegida en el pago (`payment_installments_max`), y lo graficamos.
#
# **Para qué:** para ver si las compras de mayor valor se financian en más cuotas, algo esperable pero
# que conviene confirmar con los datos antes de asumirlo.
#

cuotas_ticket = df.groupby("payment_installments_max")["price"].mean().dropna()

fig, ax = plt.subplots(figsize=(10, 4.5))
barras = cuotas_ticket.plot(kind="bar", color=OLIST_BLUE, ax=ax)
ax.bar_label(barras.containers[0], fmt="%.0f", padding=2, fontsize=7, rotation=90)
ax.set_ylim(0, cuotas_ticket.max() * 1.25)
ax.set_ylabel("Ticket promedio (price)")
ax.set_title("Ticket promedio según cantidad de cuotas elegidas")
plt.tight_layout()
_fig_i += 1
plt.savefig(FIGS_DIR / f"fig_{_fig_i:02d}.png", dpi=110, bbox_inches="tight")
plt.show()

#
# **Insight:** el ticket promedio sube de forma bastante consistente a medida que aumenta la
# cantidad de cuotas — confirma que el financiamiento en cuotas se usa mayormente para compras de
# mayor valor, no de forma pareja en todos los rangos de precio.
#

#
# **Qué hacemos:** calculamos la facturación total (`price`) por categoría de producto y el %
# acumulado que representan, ordenadas de mayor a menor, y graficamos un Pareto de las 15 categorías
# top.
#
# **Para qué:** para ver qué tan concentrada está la facturación en pocas categorías (lógica 80/20),
# información relevante para decidir dónde enfocar esfuerzos comerciales o de negociación con
# vendedores.
#

facturacion_categoria = df.groupby("product_category_name_english")["price"].sum().sort_values(ascending=False)
pct_acumulado = facturacion_categoria.cumsum() / facturacion_categoria.sum() * 100

fig, ax1 = plt.subplots(figsize=(11, 5))
facturacion_categoria.head(15).plot(kind="bar", color=OLIST_BLUE, ax=ax1)
ax1.set_ylabel("Facturación (price)")
ax1.tick_params(axis="x", rotation=75)

ax2 = ax1.twinx()
pct_acumulado.head(15).plot(color=PALETA_CATEGORICA[1], marker="o", ax=ax2)
ax2.set_ylabel("% acumulado")
ax2.set_ylim(0, 105)
for x, y in zip(range(15), pct_acumulado.head(15).values):
    ax2.annotate(f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(0, 8),
                 ha="center", fontsize=7, color=PALETA_CATEGORICA[1])

ax1.set_title("Pareto de facturación por categoría (top 15)")
plt.tight_layout()
_fig_i += 1
plt.savefig(FIGS_DIR / f"fig_{_fig_i:02d}.png", dpi=110, bbox_inches="tight")
plt.show()

print(f"Las 5 categorías top representan el {pct_acumulado.iloc[4]:.1f}% de la facturación total.")

#
# **Insight:** la facturación está bastante repartida entre varias categorías —no hay una
# concentración extrema tipo "2-3 categorías son todo el negocio"—, lo cual es consistente con un
# marketplace generalista y no con una tienda especializada en pocos rubros.
#

#
# 8. Mapa geográfico
#
# **Qué hacemos:** construimos un mapa interactivo de puntos (Plotly) coloreado por review_score sobre
# una muestra de 5.000 clientes, usando `customer_lat`/`customer_lng`.
#
# **Para qué:** para ver si hay un patrón geográfico en la satisfacción del cliente, más allá de lo
# que muestran los promedios agregados por estado.
#

import plotly.express as px

validos = df.dropna(subset=["customer_lat", "customer_lng", "review_score"])
muestra = validos.sample(n=min(5000, len(validos)), random_state=42)

try:
    fig = px.scatter_map(
        muestra, lat="customer_lat", lon="customer_lng",
        color="review_score", color_continuous_scale="Blues", zoom=3,
        hover_data=["customer_state", "customer_city", "envio_demorado", "distancia_km"],
        title="Ubicación de clientes (muestra) por review_score",
    )
except AttributeError:
    fig = px.scatter_mapbox(
        muestra, lat="customer_lat", lon="customer_lng",
        color="review_score", color_continuous_scale="Blues", zoom=3,
        mapbox_style="open-street-map",
        hover_data=["customer_state", "customer_city", "envio_demorado", "distancia_km"],
        title="Ubicación de clientes (muestra) por review_score",
    )
fig.show()

#
# **Insight:** los clientes se concentran fuertemente en el eje Sudeste (São Paulo, Río de
# Janeiro, Minas Gerais), la misma zona donde se concentran los vendedores.
#

#
# **Qué hacemos:** construimos un mapa de calor (Folium) de densidad de pedidos sobre el dataset
# completo (no una muestra).
#
# **Para qué:** complementa el mapa anterior con una vista de volumen (dónde hay más pedidos, no solo
# dónde están coloreados por puntaje), sin el límite de 5.000 puntos.
#

import folium
from folium.plugins import HeatMap

puntos = df.dropna(subset=["customer_lat", "customer_lng"])[["customer_lat", "customer_lng"]]
mapa_calor = folium.Map(location=[-15.0, -50.0], zoom_start=4, tiles="OpenStreetMap")
HeatMap(puntos.values.tolist(), radius=10, blur=8).add_to(mapa_calor)
mapa_calor

#
# **Insight:** el mapa de calor confirma la misma concentración geográfica que el mapa de
# puntos, ahora sin el sesgo de tomar solo una muestra de 5.000 clientes.
#

#
# **Qué hacemos:** armamos una tabla resumen por estado (los 10 con más pedidos), con cantidad de
# pedidos, review_score promedio y % de envíos demorados.
#
# **Para qué:** para respaldar en números lo que se ve en los mapas, y tener los datos concretos que
# se usan en la hipótesis 3 (sección 9) sobre la brecha geográfica.
#

resumen_estado = (
    df.groupby("customer_state")
    .agg(pedidos=("order_id", "nunique"),
         review_score_prom=("review_score", "mean"),
         pct_demorado=("envio_demorado", "mean"))
    .sort_values("pedidos", ascending=False)
    .head(10)
    .round(2)
)
resumen_estado["pct_demorado"] = (resumen_estado["pct_demorado"] * 100).round(1)
resumen_estado

#
# **Insight:** los estados fuera del eje Sudeste, aunque tienen menos pedidos, muestran un
# review_score promedio algo más bajo y un % de envíos demorados más alto que SP/RJ/MG — la distancia
# a los vendedores parece pesar en la experiencia de entrega.
#

#
# 9. NLP sobre las reseñas (portugués)
#
# **Qué hacemos:** definimos una función de limpieza de texto en portugués (minúsculas, sin tildes,
# sin puntuación/números) y cargamos una lista de stopwords en portugués (con una lista de respaldo
# propia por si no hay conexión a internet para bajar las de NLTK). Con eso armamos tres "bolsas de
# palabras": todas las reseñas con comentario, solo las negativas (score 1-2) y solo las positivas
# (score 4-5).
#
# **Para qué:** es el paso previo necesario para poder generar nubes de palabras y rankings de
# frecuencia que tengan sentido (sin stopwords ni ruido de mayúsculas/tildes/puntuación).
#

def limpiar_texto(texto):
    if pd.isna(texto) or texto == "":
        return ""
    texto = str(texto).lower()
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("utf-8")
    texto = re.sub(r"[^a-z\s]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()

STOPWORDS_PT_RESPALDO = (
    "de a o que e do da em um para com nao uma os no se na por mais as dos como mas "
    "ao ele das a sua ou quando muito nos ja eu tambem so pelo pelas pela ate isso "
    "ela entre depois sem mesmo aos seus quem nas me esse eles voce essa num nem "
    "suas meu as minha numa pelos elas qual sera nos tenho lhe deles essas esses "
    "pelas este fosse dela tu te voces vos lhes meus minhas teu tua teus tuas "
    "nosso nossa nossos nossas dele"
).split()

try:
    import nltk
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    STOPWORDS_PT = set(stopwords.words("portuguese"))
except Exception:
    STOPWORDS_PT = set(STOPWORDS_PT_RESPALDO)

df["review_limpia"] = df["review_comment_message"].apply(limpiar_texto)

def armar_texto(serie, sw, largo_min=2):
    palabras = " ".join(serie).split()
    return " ".join(p for p in palabras if p not in sw and len(p) > largo_min)

con_comentario = df["tiene_comentario"] == 1
texto_general = armar_texto(df.loc[con_comentario, "review_limpia"], STOPWORDS_PT)
texto_neg = armar_texto(df.loc[con_comentario & (df["review_score"] <= 2), "review_limpia"], STOPWORDS_PT)
texto_pos = armar_texto(df.loc[con_comentario & (df["review_score"] >= 4), "review_limpia"], STOPWORDS_PT)

#
# **Qué hacemos:** generamos la nube de palabras general, con todas las reseñas que tienen
# comentario.
#
# **Para qué:** da una primera foto visual de qué se comenta más, en general, sobre las compras en
# Olist, antes de separar por sentimiento.
#

from wordcloud import WordCloud

nube_general = WordCloud(width=1000, height=500, background_color="white",
                          colormap="Blues", max_words=100, collocations=False).generate(texto_general)
plt.figure(figsize=(12, 6))
plt.imshow(nube_general, interpolation="bilinear")
plt.axis("off")
plt.title("Nube de palabras general", color=OLIST_BLUE_DARK)
_fig_i += 1
plt.savefig(FIGS_DIR / f"fig_{_fig_i:02d}.png", dpi=110, bbox_inches="tight")
plt.show()

#
# **Insight:** ya en la nube general aparecen con fuerza palabras vinculadas a entrega/producto
# ("entrega", "produto", "prazo"), lo cual anticipa que la logística también domina la conversación
# positiva y negativa por igual — la matiz aparece al separar por sentimiento (próximo bloque).
#

#
# **Qué hacemos:** generamos dos nubes de palabras separadas: una con las reseñas negativas
# (score 1-2) y otra con las positivas (score 4-5).
#
# **Para qué:** para contrastar el vocabulario de clientes insatisfechos vs. satisfechos y así
# contrastar la hipótesis 2 (el contenido de las reseñas negativas está dominado por menciones a
# demoras/entrega).
#

fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

nube_neg = WordCloud(width=800, height=500, background_color="white",
                      colormap="Reds", max_words=80, collocations=False).generate(texto_neg)
axes[0].imshow(nube_neg, interpolation="bilinear")
axes[0].axis("off")
axes[0].set_title("Reseñas negativas (score 1-2)")

nube_pos = WordCloud(width=800, height=500, background_color="white",
                      colormap="Blues", max_words=80, collocations=False).generate(texto_pos)
axes[1].imshow(nube_pos, interpolation="bilinear")
axes[1].axis("off")
axes[1].set_title("Reseñas positivas (score 4-5)")

plt.tight_layout()
_fig_i += 1
plt.savefig(FIGS_DIR / f"fig_{_fig_i:02d}.png", dpi=110, bbox_inches="tight")
plt.show()

#
# **Insight:** en las negativas predominan palabras de queja sobre entrega/atraso/producto
# ("prazo", "atraso", "chegou", "recebi"), mientras que en las positivas predominan elogios y menciones
# al cumplimiento del plazo — la misma conclusión que ya se veía con envio_demorado, ahora confirmada
# desde el texto libre.
#

#
# **Qué hacemos:** armamos un ranking de las 15 palabras más frecuentes dentro de las reseñas
# negativas.
#
# **Para qué:** complementa la nube de palabras con un número concreto de frecuencia por palabra, más
# fácil de citar en el informe que una imagen.
#

top_negativas = pd.DataFrame(Counter(texto_neg.split()).most_common(15), columns=["palabra", "frecuencia"])
top_negativas

#
# **Insight:** las palabras más frecuentes en las reseñas negativas están directamente
# vinculadas al proceso de entrega, reforzando que la logística —y no el producto en sí— es la
# principal fuente de insatisfacción expresada por escrito.
#

#
# 10. Contraste de hipótesis
#
# **Qué hacemos:** comparamos el review_score promedio entre pedidos a tiempo y demorados, y además
# corremos un test de Mann-Whitney (no paramétrico, apropiado porque review_score no es una variable
# continua ni normal) para confirmar que la diferencia observada es estadísticamente significativa y
# no producto del azar.
#
# **Para qué:** contrasta formalmente la hipótesis 1 ("la demora en la entrega reduce la satisfacción
# del cliente"), yendo un paso más allá del gráfico descriptivo de la sección 6.
#

print(df.groupby("envio_demorado")["review_score"].mean().round(2))

a_tiempo = df.loc[df["envio_demorado"] == 0, "review_score"].dropna()
demorado = df.loc[df["envio_demorado"] == 1, "review_score"].dropna()
u_stat, p_valor = stats.mannwhitneyu(a_tiempo, demorado, alternative="greater")
print(f"\nMann-Whitney U: p-valor = {p_valor:.2e}")

#
# **Insight:** el p-valor es prácticamente cero, muy por debajo del umbral habitual de 0,05:
# la diferencia de satisfacción entre envíos a tiempo y demorados no es casualidad — es estadísticamente
# significativa. Hipótesis 1: **confirmada**.
#

#
# **Qué hacemos:** comparamos flete promedio, tiempo de entrega promedio y % de envíos demorados
# entre la región Sudeste y el resto del país, y corremos el mismo tipo de test estadístico sobre el
# tiempo de entrega entre ambos grupos.
#
# **Para qué:** contrasta formalmente la hipótesis 3 ("la distancia geográfica al eje Sudeste está
# asociada a mayores tiempos de entrega y flete más caro"), la que se venía anticipando en el mapa y el
# resumen por estado de la sección 8.
#

comparacion = df.groupby("region_sudeste").agg(
    flete_promedio=("freight_value", "mean"),
    tiempo_entrega_promedio=("tiempo_entrega_dias", "mean"),
    pct_demorado=("envio_demorado", "mean"),
).round(2)
comparacion["pct_demorado"] = (comparacion["pct_demorado"] * 100).round(1)
print(comparacion)

sudeste = df.loc[df["region_sudeste"] == "Sudeste", "tiempo_entrega_dias"].dropna()
resto = df.loc[df["region_sudeste"] == "Resto del país", "tiempo_entrega_dias"].dropna()
u_stat, p_valor = stats.mannwhitneyu(sudeste, resto, alternative="less")
print(f"\nMann-Whitney U (tiempo de entrega, Sudeste < Resto): p-valor = {p_valor:.2e}")

#
# **Insight:** los clientes fuera del eje Sudeste pagan más flete, esperan más días y tienen
# mayor proporción de envíos demorados, y el p-valor del test confirma que la diferencia en tiempo de
# entrega no es azar. Hipótesis 3: **confirmada**.
#

#
# **Qué hacemos:** calculamos la correlación de Spearman (no paramétrica, apropiada para
# relaciones no necesariamente lineales) entre `distancia_km` y `tiempo_entrega_dias`, con su p-valor.
#
# **Para qué:** contrasta formalmente una hipótesis adicional que surge del análisis exhaustivo de la
# sección 7: la distancia real cliente-vendedor (no solo la región Sudeste/resto) está asociada a
# mayores tiempos de entrega. Complementa la hipótesis 3 con una variable continua en vez de una
# categórica.
#

rho, p_valor = stats.spearmanr(df["distancia_km"], df["tiempo_entrega_dias"], nan_policy="omit")
print(f"Correlación de Spearman (distancia_km vs. tiempo_entrega_dias): rho={rho:.2f}, p-valor={p_valor:.2e}")

#
# **Insight:** la correlación es positiva y el p-valor prácticamente cero: a mayor distancia
# entre cliente y vendedor, mayor tiempo de entrega, y la relación no es casualidad. Esto confirma,
# con una métrica continua y no solo con la partición Sudeste/resto, que la distancia geográfica real
# es un driver logístico medible. Hipótesis adicional: **confirmada**.
#

#
# ---
# Ver el informe funcional (`.docx`) para la explicación de negocio completa, las conclusiones y los
# próximos pasos. El modelado predictivo (clasificación de reseñas bajas, segmentación de vendedores)
# queda para una etapa posterior, una vez validado este análisis exploratorio.
#
