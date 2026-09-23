#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Olist E-Commerce - Modelos predictivos
# Clasificacion (envio demorado) + clustering (segmentacion de vendedores).
# Se puede correr directo con: python modelos_predictivos.py
#


#
# Olist E-Commerce — Modelos predictivos
#
# Este script es la segunda etapa del proyecto integrador, sobre la base del EDA (`eda_completo.py`).
# Desarrolla los dos modelos que pide la consigna, elegidos y justificados en el informe funcional:
#
# 1. **Clasificación supervisada multiclase** — predecir si una compra va a terminar en reseña
#    **negativa, neutral o positiva** (`reseña_categoria`), con Decision Tree, Random Forest y XGBoost,
#    comparados por accuracy y F1 macro. A propósito, no usamos `envio_demorado`/`dias_vs_estimado` como
#    feature de entrada (aunque sabemos que son las que más pesan en la reseña) — la idea es que el
#    modelo sea útil *antes* de que el pedido llegue, no que repita, con otro nombre, la correlación que
#    ya confirmamos en el EDA.
# 2. **Clustering no supervisado** — segmentar vendedores por comportamiento logístico y de negocio
#    (agregando también reseña, precio y variedad de categorías) con K-Means (elbow + silhouette para
#    elegir la cantidad de clusters).
#
# Igual que el script de EDA, no depende de token de Kaggle: reutiliza los mismos CSV crudos de
# `data/raw/`. Cada bloque tiene su comentario de **qué hacemos / para qué**, y su **insight** cuando
# corresponde. Los resultados de ambos modelos se guardan en `modelos_output/` para que la app de
# Streamlit los pueda graficar sin tener que reentrenar nada.
#

#
# 0. Configuración inicial
#
# **Qué hacemos:** importamos librerías de datos (pandas/numpy), de visualización (matplotlib/seaborn),
# y de modelado (scikit-learn para Decision Tree, Random Forest, K-Means, métricas y preprocesamiento;
# xgboost para el gradient boosting), y definimos la paleta de colores de Olist.
#
# **Para qué:** mismo criterio que el script de EDA: imports y paleta en un solo lugar, para que todo
# el script sea consistente.
#

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

OLIST_BLUE = "#0A4EE4"
OLIST_BLUE_DARK = "#0D366B"
OLIST_GRAY = "#52514E"
PALETA_CATEGORICA = ["#0A4EE4", "#eb6834", "#1baf7a", "#eda100",
                     "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SUDESTE = {"SP", "RJ", "MG", "ES"}

sns.set_theme(style="whitegrid", rc={
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": OLIST_GRAY,
    "text.color": "#0b0b0b", "xtick.color": OLIST_GRAY, "ytick.color": OLIST_GRAY,
})
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.titleweight"] = "bold"

RAW_DIR = Path("data/raw")
CSV_UNIFICADO = Path("data/olist_dataset_unificado.csv")
OUT_DIR = Path("modelos_output")
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

FIGS_DIR = Path("figuras_modelos")
FIGS_DIR.mkdir(exist_ok=True)
_fig_i = 0

#
# 1. Preparación de datos
#
# **Qué hacemos:** cargamos el dataset unificado (el mismo `data/olist_dataset_unificado.csv` que
# genera el script de EDA — si no existe todavía, este script también sabe unificar las 9 tablas
# crudas) y repetimos exactamente la misma limpieza y el mismo feature engineering del EDA, para que
# este script sea autocontenido y no dependa de haber corrido antes el otro.
#
# **Para qué:** los modelos necesitan las mismas variables (`envio_demorado`, `distancia_km`,
# `volumen_cm3`, etc.) que ya definimos y validamos en el EDA — repetir el mismo pipeline evita
# inconsistencias entre ambos scripts.
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

    payments_agg = payments.groupby("order_id").agg(
        payment_value_total=("payment_value", "sum"),
        payment_installments_max=("payment_installments", "max"),
        payment_type_principal=(
            "payment_type", lambda x: x.value_counts().idxmax() if x.notna().any() else None,
        ),
    ).reset_index()

    reviews_agg = (
        reviews.sort_values("review_creation_date").groupby("order_id").last()
        .reset_index()[["order_id", "review_score", "review_comment_message"]]
    )

    products = products.merge(cat_translation, on="product_category_name", how="left")

    df = items.merge(orders, on="order_id", how="left")
    df = df.merge(products, on="product_id", how="left")
    df = df.merge(sellers, on="seller_id", how="left")
    df = df.merge(customers, on="customer_id", how="left")
    df = df.merge(payments_agg, on="order_id", how="left")
    df = df.merge(reviews_agg, on="order_id", how="left")

    geo_prom = geolocation.groupby("geolocation_zip_code_prefix").agg(
        lat=("geolocation_lat", "mean"), lng=("geolocation_lng", "mean")
    ).reset_index()
    df = df.merge(
        geo_prom.rename(columns={"geolocation_zip_code_prefix": "customer_zip_code_prefix",
                                  "lat": "customer_lat", "lng": "customer_lng"}),
        on="customer_zip_code_prefix", how="left")
    df = df.merge(
        geo_prom.rename(columns={"geolocation_zip_code_prefix": "seller_zip_code_prefix",
                                  "lat": "seller_lat", "lng": "seller_lng"}),
        on="seller_zip_code_prefix", how="left")

    CSV_UNIFICADO.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_UNIFICADO, index=False)

df = pd.read_csv(CSV_UNIFICADO)

# --- limpieza (idéntica al EDA) ---
COLUMNAS_FECHA = [
    "shipping_limit_date", "order_purchase_timestamp", "order_approved_at",
    "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_delivery_date",
]
df = df.drop_duplicates().reset_index(drop=True)
for col in COLUMNAS_FECHA:
    df[col] = pd.to_datetime(df[col], errors="coerce")

df["product_category_name_english"] = df["product_category_name_english"].fillna("sin_categoria").str.strip()
df["tiene_comentario"] = df["review_comment_message"].notna().astype(int)
df["review_comment_message"] = df["review_comment_message"].fillna("")

for prefijo in ["customer", "seller"]:
    df[f"{prefijo}_lat"] = pd.to_numeric(df[f"{prefijo}_lat"], errors="coerce")
    df[f"{prefijo}_lng"] = pd.to_numeric(df[f"{prefijo}_lng"], errors="coerce")
    geo_valida = df[f"{prefijo}_lat"].between(-34, 6) & df[f"{prefijo}_lng"].between(-74, -32)
    df.loc[~geo_valida, [f"{prefijo}_lat", f"{prefijo}_lng"]] = np.nan

df["items_por_pedido"] = df.groupby("order_id")["order_item_id"].transform("count")
df = df.drop(columns=[c for c in [
    "product_category_name", "product_name_lenght", "product_description_lenght",
    "product_photos_qty", "order_item_id", "customer_zip_code_prefix", "seller_zip_code_prefix",
] if c in df.columns])

# --- feature engineering (idéntico al EDA) ---
df["tiempo_entrega_dias"] = (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.days
df["dias_vs_estimado"] = (df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]).dt.days
df["envio_demorado"] = np.where(
    df["order_delivered_customer_date"].isna(), np.nan,
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
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

df["distancia_km"] = haversine_km(df["customer_lat"], df["customer_lng"], df["seller_lat"], df["seller_lng"])
df["volumen_cm3"] = df["product_length_cm"] * df["product_height_cm"] * df["product_width_cm"]

print(f"Dataset listo: {df.shape[0]:,} filas, {df.shape[1]} columnas")

#
# 2. Modelo 1 — Clasificación multiclase: ¿reseña negativa, neutral o positiva?
#
# **Por qué este modelo (y por qué reemplaza al de `envio_demorado`):** en la conversación con el
# resto del equipo notamos que un compañero predijo reseña baja usando `is_late`/`promise_gap` como
# feature — y le fue mucho mejor (AUC ~0.83) porque esa feature es casi la causa directa de una mala
# reseña, no porque su modelo sea mejor. Decidimos ir por la pregunta de reseña también, pero
# manteniendo el estándar más exigente que usábamos para `envio_demorado`: **no** incluimos
# `envio_demorado`, `dias_vs_estimado` ni `tiempo_entrega_dias` como features de entrada. Así el modelo
# predice satisfacción usando solo señales que existen *antes* de que el pedido llegue (distancia,
# tamaño, categoría, historial del vendedor), que es la versión útil de esta pregunta: sirve para
# decidir, por ejemplo, a qué pedidos prestarle atención proactiva antes de que la reseña ya esté mal.
#
# Usamos 3 clases en vez de 2 (negativa/neutral/positiva, según `review_score`) para no perder la
# categoría neutral como hacíamos antes, y comparamos los mismos tres algoritmos que en la versión
# anterior: **Decision Tree**, **Random Forest** y **XGBoost** (este último con `objective="multi:softprob"`
# para que `predict_proba` devuelva las 3 probabilidades por pedido).
#

#
# **Qué hacemos:** definimos las 3 clases a partir de `review_score` (negativa: 1-2, neutral: 3,
# positiva: 4-5), armamos la tabla de features (sin las variables de demora) y separamos en train/test.
#
# **Para qué:** dejar `review_score` como única fuente del target, y no usar variables posteriores a la
# entrega como feature, es lo que hace que el modelo mida algo genuinamente útil y no una correlación
# ya conocida.
#

FEATURES_NUM = [
    "distancia_km", "volumen_cm3", "product_weight_g", "price", "freight_value",
    "flete_ratio", "items_por_pedido", "payment_installments_max",
]
FEATURES_CAT = ["product_category_name_english", "region_sudeste", "payment_type_principal"]

def clasificar_reseña_3clases(score):
    if pd.isna(score):
        return np.nan
    if score <= 2:
        return 0  # negativa
    if score == 3:
        return 1  # neutral
    return 2  # positiva

df["reseña_categoria"] = df["review_score"].apply(clasificar_reseña_3clases)

datos_modelo = df.dropna(subset=["reseña_categoria"]).copy()

X_base = datos_modelo[["seller_id"] + FEATURES_NUM + FEATURES_CAT].copy()
y = datos_modelo["reseña_categoria"].astype(int)

for col in FEATURES_NUM:
    X_base[col] = X_base[col].fillna(X_base[col].median())

X_train_base, X_test_base, y_train, y_test = train_test_split(
    X_base, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {X_train_base.shape[0]:,} filas | Test: {X_test_base.shape[0]:,} filas")
print("Distribución de clases (0=negativa, 1=neutral, 2=positiva):")
print((y.value_counts(normalize=True).sort_index() * 100).round(1))

#
# **Insight:** lo más probable es que la clase "positiva" domine el dataset (la mayoría de los
# pedidos en Olist se entregan bien) y "neutral" sea la más chica — la clase neutral suele ser, en
# cualquier problema de sentimiento, la más difícil de distinguir, porque en la práctica es una mezcla
# de "conforme" y "medio decepcionado" que no siempre se refleja igual en las features.
#

#
# **Qué hacemos:** agregamos la misma feature de **historial del vendedor** que usábamos en el
# modelo de demora (% histórico de envíos demorados y cantidad de pedidos previos), calculada solo con
# train, y codificamos las categóricas.
#
# **Para qué:** aunque no usamos la demora del *propio* pedido como feature (sería la misma trampa que
# detectamos en el compañero), el historial *pasado* del vendedor sigue siendo información legítima y
# disponible antes de la entrega — y es razonable esperar que un vendedor con mal historial de demoras
# tienda a generar peores reseñas en general.
#

historial_vendedor = (datos_modelo.loc[X_train_base.index, "envio_demorado"]
                      .groupby(X_train_base["seller_id"]).agg(["mean", "count"]))
historial_vendedor.columns = ["historial_pct_demora_vendedor", "historial_pedidos_vendedor"]
promedio_global_train = datos_modelo.loc[X_train_base.index, "envio_demorado"].mean()

def agregar_historial(X_parte):
    X_parte = X_parte.merge(historial_vendedor, left_on="seller_id", right_index=True, how="left")
    X_parte["historial_pct_demora_vendedor"] = X_parte["historial_pct_demora_vendedor"].fillna(promedio_global_train)
    X_parte["historial_pedidos_vendedor"] = X_parte["historial_pedidos_vendedor"].fillna(0)
    return X_parte.drop(columns="seller_id")

X_train = agregar_historial(X_train_base)
X_test = agregar_historial(X_test_base)

FEATURES_NUM = FEATURES_NUM + ["historial_pct_demora_vendedor", "historial_pedidos_vendedor"]

X_train = pd.get_dummies(X_train, columns=FEATURES_CAT, drop_first=True)
X_test = pd.get_dummies(X_test, columns=FEATURES_CAT, drop_first=True)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
X = X_train

print(f"Features finales: {X_train.shape[1]}")

#
# **Qué hacemos:** entrenamos Decision Tree, Random Forest y XGBoost para las 3 clases, cada uno
# con el desbalance compensado (`class_weight="balanced"` en los árboles; pesos por muestra vía
# `compute_sample_weight` en XGBoost, que no tiene un `scale_pos_weight` para multiclase), y calculamos
# accuracy, precision/recall/F1 macro (promediando las 3 clases por igual, para que la clase neutral no
# quede invisible) y AUC one-vs-rest.
#
# **Para qué:** las métricas macro son más justas que accuracy en un problema con 3 clases desbalanceadas
# — accuracy alto se puede lograr solo por acertar siempre "positiva", igual que antes acertábamos
# "a tiempo".
#

def evaluar_modelo(modelo, X_test, y_test, nombre):
    pred = modelo.predict(X_test)
    proba = modelo.predict_proba(X_test)
    return {
        "modelo": nombre,
        "accuracy": accuracy_score(y_test, pred),
        "precision_macro": precision_score(y_test, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, pred, average="macro", zero_division=0),
        "auc_ovr": roc_auc_score(y_test, proba, multi_class="ovr", average="macro"),
    }

arbol = DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=RANDOM_STATE)
arbol.fit(X_train, y_train)
metricas_arbol = evaluar_modelo(arbol, X_test, y_test, "Decision Tree")
print(metricas_arbol)

bosque = RandomForestClassifier(
    n_estimators=200, max_depth=12, class_weight="balanced",
    random_state=RANDOM_STATE, n_jobs=-1,
)
bosque.fit(X_train, y_train)
metricas_bosque = evaluar_modelo(bosque, X_test, y_test, "Random Forest")
print(metricas_bosque)

pesos_muestra = compute_sample_weight(class_weight="balanced", y=y_train)

xgb = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    objective="multi:softprob", num_class=3, random_state=RANDOM_STATE,
    eval_metric="mlogloss",
)
xgb.fit(X_train, y_train, sample_weight=pesos_muestra)
metricas_xgb = evaluar_modelo(xgb, X_test, y_test, "XGBoost")
print(metricas_xgb)

#
# **Qué hacemos:** juntamos las métricas de los tres modelos, las graficamos (F1 macro), elegimos
# el mejor por F1 macro (más representativo que accuracy con clases desbalanceadas) y mostramos su
# matriz de confusión 3x3 y sus métricas por clase.
#
# **Para qué:** la matriz de confusión 3x3 y el detalle por clase muestran algo que un solo número no
# puede: es muy probable que el modelo confunda bastante "neutral" con las otras dos clases, mientras
# distingue mejor "negativa" de "positiva" (son más distintas entre sí).
#

comparacion = pd.DataFrame([metricas_arbol, metricas_bosque, metricas_xgb]).set_index("modelo")
comparacion.to_csv(OUT_DIR / "metricas_clasificacion.csv")
print(comparacion.round(3))

fig, ax = plt.subplots(figsize=(7, 4.5))
barras = comparacion["f1_macro"].sort_values().plot(kind="barh", color=OLIST_BLUE, ax=ax)
ax.set_xlim(0, 1)
ax.bar_label(barras.containers[0], fmt="%.3f", padding=3)
ax.set_title("F1 macro por modelo (test)")
plt.tight_layout()
_fig_i += 1
plt.savefig(FIGS_DIR / f"fig_{_fig_i:02d}.png", dpi=110, bbox_inches="tight")
plt.show()

mejor_nombre = comparacion["f1_macro"].idxmax()
modelos = {"Decision Tree": arbol, "Random Forest": bosque, "XGBoost": xgb}
mejor_modelo = modelos[mejor_nombre]
print(f"\nMejor modelo por F1 macro: {mejor_nombre}")

etiquetas = ["Negativa", "Neutral", "Positiva"]
pred_mejor = mejor_modelo.predict(X_test)
proba_mejor = mejor_modelo.predict_proba(X_test)

fig, ax = plt.subplots(figsize=(5.5, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, pred_mejor, display_labels=etiquetas, cmap="Blues", ax=ax, colorbar=False,
)
ax.set_title(f"Matriz de confusión — {mejor_nombre}")
plt.tight_layout()
_fig_i += 1
plt.savefig(FIGS_DIR / f"fig_{_fig_i:02d}.png", dpi=110, bbox_inches="tight")
plt.show()

cm = confusion_matrix(y_test, pred_mejor)
pd.DataFrame(cm, index=[f"real_{e}" for e in etiquetas], columns=[f"pred_{e}" for e in etiquetas]) \
    .to_csv(OUT_DIR / "matriz_confusion.csv")

metricas_por_clase = pd.DataFrame({
    "precision": precision_score(y_test, pred_mejor, average=None, zero_division=0),
    "recall": recall_score(y_test, pred_mejor, average=None, zero_division=0),
    "f1": f1_score(y_test, pred_mejor, average=None, zero_division=0),
}, index=etiquetas)
print("\nMétricas por clase:")
print(metricas_por_clase.round(3))

pd.DataFrame({
    "y_real": y_test.values,
    "prediccion": pred_mejor,
    "proba_negativa": proba_mejor[:, 0],
    "proba_neutral": proba_mejor[:, 1],
    "proba_positiva": proba_mejor[:, 2],
}).to_csv(OUT_DIR / "predicciones_test.csv", index=False)

(OUT_DIR / "mejor_modelo.txt").write_text(mejor_nombre)

#
# **Insight:** si "neutral" tiene el recall más bajo de las tres clases (lo esperable), confirma
# que la línea entre "conforme" y "casi decepcionado" es borrosa incluso con buenas features — es una
# limitación real del problema, no del modelo, y vale la pena decirlo así en el informe en vez de
# esconderla.
#

#
# **Qué hacemos:** graficamos la importancia de features del mejor modelo (top 15).
#
# **Para qué:** conecta el modelo de vuelta con el EDA — interesa ver si `historial_pct_demora_vendedor`
# o `distancia_km` aparecen entre las más relevantes para anticipar satisfacción, ya sin la ventaja
# "artificial" de usar la demora del propio pedido.
#

if hasattr(mejor_modelo, "feature_importances_"):
    importancias = pd.Series(mejor_modelo.feature_importances_, index=X.columns)
else:
    importancias = pd.Series(np.abs(mejor_modelo.coef_).mean(axis=0), index=X.columns)

top_importancias = importancias.sort_values(ascending=False).head(15)
top_importancias.to_csv(OUT_DIR / "feature_importance.csv", header=["importancia"])

fig, ax = plt.subplots(figsize=(9, 6))
barras = top_importancias.sort_values().plot(kind="barh", color=OLIST_BLUE, ax=ax)
ax.set_title(f"Importancia de variables — {mejor_nombre} (top 15)")
plt.tight_layout()
_fig_i += 1
plt.savefig(FIGS_DIR / f"fig_{_fig_i:02d}.png", dpi=110, bbox_inches="tight")
plt.show()

#
# **Insight:** si `historial_pct_demora_vendedor` aparece entre las más importantes, confirma que
# el comportamiento pasado del vendedor sigue siendo una señal útil de satisfacción incluso sin usar la
# demora del pedido actual — es la versión "justa" de la intuición que motivó este modelo.
#

#
# 3. Modelo 2 — Clustering: segmentación de vendedores por comportamiento logístico
#
# **Por qué este modelo:** en el EDA (sección de análisis exhaustivo) ya armamos un ranking simple de
# vendedores por volumen y por % de demora, y notamos que no son los mismos vendedores en ambos
# extremos. Un clustering formaliza esa intuición: en vez de mirar cada métrica por separado, agrupa a
# los vendedores según su perfil logístico completo. Usamos **K-Means**, el algoritmo de clustering más
# estándar y fácil de interpretar, eligiendo la cantidad de clusters con el método del codo (elbow) y el
# puntaje de silhouette.
#

#
# **Qué hacemos:** agregamos el dataset a nivel de vendedor, con **todos** los vendedores (sin
# filtrar por cantidad mínima de pedidos) y con 8 variables en vez de 5: las logísticas que ya
# teníamos (tiempo de entrega promedio, % de envíos demorados, distancia promedio, flete promedio,
# cantidad de pedidos) más 3 de negocio — **reseña promedio**, **precio promedio** y **cantidad de
# categorías distintas** que vende. A `pedidos` y `precio_promedio` (las dos con distribución más
# sesgada) les aplicamos `log1p` antes de escalar.
#
# **Para qué:** en la primera versión filtrábamos a vendedores con >= 20 pedidos "para tener perfiles
# confiables" — pero eso corta justo la cola de vendedores chicos y problemáticos, que es donde
# aparecen los casos más extremos (y más interesantes de detectar). Agregar reseña/precio/categorías
# como features de negocio, no solo logísticas, hace que los clusters separen también por calidad y no
# solo por velocidad de entrega. El `log1p` evita que los pocos vendedores con miles de pedidos
# dominen la distancia euclidiana por su escala, no por ser realmente distintos.
#

perfil_vendedores = df.groupby("seller_id").agg(
    pedidos=("order_id", "nunique"),
    tiempo_entrega_promedio=("tiempo_entrega_dias", "mean"),
    pct_demorado=("envio_demorado", "mean"),
    distancia_promedio=("distancia_km", "mean"),
    flete_promedio=("freight_value", "mean"),
    reseña_promedio=("review_score", "mean"),
    precio_promedio=("price", "mean"),
    categorias_distintas=("product_category_name_english", "nunique"),
).dropna()

FEATURES_CLUSTER = [
    "tiempo_entrega_promedio", "pct_demorado", "distancia_promedio", "flete_promedio",
    "pedidos", "reseña_promedio", "precio_promedio", "categorias_distintas",
]

X_cluster_df = perfil_vendedores[FEATURES_CLUSTER].copy()
for col in ["pedidos", "precio_promedio"]:
    X_cluster_df[col] = np.log1p(X_cluster_df[col])

scaler = StandardScaler()
X_cluster = scaler.fit_transform(X_cluster_df)

print(f"Vendedores a clusterizar: {len(perfil_vendedores):,}")

#
# **Qué hacemos:** probamos K-Means con distintas cantidades de clusters (de 2 a 8) y graficamos,
# para cada una, la inercia (método del codo) y el puntaje de silhouette.
#
# **Para qué:** no hay una "cantidad correcta" de clusters dada de antemano — el codo muestra dónde
# agregar más clusters deja de reducir mucho el error, y el silhouette mide qué tan bien separados
# quedan los grupos; entre los dos, eligen la cantidad de clusters más razonable.
#

K_MAXIMO = min(9, len(perfil_vendedores))  # no se puede pedir más clusters que vendedores

resultados_k = []
for k in range(2, K_MAXIMO):
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_cluster)
    sil = silhouette_score(X_cluster, labels) if len(set(labels)) > 1 else np.nan
    resultados_k.append({"k": k, "inercia": km.inertia_, "silhouette": sil})

resultados_k = pd.DataFrame(resultados_k)
resultados_k.to_csv(OUT_DIR / "elbow_silhouette.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(resultados_k["k"], resultados_k["inercia"], color=OLIST_BLUE, marker="o")
axes[0].set_title("Método del codo")
axes[0].set_xlabel("k (cantidad de clusters)")
axes[0].set_ylabel("Inercia")

axes[1].plot(resultados_k["k"], resultados_k["silhouette"], color=PALETA_CATEGORICA[1], marker="o")
axes[1].set_title("Puntaje de silhouette")
axes[1].set_xlabel("k (cantidad de clusters)")
plt.tight_layout()
_fig_i += 1
plt.savefig(FIGS_DIR / f"fig_{_fig_i:02d}.png", dpi=110, bbox_inches="tight")
plt.show()

print(resultados_k)

#
# **Insight:** el k elegido es el que mejor combina un codo visible en la inercia con un
# silhouette razonable (valores de silhouette por debajo de 0.25-0.3 son normales en datos de negocio
# reales, que no forman grupos perfectamente separados como los datos sintéticos de un tutorial).
#

#
# **Qué hacemos:** entrenamos K-Means con la cantidad de clusters elegida (3, en línea con lo
# que suele salir en este tipo de segmentación logística, y ajustable si el gráfico anterior sugiere
# otra), asignamos el cluster a cada vendedor, y armamos una tabla resumen con el perfil promedio de
# cada cluster.
#
# **Para qué:** convierte los 3 (o los que hayan salido) grupos en algo interpretable: qué caracteriza
# a cada cluster en términos de negocio, no solo en números de un algoritmo.
#

K_ELEGIDO = min(3, len(perfil_vendedores))

kmeans_final = KMeans(n_clusters=K_ELEGIDO, random_state=RANDOM_STATE, n_init=10)
perfil_vendedores["cluster"] = kmeans_final.fit_predict(X_cluster)

resumen_clusters = perfil_vendedores.groupby("cluster")[FEATURES_CLUSTER].mean().round(1)
resumen_clusters["n_vendedores"] = perfil_vendedores.groupby("cluster").size()
resumen_clusters = resumen_clusters.sort_values("pct_demorado")

resumen_clusters.to_csv(OUT_DIR / "clustering_resumen.csv")
perfil_vendedores.reset_index().to_csv(OUT_DIR / "clustering_asignaciones.csv", index=False)

print(resumen_clusters)

#
# **Qué hacemos:** graficamos a los vendedores en un scatter de tiempo de entrega promedio vs.
# % de demora, coloreado por cluster.
#
# **Para qué:** una visualización directa de cómo separó K-Means a los vendedores en el plano de las
# dos variables más relacionadas con el problema de negocio (demora), aunque el clustering usó cinco
# variables en total.
#

fig, ax = plt.subplots(figsize=(8, 6))
for cluster_id in sorted(perfil_vendedores["cluster"].unique()):
    subset = perfil_vendedores[perfil_vendedores["cluster"] == cluster_id]
    ax.scatter(
        subset["tiempo_entrega_promedio"], subset["pct_demorado"] * 100,
        label=f"Cluster {cluster_id}", color=PALETA_CATEGORICA[cluster_id % len(PALETA_CATEGORICA)],
        alpha=0.6, s=40,
    )
ax.set_xlabel("Tiempo de entrega promedio (días)")
ax.set_ylabel("% de envíos demorados")
ax.set_title("Segmentación de vendedores por comportamiento logístico")
ax.legend()
plt.tight_layout()
_fig_i += 1
plt.savefig(FIGS_DIR / f"fig_{_fig_i:02d}.png", dpi=110, bbox_inches="tight")
plt.show()

#
# **Insight:** debería verse un cluster chico y claramente separado con tiempos de entrega altos,
# alto % de demora y reseña promedio baja (los "rezagados", candidatos a intervención prioritaria) y
# otro(s) con buen desempeño — al incluir `reseña_promedio` como feature (algo que la primera versión
# de este clustering no tenía), la separación debería notarse no solo en la logística sino también en
# la satisfacción, que es al final la métrica de negocio que más importa.
#

#
# ---
# Los archivos en `modelos_output/` (`metricas_clasificacion.csv`, `feature_importance.csv`,
# `elbow_silhouette.csv`, `clustering_resumen.csv`, `clustering_asignaciones.csv`) alimentan las
# pestañas de modelos de la app de Streamlit (`app.py`), para no tener que reentrenar nada al abrir
# la app.
#
