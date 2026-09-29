#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Olist E-Commerce - Entrenamiento del modelo final (escenario A) para predicción manual
#
# Reentrena el MEJOR modelo del experimento del escenario A:
#   LightGBM, set completo de features "+geo", target BINARIO (negativa 1-2 vs positiva 4-5).
#   (F1 macro ~0.618, AUC ~0.718 en experimento_features_compra.py)
#
# A diferencia de los scripts de experimento, este SÍ serializa todo lo necesario para
# predecir una compra futura desde la app (app1.py), sin reentrenar:
#   - el modelo entrenado
#   - las columnas exactas del one-hot (para reindexar el vector de entrada)
#   - la tabla de historial por vendedor (reseña prom., % demora, pedidos, categorías)
#   - las medianas de imputación de las numéricas
#   - los promedios globales para vendedores sin historial
#   - las opciones de las categóricas (para poblar los desplegables del formulario)
#   - un catálogo de vendedores para elegir en la app (opción a)
#
# Todo el historial y las medianas se calculan SOLO con train (misma regla anti-fuga).
#
# Se corre con: python entrenar_modelo_final.py
#

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

RAW_DIR = Path("data/raw")
CSV_UNIFICADO = Path("data/olist_dataset_unificado.csv")
OUT_DIR = Path("modelos_output_experimento")
OUT_DIR.mkdir(exist_ok=True)
MODELO_PATH = OUT_DIR / "modelo_final_compra.joblib"
SUDESTE = {"SP", "RJ", "MG", "ES"}
RANDOM_STATE = 42

# ------------------------------------------------------------------
# 1. Preparación de datos (idéntica a experimento_features_compra.py)
# ------------------------------------------------------------------
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

COLUMNAS_FECHA = [
    "shipping_limit_date", "order_purchase_timestamp", "order_approved_at",
    "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_delivery_date",
]
df = df.drop_duplicates().reset_index(drop=True)
for col in COLUMNAS_FECHA:
    df[col] = pd.to_datetime(df[col], errors="coerce")

df["product_category_name_english"] = df["product_category_name_english"].fillna("sin_categoria").str.strip()

for prefijo in ["customer", "seller"]:
    df[f"{prefijo}_lat"] = pd.to_numeric(df[f"{prefijo}_lat"], errors="coerce")
    df[f"{prefijo}_lng"] = pd.to_numeric(df[f"{prefijo}_lng"], errors="coerce")
    geo_valida = df[f"{prefijo}_lat"].between(-34, 6) & df[f"{prefijo}_lng"].between(-74, -32)
    df.loc[~geo_valida, [f"{prefijo}_lat", f"{prefijo}_lng"]] = np.nan

df["items_por_pedido"] = df.groupby("order_id")["order_item_id"].transform("count")

df["envio_demorado"] = np.where(
    df["order_delivered_customer_date"].isna(), np.nan,
    (df["order_delivered_customer_date"] > df["order_estimated_delivery_date"]).astype(float),
)
df["flete_ratio"] = df["freight_value"] / df["price"].replace(0, np.nan)
df["region_sudeste"] = df["customer_state"].isin(SUDESTE).map({True: "Sudeste", False: "Resto del país"})
df["volumen_cm3"] = df["product_length_cm"] * df["product_height_cm"] * df["product_width_cm"]


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


df["distancia_km"] = haversine_km(df["customer_lat"], df["customer_lng"], df["seller_lat"], df["seller_lng"])
df["mes_compra_num"] = df["order_purchase_timestamp"].dt.month
df["dia_semana"] = df["order_purchase_timestamp"].dt.dayofweek
df["es_temporada_alta"] = df["mes_compra_num"].isin([11, 12, 1]).astype(int)
df["mismo_estado"] = (df["customer_state"] == df["seller_state"]).astype(int)


def target_binario(s):
    if pd.isna(s):
        return np.nan
    if s <= 2:
        return 0
    if s >= 4:
        return 1
    return np.nan


df["reseña_binaria"] = df["review_score"].apply(target_binario)
print(f"Dataset listo: {df.shape[0]:,} filas")

# ------------------------------------------------------------------
# 2. Features del set completo "+geo" (escenario A)
# ------------------------------------------------------------------
BASE_NUM = [
    "distancia_km", "volumen_cm3", "product_weight_g", "price", "freight_value",
    "flete_ratio", "items_por_pedido", "payment_installments_max",
]
VEND_NUM = ["hist_pct_demora_vendedor", "hist_pedidos_vendedor",
            "hist_resena_prom_vendedor", "hist_categorias_vendedor"]
TEMP_NUM = ["mes_compra_num", "dia_semana", "es_temporada_alta"]
GEO_NUM = ["mismo_estado"]
BASE_CAT = ["product_category_name_english", "region_sudeste", "payment_type_principal"]
GEO_CAT = ["customer_state"]

NUM_COLS = BASE_NUM + VEND_NUM + TEMP_NUM + GEO_NUM
CAT_COLS = BASE_CAT + GEO_CAT

TARGET = "reseña_binaria"
datos = df.dropna(subset=[TARGET]).copy()
y = datos[TARGET].astype(int)

cols_arrastre = (["seller_id", "seller_state", "envio_demorado"] + BASE_NUM
                 + BASE_CAT + TEMP_NUM + GEO_NUM + GEO_CAT)
X_all = datos[cols_arrastre].copy()

# medianas de imputación (se calculan con todo el df de modelado, se guardan para predicción)
medianas = {c: float(X_all[c].median()) for c in BASE_NUM}
for col in BASE_NUM:
    X_all[col] = X_all[col].fillna(medianas[col])

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_all, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

# ------------------------------------------------------------------
# 3. Historial del vendedor SOLO con train (evita fuga)
# ------------------------------------------------------------------
idx_tr = X_train_raw.index
hist_demora = datos.loc[idx_tr, "envio_demorado"].groupby(X_train_raw["seller_id"]).agg(["mean", "count"])
hist_demora.columns = ["hist_pct_demora_vendedor", "hist_pedidos_vendedor"]
hist_resena = datos.loc[idx_tr, "review_score"].groupby(X_train_raw["seller_id"]).mean()
hist_cats = datos.loc[idx_tr, "product_category_name_english"].groupby(X_train_raw["seller_id"]).nunique()
seller_estado = datos.loc[idx_tr].groupby(X_train_raw["seller_id"])["seller_state"].first()

prom_demora_global = float(datos.loc[idx_tr, "envio_demorado"].mean())
prom_resena_global = float(datos.loc[idx_tr, "review_score"].mean())

historial = pd.DataFrame(hist_demora)
historial["hist_resena_prom_vendedor"] = hist_resena
historial["hist_categorias_vendedor"] = hist_cats
historial["seller_state"] = seller_estado


def agregar_historial(X_parte):
    out = X_parte.merge(historial.drop(columns="seller_state"),
                        left_on="seller_id", right_index=True, how="left")
    out["hist_pct_demora_vendedor"] = out["hist_pct_demora_vendedor"].fillna(prom_demora_global)
    out["hist_pedidos_vendedor"] = out["hist_pedidos_vendedor"].fillna(0)
    out["hist_resena_prom_vendedor"] = out["hist_resena_prom_vendedor"].fillna(prom_resena_global)
    out["hist_categorias_vendedor"] = out["hist_categorias_vendedor"].fillna(0)
    return out


X_train_h = agregar_historial(X_train_raw)
X_test_h = agregar_historial(X_test_raw)


def construir(Xtr, Xte):
    Xtr = pd.get_dummies(Xtr[NUM_COLS + CAT_COLS], columns=CAT_COLS, drop_first=True)
    Xte = pd.get_dummies(Xte[NUM_COLS + CAT_COLS], columns=CAT_COLS, drop_first=True)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)
    return Xtr, Xte


X_train, X_test = construir(X_train_h, X_test_h)
COLUMNAS_MODELO = list(X_train.columns)

# ------------------------------------------------------------------
# 4. Entrenamiento del modelo final
# ------------------------------------------------------------------
modelo = LGBMClassifier(
    n_estimators=300, max_depth=-1, learning_rate=0.05, num_leaves=31,
    class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
)
modelo.fit(X_train, y_train)

# verificación rápida (debe coincidir con el experimento)
pred = modelo.predict(X_test)
proba = modelo.predict_proba(X_test)[:, 1]
print(f"Verificación en test -> F1 macro={f1_score(y_test, pred, average='macro'):.3f} "
      f"AUC={roc_auc_score(y_test, proba):.3f}")

# ------------------------------------------------------------------
# 5. Catálogo de vendedores para la app (opción a) + opciones de categóricas
# ------------------------------------------------------------------
# catálogo: solo vendedores con historial real en train (los que tienen perfil confiable)
catalogo_vendedores = historial.reset_index().rename(columns={"index": "seller_id"})
catalogo_vendedores = catalogo_vendedores[catalogo_vendedores["hist_pedidos_vendedor"] > 0].copy()
catalogo_vendedores = catalogo_vendedores.sort_values("hist_pedidos_vendedor", ascending=False)
catalogo_vendedores.to_csv(OUT_DIR / "catalogo_vendedores.csv", index=False)

opciones_categoricas = {
    "product_category_name_english": sorted(datos["product_category_name_english"].dropna().unique().tolist()),
    "payment_type_principal": sorted(datos["payment_type_principal"].dropna().unique().tolist()),
    "customer_state": sorted(datos["customer_state"].dropna().unique().tolist()),
}

# ------------------------------------------------------------------
# 6. Serialización de todo el bundle
# ------------------------------------------------------------------
bundle = {
    "modelo": modelo,
    "columnas_modelo": COLUMNAS_MODELO,
    "num_cols": NUM_COLS,
    "cat_cols": CAT_COLS,
    "base_num": BASE_NUM,
    "medianas": medianas,
    "prom_demora_global": prom_demora_global,
    "prom_resena_global": prom_resena_global,
    "opciones_categoricas": opciones_categoricas,
    "sudeste": sorted(SUDESTE),
    "descripcion": "LightGBM +geo binario (escenario A) — reseña negativa(0)/positiva(1) al momento de la compra",
    "metricas": {"f1_macro": float(f1_score(y_test, pred, average="macro")),
                 "auc": float(roc_auc_score(y_test, proba))},
}
joblib.dump(bundle, MODELO_PATH)
print(f"Modelo + artefactos guardados en {MODELO_PATH}")
print(f"Catálogo de vendedores ({len(catalogo_vendedores):,}) en {OUT_DIR / 'catalogo_vendedores.csv'}")
