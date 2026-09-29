#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Olist E-Commerce - Experimento de features "en el instante de la compra" (escenario A)
#
# OBJETIVO: predecir si una reseña va a ser negativa (1-2) o positiva (4-5) usando
# SOLO información conocida en el momento de la compra. Nada de texto de la reseña,
# nada de tiempos/demoras reales de la entrega, nada derivado del review_score.
#
# Continúa las conclusiones de experimento_smote_binario.py:
#   - Target BINARIO (mejor que 3 clases).
#   - class_weight="balanced" (SMOTE no aportaba).
#
# La pregunta de este script: ¿features nuevas legítimas (historial del vendedor,
# temporales, geográficas) suben el techo de ~0.60 F1 macro del modelo tabular?
#
# Comparamos:
#   base   -> features actuales del pipeline (las que ya usa modelos_predictivos.py)
#   +vend  -> base + reseña histórica y variedad de categorías del vendedor
#   +temp  -> +vend + mes / día de semana / temporada alta
#   +geo   -> +temp + estado del cliente y mismo_estado cliente-vendedor  (set completo A)
#
# Regla anti-fuga: todo lo "histórico del vendedor" se calcula SOLO con train.
# Escribe en modelos_output_experimento/. No toca el pipeline principal ni la app.
#
# Se corre directo con: python experimento_features_compra.py
#

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

RAW_DIR = Path("data/raw")
CSV_UNIFICADO = Path("data/olist_dataset_unificado.csv")
OUT_DIR = Path("modelos_output_experimento")
OUT_DIR.mkdir(exist_ok=True)
SUDESTE = {"SP", "RJ", "MG", "ES"}
RANDOM_STATE = 42

# ------------------------------------------------------------------
# 1. Preparación de datos (idéntica a modelos_predictivos.py)
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

# envio_demorado se usa SOLO para calcular el historial pasado del vendedor (no como feature directa)
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

# --- features temporales (conocidas al comprar) ---
df["mes_compra_num"] = df["order_purchase_timestamp"].dt.month
df["dia_semana"] = df["order_purchase_timestamp"].dt.dayofweek
# temporada alta: nov-ene (Black Friday, Navidad, verano brasileño)
df["es_temporada_alta"] = df["mes_compra_num"].isin([11, 12, 1]).astype(int)
# --- geográfica: mismo estado cliente-vendedor (conocida al comprar) ---
df["mismo_estado"] = (df["customer_state"] == df["seller_state"]).astype(int)


# target BINARIO: 0=neg (1-2), 1=pos (4-5), se descarta el 3
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
# 2. Definición de los conjuntos de features (todos conocidos al comprar)
# ------------------------------------------------------------------
# base: lo que ya usa el pipeline actual (modelos_predictivos.py)
BASE_NUM = [
    "distancia_km", "volumen_cm3", "product_weight_g", "price", "freight_value",
    "flete_ratio", "items_por_pedido", "payment_installments_max",
]
BASE_CAT = ["product_category_name_english", "region_sudeste", "payment_type_principal"]

# nuevas categóricas / numéricas por bloque
TEMP_NUM = ["mes_compra_num", "dia_semana", "es_temporada_alta"]
GEO_NUM = ["mismo_estado"]
GEO_CAT = ["customer_state"]
# el historial del vendedor (numérico) se calcula aparte, solo con train

TARGET = "reseña_binaria"
datos = df.dropna(subset=[TARGET]).copy()
y = datos[TARGET].astype(int)

# columnas crudas que necesitamos arrastrar hasta después del split
cols_arrastre = (["seller_id", "envio_demorado"] + BASE_NUM + BASE_CAT
                 + TEMP_NUM + GEO_NUM + GEO_CAT)
X_all = datos[cols_arrastre].copy()
for col in BASE_NUM:
    X_all[col] = X_all[col].fillna(X_all[col].median())

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_all, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {len(X_train_raw):,} | Test: {len(X_test_raw):,}")
print(f"Distribución train (%): {(y_train.value_counts(normalize=True).sort_index() * 100).round(1).to_dict()}")

# ------------------------------------------------------------------
# 3. Historial del vendedor calculado SOLO con train (evita fuga)
#    Incluye: % histórico de demora, cantidad de pedidos, reseña promedio
#    histórica y cantidad de categorías distintas que vende.
# ------------------------------------------------------------------
idx_tr = X_train_raw.index
hist_demora = datos.loc[idx_tr, "envio_demorado"].groupby(X_train_raw["seller_id"]).agg(["mean", "count"])
hist_demora.columns = ["hist_pct_demora_vendedor", "hist_pedidos_vendedor"]
hist_resena = datos.loc[idx_tr, "review_score"].groupby(X_train_raw["seller_id"]).mean()
hist_cats = datos.loc[idx_tr, "product_category_name_english"].groupby(X_train_raw["seller_id"]).nunique()

prom_demora_global = datos.loc[idx_tr, "envio_demorado"].mean()
prom_resena_global = datos.loc[idx_tr, "review_score"].mean()

historial = pd.DataFrame(hist_demora)
historial["hist_resena_prom_vendedor"] = hist_resena
historial["hist_categorias_vendedor"] = hist_cats


def agregar_historial(X_parte):
    out = X_parte.merge(historial, left_on="seller_id", right_index=True, how="left")
    out["hist_pct_demora_vendedor"] = out["hist_pct_demora_vendedor"].fillna(prom_demora_global)
    out["hist_pedidos_vendedor"] = out["hist_pedidos_vendedor"].fillna(0)
    out["hist_resena_prom_vendedor"] = out["hist_resena_prom_vendedor"].fillna(prom_resena_global)
    out["hist_categorias_vendedor"] = out["hist_categorias_vendedor"].fillna(0)
    return out


X_train_h = agregar_historial(X_train_raw)
X_test_h = agregar_historial(X_test_raw)

# bloques de features de vendedor (el pipeline actual ya tenía demora+pedidos; sumamos reseña+categorías)
VEND_BASE = ["hist_pct_demora_vendedor", "hist_pedidos_vendedor"]  # ya en el pipeline actual
VEND_NUEVO = ["hist_resena_prom_vendedor", "hist_categorias_vendedor"]  # nuevas

# ------------------------------------------------------------------
# 4. Armado de matrices por configuración
# ------------------------------------------------------------------
def construir(Xtr, Xte, num_cols, cat_cols):
    Xtr = pd.get_dummies(Xtr[num_cols + cat_cols], columns=cat_cols, drop_first=True)
    Xte = pd.get_dummies(Xte[num_cols + cat_cols], columns=cat_cols, drop_first=True)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)
    return Xtr, Xte


CONFIGS = {
    "base":  (BASE_NUM + VEND_BASE, BASE_CAT),
    "+vend": (BASE_NUM + VEND_BASE + VEND_NUEVO, BASE_CAT),
    "+temp": (BASE_NUM + VEND_BASE + VEND_NUEVO + TEMP_NUM, BASE_CAT),
    "+geo":  (BASE_NUM + VEND_BASE + VEND_NUEVO + TEMP_NUM + GEO_NUM, BASE_CAT + GEO_CAT),
}


def rf():
    return RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced",
                                  random_state=RANDOM_STATE, n_jobs=-1)


def lgbm():
    return LGBMClassifier(n_estimators=300, max_depth=-1, learning_rate=0.05,
                          num_leaves=31, class_weight="balanced",
                          random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)


def evaluar(modelo, X_test, y_test):
    pred = modelo.predict(X_test)
    proba = modelo.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision_macro": precision_score(y_test, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, pred, average="macro", zero_division=0),
        "auc": roc_auc_score(y_test, proba),
    }, pred


resultados = []
matrices = {}
for nombre_cfg, (num_cols, cat_cols) in CONFIGS.items():
    Xtr, Xte = construir(X_train_h, X_test_h, num_cols, cat_cols)
    for nombre_mod, constructor in [("Random Forest", rf), ("LightGBM", lgbm)]:
        modelo = constructor()
        modelo.fit(Xtr, y_train)
        m, pred = evaluar(modelo, Xte, y_test)
        m.update(configuracion=nombre_cfg, modelo=nombre_mod, n_features=Xtr.shape[1])
        resultados.append(m)
        clave = f"{nombre_cfg}_{nombre_mod.replace(' ', '')}"
        matrices[clave] = confusion_matrix(y_test, pred)
        print(f"{nombre_cfg:6s} {nombre_mod:14s} ({Xtr.shape[1]:3d} feats) "
              f"-> F1={m['f1_macro']:.3f} AUC={m['auc']:.3f} acc={m['accuracy']:.3f}")

# ------------------------------------------------------------------
# 5. Consolidado + importancia de la mejor config
# ------------------------------------------------------------------
tabla = pd.DataFrame(resultados)[
    ["configuracion", "modelo", "n_features", "accuracy", "precision_macro",
     "recall_macro", "f1_macro", "auc"]
].sort_values("f1_macro", ascending=False)
tabla.to_csv(OUT_DIR / "comparacion_features_compra.csv", index=False)

for nombre, cm in matrices.items():
    pd.DataFrame(cm, index=["real_Negativa", "real_Positiva"],
                 columns=["pred_Negativa", "pred_Positiva"]).to_csv(
        OUT_DIR / f"matriz_confusion_compra_{nombre}.csv")

# importancia del mejor modelo (set completo +geo)
Xtr_full, Xte_full = construir(X_train_h, X_test_h, *CONFIGS["+geo"])
mejor_cfg_geo = max(
    [r for r in resultados if r["configuracion"] == "+geo"],
    key=lambda r: r["f1_macro"],
)
modelo_final = rf() if mejor_cfg_geo["modelo"] == "Random Forest" else lgbm()
modelo_final.fit(Xtr_full, y_train)
importancias = pd.Series(modelo_final.feature_importances_, index=Xtr_full.columns)
importancias.sort_values(ascending=False).head(20).to_csv(
    OUT_DIR / "feature_importance_compra.csv", header=["importancia"])

print("\n" + "=" * 82)
print("RESUMEN - features 'en el instante de la compra' (escenario A, target binario):")
print("=" * 82)
print(tabla.round(3).to_string(index=False))
print("\nTop features del set completo (+geo):")
print(importancias.sort_values(ascending=False).head(12).round(3).to_string())
print(f"\nOutputs guardados en {OUT_DIR}/")
