#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Olist E-Commerce - Experimento de balanceo de clases
#
# Objetivo: el dataset de reseñas está fuertemente desbalanceado (~77-80% positivas).
# El pipeline principal (modelos_predictivos.py) ya compensa con class_weight="balanced",
# pero el F1 macro se queda en ~0.42 y la clase Neutral casi no se detecta.
#
# Este script NO reemplaza al pipeline principal ni toca modelos_output/ (lo que consume
# la app de Streamlit). Escribe en modelos_output_experimento/ y compara 4 configuraciones
# con el mismo split y las mismas features, para decidir con números si conviene:
#
#   A) 3 clases  + class_weight="balanced"   (baseline, = pipeline actual)
#   B) 3 clases  + SMOTE (solo en train)
#   C) binario   + class_weight="balanced"   (negativa 1-2 vs positiva 4-5, sin neutral)
#   D) binario   + SMOTE (solo en train)
#
# Regla de oro: SMOTE se aplica SOLO sobre el conjunto de entrenamiento, nunca sobre el
# test, para no inflar las métricas artificialmente.
#
# Se corre directo con: python experimento_smote_binario.py
#

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

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

df["tiempo_entrega_dias"] = (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.days
df["dias_vs_estimado"] = (df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]).dt.days
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

print(f"Dataset listo: {df.shape[0]:,} filas, {df.shape[1]} columnas")

# ------------------------------------------------------------------
# 2. Features (idénticas al pipeline principal, sin variables de demora del propio pedido)
# ------------------------------------------------------------------
FEATURES_NUM = [
    "distancia_km", "volumen_cm3", "product_weight_g", "price", "freight_value",
    "flete_ratio", "items_por_pedido", "payment_installments_max",
]
FEATURES_CAT = ["product_category_name_english", "region_sudeste", "payment_type_principal"]


def construir_xy(df, target_col):
    datos = df.dropna(subset=[target_col]).copy()
    X_base = datos[["seller_id"] + FEATURES_NUM + FEATURES_CAT].copy()
    y = datos[target_col].astype(int)
    for col in FEATURES_NUM:
        X_base[col] = X_base[col].fillna(X_base[col].median())

    X_train_base, X_test_base, y_train, y_test = train_test_split(
        X_base, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    # historial del vendedor calculado SOLO con train (evita fuga de información)
    historial = (datos.loc[X_train_base.index, "envio_demorado"]
                 .groupby(X_train_base["seller_id"]).agg(["mean", "count"]))
    historial.columns = ["historial_pct_demora_vendedor", "historial_pedidos_vendedor"]
    prom_global = datos.loc[X_train_base.index, "envio_demorado"].mean()

    def agregar_historial(X_parte):
        X_parte = X_parte.merge(historial, left_on="seller_id", right_index=True, how="left")
        X_parte["historial_pct_demora_vendedor"] = X_parte["historial_pct_demora_vendedor"].fillna(prom_global)
        X_parte["historial_pedidos_vendedor"] = X_parte["historial_pedidos_vendedor"].fillna(0)
        return X_parte.drop(columns="seller_id")

    X_train = agregar_historial(X_train_base)
    X_test = agregar_historial(X_test_base)

    X_train = pd.get_dummies(X_train, columns=FEATURES_CAT, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=FEATURES_CAT, drop_first=True)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    return X_train, X_test, y_train, y_test


# Targets: 3 clases (0=neg, 1=neutral, 2=pos) y binario (0=neg 1-2, 1=pos 4-5, se descarta el 3)
def target_3clases(s):
    if pd.isna(s):
        return np.nan
    if s <= 2:
        return 0
    if s == 3:
        return 1
    return 2


def target_binario(s):
    if pd.isna(s):
        return np.nan
    if s <= 2:
        return 0
    if s >= 4:
        return 1
    return np.nan  # se descarta la clase neutral


df["reseña_categoria"] = df["review_score"].apply(target_3clases)
df["reseña_binaria"] = df["review_score"].apply(target_binario)

# ------------------------------------------------------------------
# 3. Entrenamiento y evaluación
# ------------------------------------------------------------------
def modelos_para(n_clases):
    """Devuelve los 3 clasificadores con la config de multiclase o binaria."""
    arbol = DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=RANDOM_STATE)
    bosque = RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    if n_clases > 2:
        xgb = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            objective="multi:softprob", num_class=n_clases, random_state=RANDOM_STATE,
            eval_metric="mlogloss",
        )
    else:
        xgb = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            objective="binary:logistic", random_state=RANDOM_STATE, eval_metric="logloss",
        )
    return {"Decision Tree": arbol, "Random Forest": bosque, "XGBoost": xgb}


def evaluar(modelo, X_test, y_test, n_clases):
    pred = modelo.predict(X_test)
    proba = modelo.predict_proba(X_test)
    if n_clases > 2:
        auc = roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
    else:
        auc = roc_auc_score(y_test, proba[:, 1])
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision_macro": precision_score(y_test, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, pred, average="macro", zero_division=0),
        "auc": auc,
    }


def correr_config(nombre_config, target_col, usar_smote):
    n_clases = df[target_col].dropna().nunique()
    X_train, X_test, y_train, y_test = construir_xy(df, target_col)

    dist_train = y_train.value_counts(normalize=True).sort_index()
    print(f"\n=== {nombre_config} ===")
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,} | Clases: {n_clases}")
    print(f"  Distribución train (%): {(dist_train * 100).round(1).to_dict()}")

    if usar_smote:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        dist_bal = pd.Series(y_train).value_counts(normalize=True).sort_index()
        print(f"  Distribución train tras SMOTE (%): {(dist_bal * 100).round(1).to_dict()}")

    filas = []
    modelos = modelos_para(n_clases)
    for nombre_modelo, modelo in modelos.items():
        if nombre_modelo == "XGBoost" and not usar_smote:
            # sin SMOTE, XGBoost compensa con pesos por muestra (multiclase no tiene scale_pos_weight)
            pesos = compute_sample_weight(class_weight="balanced", y=y_train)
            modelo.fit(X_train, y_train, sample_weight=pesos)
        else:
            modelo.fit(X_train, y_train)
        m = evaluar(modelo, X_test, y_test, n_clases)
        m["configuracion"] = nombre_config
        m["modelo"] = nombre_modelo
        filas.append(m)
        print(f"  {nombre_modelo:15s} -> F1 macro={m['f1_macro']:.3f}  "
              f"AUC={m['auc']:.3f}  acc={m['accuracy']:.3f}")

    # matriz de confusión del mejor modelo de esta config (por F1 macro)
    mejor_fila = max(filas, key=lambda r: r["f1_macro"])
    mejor_modelo = modelos[mejor_fila["modelo"]]
    cm = confusion_matrix(y_test, mejor_modelo.predict(X_test))
    return filas, mejor_fila, cm


CONFIGS = [
    ("A_3clases_classweight", "reseña_categoria", False),
    ("B_3clases_SMOTE", "reseña_categoria", True),
    ("C_binario_classweight", "reseña_binaria", False),
    ("D_binario_SMOTE", "reseña_binaria", True),
]

todas_filas = []
mejores = []
for nombre, target, smote in CONFIGS:
    filas, mejor, cm = correr_config(nombre, target, smote)
    todas_filas.extend(filas)
    mejores.append({**mejor})
    etiquetas = ["Negativa", "Neutral", "Positiva"] if "3clases" in nombre else ["Negativa", "Positiva"]
    pd.DataFrame(cm, index=[f"real_{e}" for e in etiquetas],
                 columns=[f"pred_{e}" for e in etiquetas]).to_csv(OUT_DIR / f"matriz_confusion_{nombre}.csv")

# ------------------------------------------------------------------
# 4. Resultados consolidados
# ------------------------------------------------------------------
resultados = pd.DataFrame(todas_filas)[
    ["configuracion", "modelo", "accuracy", "precision_macro", "recall_macro", "f1_macro", "auc"]
]
resultados.to_csv(OUT_DIR / "comparacion_completa.csv", index=False)

resumen_mejores = pd.DataFrame(mejores)[
    ["configuracion", "modelo", "accuracy", "precision_macro", "recall_macro", "f1_macro", "auc"]
].sort_values("f1_macro", ascending=False)
resumen_mejores.to_csv(OUT_DIR / "mejor_por_configuracion.csv", index=False)

print("\n" + "=" * 70)
print("RESUMEN - mejor modelo por configuración (ordenado por F1 macro):")
print("=" * 70)
print(resumen_mejores.round(3).to_string(index=False))
print(f"\nOutputs guardados en {OUT_DIR}/")
