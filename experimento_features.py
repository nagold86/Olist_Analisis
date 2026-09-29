#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Olist E-Commerce - Experimento de mejoras al dataset
#
# Continuación de experimento_smote_binario.py, que dejó dos conclusiones:
#   - SMOTE no aporta sobre class_weight="balanced".
#   - El target binario (negativa 1-2 vs positiva 4-5) casi duplica el F1 macro
#     respecto de las 3 clases (0.60 vs 0.42).
#
# Este script parte de esa base (target BINARIO) y prueba las dos mejoras de mayor
# retorno esperado, sin tocar el pipeline principal ni la app. Escribe en
# modelos_output_experimento/.
#
# Mejora 1 - Categóricas de alta cardinalidad:
#   1a) Baseline: one-hot (drop_first) + Random Forest      -> como el pipeline actual
#   1b) Target encoding con suavizado (solo train) + LightGBM
#   1c) LightGBM con categóricas nativas (sin one-hot ni target encoding)
#
# Mejora 2 - Features de texto de las reseñas (NLP):
#   2)  Mejor config de la Mejora 1  +  TF-IDF del comentario de la reseña
#
# NOTA de diseño: las features de texto de la reseña se conocen DESPUÉS de la compra,
# así que el modelo con texto es un clasificador "en el momento de la reseña", no un
# predictor previo a la entrega. Se reporta por separado para no mezclar los objetivos.
#
# Regla anti-fuga: target encoding, TF-IDF e historial de vendedor se AJUSTAN SOLO con
# el conjunto de entrenamiento y se aplican al test.
#
# Se corre directo con: python experimento_features.py
#

import re
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
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

# target BINARIO (mejor config del experimento anterior): 0=neg (1-2), 1=pos (4-5), se descarta el 3
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
# 2. Split base compartido por todas las configuraciones
# ------------------------------------------------------------------
FEATURES_NUM = [
    "distancia_km", "volumen_cm3", "product_weight_g", "price", "freight_value",
    "flete_ratio", "items_por_pedido", "payment_installments_max",
]
FEATURES_CAT = ["product_category_name_english", "region_sudeste", "payment_type_principal"]
TARGET = "reseña_binaria"

datos = df.dropna(subset=[TARGET]).copy()
cols_base = ["seller_id", "review_comment_message"] + FEATURES_NUM + FEATURES_CAT
X_base = datos[cols_base].copy()
y = datos[TARGET].astype(int)
for col in FEATURES_NUM:
    X_base[col] = X_base[col].fillna(X_base[col].median())

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_base, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {len(X_train_raw):,} | Test: {len(X_test_raw):,}")
print(f"Distribución train (%): {(y_train.value_counts(normalize=True).sort_index() * 100).round(1).to_dict()}")


# --- historial del vendedor (solo train), común a todas las configs ---
historial = (datos.loc[X_train_raw.index, "envio_demorado"]
             .groupby(X_train_raw["seller_id"]).agg(["mean", "count"]))
historial.columns = ["historial_pct_demora_vendedor", "historial_pedidos_vendedor"]
prom_global = datos.loc[X_train_raw.index, "envio_demorado"].mean()


def agregar_historial(X_parte):
    out = X_parte.merge(historial, left_on="seller_id", right_index=True, how="left")
    out["historial_pct_demora_vendedor"] = out["historial_pct_demora_vendedor"].fillna(prom_global)
    out["historial_pedidos_vendedor"] = out["historial_pedidos_vendedor"].fillna(0)
    return out


X_train_h = agregar_historial(X_train_raw)
X_test_h = agregar_historial(X_test_raw)
FEATURES_NUM_H = FEATURES_NUM + ["historial_pct_demora_vendedor", "historial_pedidos_vendedor"]


# ------------------------------------------------------------------
# 3. Utilidades de encoding de categóricas
# ------------------------------------------------------------------
def onehot(Xtr, Xte):
    Xtr = pd.get_dummies(Xtr[FEATURES_NUM_H + FEATURES_CAT], columns=FEATURES_CAT, drop_first=True)
    Xte = pd.get_dummies(Xte[FEATURES_NUM_H + FEATURES_CAT], columns=FEATURES_CAT, drop_first=True)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)
    return Xtr, Xte


def target_encode(Xtr, Xte, ytr, suavizado=20):
    """Target encoding con suavizado bayesiano. Se ajusta SOLO con train."""
    Xtr = Xtr.copy()
    Xte = Xte.copy()
    media_global = ytr.mean()
    ytr_s = ytr.reset_index(drop=True)
    for col in FEATURES_CAT:
        cat_tr = Xtr[col].reset_index(drop=True)
        agg = ytr_s.groupby(cat_tr).agg(["mean", "count"])
        # suavizado: acerca las categorías con pocos casos a la media global
        codificado = (agg["mean"] * agg["count"] + media_global * suavizado) / (agg["count"] + suavizado)
        Xtr[col] = Xtr[col].map(codificado).fillna(media_global)
        Xte[col] = Xte[col].map(codificado).fillna(media_global)
    return Xtr[FEATURES_NUM_H + FEATURES_CAT], Xte[FEATURES_NUM_H + FEATURES_CAT]


def categoricas_nativas(Xtr, Xte):
    """Deja las categóricas como dtype 'category' para que LightGBM las maneje solo."""
    Xtr = Xtr[FEATURES_NUM_H + FEATURES_CAT].copy()
    Xte = Xte[FEATURES_NUM_H + FEATURES_CAT].copy()
    for col in FEATURES_CAT:
        Xtr[col] = Xtr[col].astype("category")
        Xte[col] = pd.Categorical(Xte[col], categories=Xtr[col].cat.categories)
    return Xtr, Xte


# ------------------------------------------------------------------
# 4. Features de texto (TF-IDF) — se ajustan SOLO con train
# ------------------------------------------------------------------
STOPWORDS_PT = (
    "de a o que e do da em um para com nao uma os no se na por mais as dos como mas "
    "ao ele das a sua ou quando muito nos ja eu tambem so pelo pelas pela ate isso "
    "ela entre depois sem mesmo aos seus quem nas me esse eles voce essa num nem "
    "suas meu as minha numa pelos elas qual sera nos tenho lhe deles essas esses "
    "pelas este fosse dela tu te voces vos lhes meus minhas teu tua teus tuas "
    "nosso nossa nossos nossas dele"
).split()


def limpiar_texto(texto):
    if pd.isna(texto) or texto == "":
        return ""
    texto = str(texto).lower()
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("utf-8")
    texto = re.sub(r"[^a-z\s]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def features_texto(train_txt, test_txt):
    tr = train_txt.apply(limpiar_texto)
    te = test_txt.apply(limpiar_texto)
    vec = TfidfVectorizer(max_features=300, min_df=5, ngram_range=(1, 2),
                          stop_words=STOPWORDS_PT)
    Xtr_txt = vec.fit_transform(tr)
    Xte_txt = vec.transform(te)
    return Xtr_txt, Xte_txt


# ------------------------------------------------------------------
# 5. Evaluación
# ------------------------------------------------------------------
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


def rf():
    return RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced",
                                  random_state=RANDOM_STATE, n_jobs=-1)


def lgbm():
    return LGBMClassifier(n_estimators=300, max_depth=-1, learning_rate=0.05,
                          num_leaves=31, class_weight="balanced",
                          random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)


resultados = []
matrices = {}

# --- 1a) Baseline: one-hot + Random Forest ---
Xtr, Xte = onehot(X_train_h, X_test_h)
modelo = rf(); modelo.fit(Xtr, y_train)
m, pred = evaluar(modelo, Xte, y_test)
m.update(configuracion="1a_onehot_RF", modelo="Random Forest")
resultados.append(m); matrices["1a_onehot_RF"] = confusion_matrix(y_test, pred)
print(f"1a one-hot + RF        -> F1={m['f1_macro']:.3f} AUC={m['auc']:.3f} acc={m['accuracy']:.3f}")

# --- 1b) Target encoding + LightGBM ---
Xtr, Xte = target_encode(X_train_h, X_test_h, y_train)
modelo = lgbm(); modelo.fit(Xtr, y_train)
m, pred = evaluar(modelo, Xte, y_test)
m.update(configuracion="1b_targetenc_LGBM", modelo="LightGBM")
resultados.append(m); matrices["1b_targetenc_LGBM"] = confusion_matrix(y_test, pred)
print(f"1b target-enc + LGBM   -> F1={m['f1_macro']:.3f} AUC={m['auc']:.3f} acc={m['accuracy']:.3f}")

# --- 1c) Categóricas nativas + LightGBM ---
Xtr, Xte = categoricas_nativas(X_train_h, X_test_h)
modelo = lgbm(); modelo.fit(Xtr, y_train, categorical_feature=FEATURES_CAT)
m, pred = evaluar(modelo, Xte, y_test)
m.update(configuracion="1c_catnativa_LGBM", modelo="LightGBM")
resultados.append(m); matrices["1c_catnativa_LGBM"] = confusion_matrix(y_test, pred)
print(f"1c cat-nativa + LGBM   -> F1={m['f1_macro']:.3f} AUC={m['auc']:.3f} acc={m['accuracy']:.3f}")

# --- Elegir la mejor config de la Mejora 1 (por F1 macro) para sumarle texto ---
mejor1 = max(resultados, key=lambda r: r["f1_macro"])
print(f"\nMejor config de la Mejora 1: {mejor1['configuracion']} (F1={mejor1['f1_macro']:.3f})")

# --- 2) Mejor config de Mejora 1 + TF-IDF del texto de la reseña ---
# Reconstruimos la matriz de la mejor config y le concatenamos TF-IDF (disperso).
if mejor1["configuracion"] == "1b_targetenc_LGBM":
    Xtr_tab, Xte_tab = target_encode(X_train_h, X_test_h, y_train)
elif mejor1["configuracion"] == "1c_catnativa_LGBM":
    # para poder concatenar con la matriz dispersa de texto usamos target-enc como base tabular
    Xtr_tab, Xte_tab = target_encode(X_train_h, X_test_h, y_train)
else:
    Xtr_tab, Xte_tab = onehot(X_train_h, X_test_h)

Xtr_txt, Xte_txt = features_texto(X_train_raw["review_comment_message"],
                                  X_test_raw["review_comment_message"])

Xtr_full = hstack([csr_matrix(Xtr_tab.to_numpy(dtype=np.float64)), Xtr_txt]).tocsr()
Xte_full = hstack([csr_matrix(Xte_tab.to_numpy(dtype=np.float64)), Xte_txt]).tocsr()

modelo = lgbm(); modelo.fit(Xtr_full, y_train)
m, pred = evaluar(modelo, Xte_full, y_test)
m.update(configuracion="2_tabular+texto_LGBM", modelo="LightGBM")
resultados.append(m); matrices["2_tabular+texto_LGBM"] = confusion_matrix(y_test, pred)
print(f"2  tabular + texto     -> F1={m['f1_macro']:.3f} AUC={m['auc']:.3f} acc={m['accuracy']:.3f}")

# también: solo texto, para ver cuánto aporta el texto por sí solo
modelo = lgbm(); modelo.fit(Xtr_txt, y_train)
m, pred = evaluar(modelo, Xte_txt, y_test)
m.update(configuracion="2b_solo_texto_LGBM", modelo="LightGBM")
resultados.append(m); matrices["2b_solo_texto_LGBM"] = confusion_matrix(y_test, pred)
print(f"2b solo texto          -> F1={m['f1_macro']:.3f} AUC={m['auc']:.3f} acc={m['accuracy']:.3f}")

# ------------------------------------------------------------------
# 6. Consolidado
# ------------------------------------------------------------------
tabla = pd.DataFrame(resultados)[
    ["configuracion", "modelo", "accuracy", "precision_macro", "recall_macro", "f1_macro", "auc"]
].sort_values("f1_macro", ascending=False)
tabla.to_csv(OUT_DIR / "comparacion_features.csv", index=False)

for nombre, cm in matrices.items():
    pd.DataFrame(cm, index=["real_Negativa", "real_Positiva"],
                 columns=["pred_Negativa", "pred_Positiva"]).to_csv(
        OUT_DIR / f"matriz_confusion_{nombre}.csv")

print("\n" + "=" * 78)
print("RESUMEN - mejoras al dataset (target binario, ordenado por F1 macro):")
print("=" * 78)
print(tabla.round(3).to_string(index=False))

# ------------------------------------------------------------------
# 7. Desglose honesto: solo el ~42% de las reseñas tiene comentario.
#    Para el resto, el TF-IDF es todo ceros. Medimos la config tabular+texto
#    por separado en el subconjunto CON comentario y SIN comentario del test.
# ------------------------------------------------------------------
tiene_com_test = (X_test_raw["review_comment_message"].astype(str).str.strip() != "").to_numpy()

modelo_full = lgbm(); modelo_full.fit(Xtr_full, y_train)
pred_full = modelo_full.predict(Xte_full)
proba_full = modelo_full.predict_proba(Xte_full)[:, 1]

filas_desglose = []
for etiqueta, mascara in [("test completo", np.ones(len(y_test), bool)),
                          ("solo CON comentario", tiene_com_test),
                          ("solo SIN comentario", ~tiene_com_test)]:
    yt = y_test.to_numpy()[mascara]
    yp = pred_full[mascara]
    pp = proba_full[mascara]
    filas_desglose.append({
        "subconjunto": etiqueta,
        "n": int(mascara.sum()),
        "pct_test": round(mascara.mean() * 100, 1),
        "f1_macro": round(f1_score(yt, yp, average="macro", zero_division=0), 3),
        "auc": round(roc_auc_score(yt, pp), 3) if len(set(yt)) > 1 else np.nan,
        "accuracy": round(accuracy_score(yt, yp), 3),
    })

desglose = pd.DataFrame(filas_desglose)
desglose.to_csv(OUT_DIR / "desglose_texto_por_comentario.csv", index=False)

print("\n" + "=" * 78)
print("DESGLOSE de 'tabular+texto' según la reseña tenga o no comentario:")
print("(el texto solo existe en ~42% de las reseñas; en el resto TF-IDF es 0)")
print("=" * 78)
print(desglose.to_string(index=False))
print(f"\nOutputs guardados en {OUT_DIR}/")
