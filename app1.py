"""
Olist E-Commerce — Panel de EXPERIMENTOS de modelado (app1)
Proyecto integrador — Tecnicatura en Ciencia de Datos e Inteligencia Artificial

App de Streamlit AUXILIAR para revisar en local los resultados de los experimentos de
la rama `experimento/smote-binario`, SIN tocar la app principal (`app.py`) ni el
pipeline de producción (`modelos_predictivos.py`).

Lee los CSV de `modelos_output_experimento/` (generados por `experimento_smote_binario.py`
y `experimento_features_compra.py`). No reentrena nada, no depende de token de Kaggle.

Cubre las tres decisiones que se tomaron con datos:
  1. 3 clases vs binario, y SMOTE vs class_weight.
  2. Mejoras de features "en el instante de la compra" (escenario A).
  3. Importancia de variables del mejor modelo del escenario A.

Se levanta con:  streamlit run app1.py
Comparte el tema de .streamlit/config.toml con la app principal.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# Estilo (mismo tema Olist que app.py; los colores base viven en config.toml)
# ============================================================
CHART_BLUE = "#0A4EE4"
BG_PAGE = "#FFFFFF"
TEXTO_GRAFICO = "#001647"
GRID_COLOR = "#E5E8ED"
PALETA_CATEGORICA = [
    "#0A4EE4", "#E8590C", "#0CA678", "#F59F00",
    "#E64980", "#2F9E44", "#7048E8", "#E03131",
]
VERDE = "#2F9E44"
ROJO = "#E03131"

EXP_DIR = Path("modelos_output_experimento")


def estilizar(fig, showlegend=None, height=None):
    """Aplica el tema claro de olist.com a un gráfico de Plotly (igual que app.py)."""
    layout_kwargs = dict(
        font_family="Plus Jakarta Sans", font_color=TEXTO_GRAFICO,
        paper_bgcolor=BG_PAGE, plot_bgcolor=BG_PAGE,
        margin=dict(t=14, b=35, l=10, r=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    if showlegend is not None:
        layout_kwargs["showlegend"] = showlegend
    if height is not None:
        layout_kwargs["height"] = height
    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, linecolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, linecolor=GRID_COLOR)
    return fig


def chart_card(fig, titulo, height=None):
    if height is not None:
        fig.update_layout(height=height)
    with st.container(border=True):
        st.markdown(f"#### {titulo}")
        st.plotly_chart(fig, width="stretch")


def insight_box(texto):
    st.markdown(f"**Lectura:** {texto}")


# ============================================================
# Carga de resultados de los experimentos
# ============================================================
@st.cache_data
def cargar(nombre, **kwargs):
    ruta = EXP_DIR / nombre
    if not ruta.exists():
        return None
    return pd.read_csv(ruta, **kwargs)


def matriz_heatmap(df_cm, titulo):
    """Renderiza una matriz de confusión (índice=real, columnas=pred) como heatmap."""
    etiquetas_real = [i.replace("real_", "") for i in df_cm.index]
    etiquetas_pred = [c.replace("pred_", "") for c in df_cm.columns]
    fig = px.imshow(
        df_cm.values, x=etiquetas_pred, y=etiquetas_real,
        color_continuous_scale="Blues", text_auto=True, aspect="auto",
        labels={"x": "Predicción del modelo", "y": "Clase real", "color": "Casos"},
    )
    estilizar(fig, height=340)
    fig.update_coloraxes(showscale=False)
    chart_card(fig, titulo)


# ============================================================
# Carga del modelo final serializado (para el tab de predicción)
# ============================================================
MODELO_PATH = EXP_DIR / "modelo_final_compra.joblib"
SUDESTE = {"SP", "RJ", "MG", "ES"}


@st.cache_resource
def cargar_bundle():
    """Carga el modelo entrenado + artefactos generados por entrenar_modelo_final.py."""
    if not MODELO_PATH.exists():
        return None
    return joblib.load(MODELO_PATH)


@st.cache_data
def cargar_catalogo():
    ruta = EXP_DIR / "catalogo_vendedores.csv"
    if not ruta.exists():
        return None
    return pd.read_csv(ruta)


def predecir_compra(bundle, entrada: dict):
    """Reconstruye el vector de features EXACTAMENTE como en el entrenamiento
    (mismas columnas derivadas, one-hot y reindexado) y devuelve la probabilidad
    de reseña positiva.

    `entrada` trae los valores crudos del formulario:
      price, freight_value, product_weight_g, volumen_cm3, distancia_km,
      items_por_pedido, payment_installments_max, mes_compra_num, dia_semana,
      product_category_name_english, payment_type_principal, customer_state,
      seller_state, hist_pct_demora_vendedor, hist_pedidos_vendedor,
      hist_resena_prom_vendedor, hist_categorias_vendedor
    """
    fila = {}
    # numéricas base
    fila["distancia_km"] = entrada["distancia_km"]
    fila["volumen_cm3"] = entrada["volumen_cm3"]
    fila["product_weight_g"] = entrada["product_weight_g"]
    fila["price"] = entrada["price"]
    fila["freight_value"] = entrada["freight_value"]
    fila["flete_ratio"] = entrada["freight_value"] / entrada["price"] if entrada["price"] else 0.0
    fila["items_por_pedido"] = entrada["items_por_pedido"]
    fila["payment_installments_max"] = entrada["payment_installments_max"]
    # historial del vendedor (viene del catálogo o ajustado a mano)
    fila["hist_pct_demora_vendedor"] = entrada["hist_pct_demora_vendedor"]
    fila["hist_pedidos_vendedor"] = entrada["hist_pedidos_vendedor"]
    fila["hist_resena_prom_vendedor"] = entrada["hist_resena_prom_vendedor"]
    fila["hist_categorias_vendedor"] = entrada["hist_categorias_vendedor"]
    # temporales
    fila["mes_compra_num"] = entrada["mes_compra_num"]
    fila["dia_semana"] = entrada["dia_semana"]
    fila["es_temporada_alta"] = int(entrada["mes_compra_num"] in (11, 12, 1))
    # geográfica derivada
    fila["mismo_estado"] = int(entrada["customer_state"] == entrada["seller_state"])
    # categóricas
    fila["product_category_name_english"] = entrada["product_category_name_english"]
    fila["region_sudeste"] = "Sudeste" if entrada["customer_state"] in SUDESTE else "Resto del país"
    fila["payment_type_principal"] = entrada["payment_type_principal"]
    fila["customer_state"] = entrada["customer_state"]

    X = pd.DataFrame([fila])
    X = pd.get_dummies(X, columns=bundle["cat_cols"], drop_first=True)
    X = X.reindex(columns=bundle["columnas_modelo"], fill_value=0)

    proba_pos = float(bundle["modelo"].predict_proba(X)[:, 1][0])
    return proba_pos


# ============================================================
# Configuración de la página
# ============================================================
st.set_page_config(page_title="Olist — Experimentos de modelado", layout="wide")

st.title("Olist — Experimentos de modelado")
st.caption(
    "Panel auxiliar (app1) para revisar los experimentos de balanceo y de features. "
    "No modifica la app principal ni el pipeline de producción. "
    "Rama: experimento/smote-binario."
)

if not EXP_DIR.exists():
    st.error(
        f"No encuentro la carpeta `{EXP_DIR}/`. Corré primero "
        "`python experimento_smote_binario.py` y `python experimento_features_compra.py`."
    )
    st.stop()

tabs = st.tabs([
    "1 · Balanceo y binario",
    "2 · Features al comprar (escenario A)",
    "3 · Importancia de variables",
    "4 · Predecir una compra",
])

# ============================================================
# TAB 1 — Balanceo de clases y target binario
# ============================================================
with tabs[0]:
    st.subheader("¿SMOTE o class_weight? ¿3 clases o binario?")
    st.markdown(
        "El dataset está muy desbalanceado (~77% de reseñas positivas). Este primer "
        "experimento compara cuatro caminos con las mismas features y el mismo split, "
        "midiendo con **F1 macro** (que pesa las clases por igual) en vez de accuracy."
    )

    mejores = cargar("mejor_por_configuracion.csv")
    if mejores is None:
        st.warning("Falta `mejor_por_configuracion.csv`. Corré `experimento_smote_binario.py`.")
    else:
        # nombres legibles
        legibles = {
            "C_binario_classweight": "Binario + class_weight",
            "D_binario_SMOTE": "Binario + SMOTE",
            "A_3clases_classweight": "3 clases + class_weight",
            "B_3clases_SMOTE": "3 clases + SMOTE",
        }
        m = mejores.copy()
        m["config_legible"] = m["configuracion"].map(legibles).fillna(m["configuracion"])
        m = m.sort_values("f1_macro")

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                m, x="f1_macro", y="config_legible", orientation="h",
                color_discrete_sequence=[CHART_BLUE],
                text=m["f1_macro"].round(3).values,
                labels={"f1_macro": "F1 macro (0 a 1, más alto es mejor)", "config_legible": ""},
            )
            fig.update_xaxes(range=[0, 1])
            estilizar(fig, height=320)
            chart_card(fig, "F1 macro del mejor modelo por configuración")
        with col2:
            fig = px.bar(
                m, x="auc", y="config_legible", orientation="h",
                color_discrete_sequence=[PALETA_CATEGORICA[2]],
                text=m["auc"].round(3).values,
                labels={"auc": "AUC (0.5 = azar)", "config_legible": ""},
            )
            fig.update_xaxes(range=[0.5, 1])
            estilizar(fig, height=320)
            chart_card(fig, "AUC del mejor modelo por configuración")

        insight_box(
            "El **target binario** (negativa vs positiva, descartando la reseña neutral) "
            "casi duplica el F1 macro frente a las 3 clases: la clase neutral era la más "
            "ambigua y arrastraba el resto hacia abajo. En cambio **SMOTE no aporta** sobre "
            "`class_weight=\"balanced\"`: sube el accuracy pero no el F1 macro ni el AUC, "
            "porque genera puntos sintéticos sin crear señal nueva."
        )

        st.markdown("##### Tabla completa")
        st.dataframe(
            m[["config_legible", "modelo", "accuracy", "precision_macro",
               "recall_macro", "f1_macro", "auc"]]
            .rename(columns={"config_legible": "configuración"})
            .set_index("configuración").round(3),
            width="stretch",
        )

        st.markdown("##### Matrices de confusión")
        c1, c2 = st.columns(2)
        cm_3c = cargar("matriz_confusion_A_3clases_classweight.csv", index_col=0)
        cm_bin = cargar("matriz_confusion_C_binario_classweight.csv", index_col=0)
        with c1:
            if cm_3c is not None:
                matriz_heatmap(cm_3c, "3 clases + class_weight")
        with c2:
            if cm_bin is not None:
                matriz_heatmap(cm_bin, "Binario + class_weight")

# ============================================================
# TAB 2 — Features "en el instante de la compra" (escenario A)
# ============================================================
with tabs[1]:
    st.subheader("¿Suman las features conocidas al momento de la compra?")
    st.markdown(
        "Objetivo A: predecir la calificación usando **solo lo que se sabe al comprar** "
        "(distancia, precio, flete, producto, historial del vendedor, mes, geografía). "
        "Nada de texto de la reseña ni de la demora real del pedido, que serían fuga de "
        "información. Se parte del target binario (el mejor del experimento anterior) y se "
        "van sumando bloques de features."
    )

    comp = cargar("comparacion_features_compra.csv")
    if comp is None:
        st.warning("Falta `comparacion_features_compra.csv`. Corré `experimento_features_compra.py`.")
    else:
        etapas = {
            "base": "base (pipeline actual)",
            "+vend": "+ vendedor (reseña/categorías)",
            "+temp": "+ temporales (mes/día/temporada)",
            "+geo": "+ geográficas (estado/mismo estado)",
        }
        orden = ["base", "+vend", "+temp", "+geo"]
        # mejor modelo por etapa
        mejor_etapa = (comp.sort_values("f1_macro", ascending=False)
                       .groupby("configuracion", as_index=False).first())
        mejor_etapa["orden"] = mejor_etapa["configuracion"].map({k: i for i, k in enumerate(orden)})
        mejor_etapa = mejor_etapa.sort_values("orden")
        mejor_etapa["etapa"] = mejor_etapa["configuracion"].map(etapas)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.line(
                mejor_etapa, x="etapa", y="f1_macro", markers=True,
                color_discrete_sequence=[CHART_BLUE],
                labels={"f1_macro": "F1 macro", "etapa": ""},
            )
            fig.update_traces(text=mejor_etapa["f1_macro"].round(3), textposition="top center", mode="lines+markers+text")
            estilizar(fig, height=340)
            chart_card(fig, "F1 macro según se agregan features")
        with col2:
            fig = px.line(
                mejor_etapa, x="etapa", y="auc", markers=True,
                color_discrete_sequence=[PALETA_CATEGORICA[2]],
                labels={"auc": "AUC", "etapa": ""},
            )
            fig.update_traces(text=mejor_etapa["auc"].round(3), textposition="top center", mode="lines+markers+text")
            estilizar(fig, height=340)
            chart_card(fig, "AUC según se agregan features")

        base_f1 = mejor_etapa.loc[mejor_etapa["configuracion"] == "base", "f1_macro"].iloc[0]
        geo_f1 = mejor_etapa.loc[mejor_etapa["configuracion"] == "+geo", "f1_macro"].iloc[0]
        base_auc = mejor_etapa.loc[mejor_etapa["configuracion"] == "base", "auc"].iloc[0]
        geo_auc = mejor_etapa.loc[mejor_etapa["configuracion"] == "+geo", "auc"].iloc[0]

        k1, k2, k3 = st.columns(3)
        k1.metric("F1 macro (base → completo)", f"{geo_f1:.3f}", f"{geo_f1 - base_f1:+.3f}")
        k2.metric("AUC (base → completo)", f"{geo_auc:.3f}", f"{geo_auc - base_auc:+.3f}")
        k3.metric("Techo del escenario A", f"AUC ≈ {geo_auc:.2f}", "predecir antes de la entrega es difícil", delta_color="off")

        insight_box(
            "Las features nuevas **sí aportan, de forma modesta pero consistente**: el AUC "
            "sube de ~0.69 a ~0.72. El mayor salto viene de las **temporales** (el mes de "
            "compra resulta informativo). Es una mejora legítima sobre el modelo actual, sin "
            "fuga de información. El techo relativamente bajo es intrínseco: gran parte de lo "
            "que define la reseña (cómo salió la entrega) todavía no ocurrió al momento de comprar."
        )

        st.markdown("##### Tabla completa (todos los modelos y etapas)")
        comp_show = comp.copy()
        comp_show["etapa"] = comp_show["configuracion"].map(etapas).fillna(comp_show["configuracion"])
        st.dataframe(
            comp_show[["etapa", "modelo", "n_features", "accuracy",
                       "precision_macro", "recall_macro", "f1_macro", "auc"]]
            .round(3),
            width="stretch", hide_index=True,
        )

# ============================================================
# TAB 3 — Importancia de variables del mejor modelo del escenario A
# ============================================================
with tabs[2]:
    st.subheader("¿Qué variables pesan para anticipar la reseña?")
    st.markdown(
        "Importancia de las variables del mejor modelo del set completo (escenario A). "
        "Confirma qué señales usa el modelo para predecir la calificación al momento de la compra."
    )

    imp = cargar("feature_importance_compra.csv", index_col=0)
    if imp is None:
        st.warning("Falta `feature_importance_compra.csv`. Corré `experimento_features_compra.py`.")
    else:
        imp = imp.sort_values("importancia").tail(15)
        fig = px.bar(
            imp, x="importancia", y=imp.index, orientation="h",
            color_discrete_sequence=[CHART_BLUE],
            text=imp["importancia"].round(0).astype(int).values,
            labels={"importancia": "Peso en la decisión del modelo", "index": ""},
        )
        estilizar(fig, height=460)
        chart_card(fig, "Top 15 variables por importancia")

        insight_box(
            "La variable más importante es **`hist_resena_prom_vendedor`** (la reseña "
            "histórica promedio del vendedor): el mejor predictor de si un cliente quedará "
            "conforme es cómo quedaron los clientes anteriores de ese mismo vendedor. La "
            "distancia y el costo de flete siguen pesando, y aparece **`mes_compra_num`** "
            "(la feature temporal nueva), lo que valida que agregarla sirvió. La línea más "
            "prometedora para seguir mejorando es sumar más features de comportamiento del vendedor."
        )

# ============================================================
# TAB 4 — Predecir una compra futura ingresando parámetros a mano
# ============================================================
with tabs[3]:
    st.subheader("Predecir la reseña de una compra futura")
    st.markdown(
        "Usá el mejor modelo del escenario A (LightGBM binario, solo datos conocidos al "
        "momento de la compra) para estimar si una compra terminará en reseña **positiva** "
        "o **negativa**. Elegí un vendedor real (se usa su historial) y completá los datos "
        "del pedido."
    )

    bundle = cargar_bundle()
    catalogo = cargar_catalogo()

    if bundle is None or catalogo is None:
        st.warning(
            "Falta el modelo entrenado. Corré primero `python entrenar_modelo_final.py` "
            "para generar `modelo_final_compra.joblib` y `catalogo_vendedores.csv`."
        )
    else:
        met = bundle.get("metricas", {})
        st.caption(
            f"Modelo: {bundle.get('descripcion', 'LightGBM binario')} · "
            f"F1 macro {met.get('f1_macro', float('nan')):.3f} · AUC {met.get('auc', float('nan')):.3f}"
        )

        opciones = bundle["opciones_categoricas"]
        dias = {0: "Lunes", 1: "Martes", 2: "Miércoles", 3: "Jueves",
                4: "Viernes", 5: "Sábado", 6: "Domingo"}

        with st.form("form_prediccion"):
            st.markdown("##### Vendedor (opción a: se usa su historial real)")
            # etiqueta legible por vendedor
            catalogo_ord = catalogo.copy()
            catalogo_ord["etiqueta"] = catalogo_ord.apply(
                lambda r: f"{r['seller_id'][:8]}… · {int(r['hist_pedidos_vendedor'])} pedidos · "
                          f"reseña prom {r['hist_resena_prom_vendedor']:.2f} · "
                          f"{r['hist_pct_demora_vendedor']*100:.0f}% demora · {r['seller_state']}",
                axis=1,
            )
            sel = st.selectbox(
                "Vendedor", options=catalogo_ord.index,
                format_func=lambda i: catalogo_ord.loc[i, "etiqueta"],
            )
            vendedor = catalogo_ord.loc[sel]
            ajustar = st.checkbox("Ajustar manualmente el historial del vendedor", value=False)

            col_v1, col_v2 = st.columns(2)
            if ajustar:
                with col_v1:
                    hist_resena = st.slider("Reseña histórica promedio", 1.0, 5.0,
                                            float(vendedor["hist_resena_prom_vendedor"]), 0.1)
                    hist_demora = st.slider("% histórico de demora", 0.0, 1.0,
                                            float(vendedor["hist_pct_demora_vendedor"]), 0.01)
                with col_v2:
                    hist_pedidos = st.number_input("Pedidos históricos", min_value=0,
                                                   value=int(vendedor["hist_pedidos_vendedor"]))
                    hist_cats = st.number_input("Categorías distintas que vende", min_value=0,
                                                value=int(vendedor["hist_categorias_vendedor"]))
            else:
                hist_resena = float(vendedor["hist_resena_prom_vendedor"])
                hist_demora = float(vendedor["hist_pct_demora_vendedor"])
                hist_pedidos = int(vendedor["hist_pedidos_vendedor"])
                hist_cats = int(vendedor["hist_categorias_vendedor"])
            seller_state = str(vendedor["seller_state"])

            st.markdown("##### Datos del pedido")
            c1, c2, c3 = st.columns(3)
            with c1:
                price = st.number_input("Precio del producto (R$)", min_value=0.0, value=120.0, step=10.0)
                freight = st.number_input("Costo de flete (R$)", min_value=0.0, value=20.0, step=5.0)
                items = st.number_input("Ítems en el pedido", min_value=1, value=1)
            with c2:
                peso = st.number_input("Peso del producto (g)", min_value=0.0, value=800.0, step=100.0)
                volumen = st.number_input("Volumen del producto (cm³)", min_value=0.0, value=5000.0, step=500.0)
                cuotas = st.number_input("Cuotas máximas del pago", min_value=1, value=1)
            with c3:
                distancia = st.number_input("Distancia cliente-vendedor (km)", min_value=0.0, value=500.0, step=50.0)
                categoria = st.selectbox("Categoría del producto", opciones["product_category_name_english"])
                pago = st.selectbox("Medio de pago", opciones["payment_type_principal"])

            c4, c5, c6 = st.columns(3)
            with c4:
                customer_state = st.selectbox("Estado del cliente", opciones["customer_state"])
            with c5:
                mes = st.selectbox("Mes de compra", list(range(1, 13)), index=0)
            with c6:
                dia = st.selectbox("Día de la semana", list(dias.keys()),
                                   format_func=lambda d: dias[d])

            enviado = st.form_submit_button("Predecir reseña", type="primary")

        if enviado:
            entrada = {
                "price": price, "freight_value": freight, "product_weight_g": peso,
                "volumen_cm3": volumen, "distancia_km": distancia, "items_por_pedido": items,
                "payment_installments_max": cuotas, "mes_compra_num": mes, "dia_semana": dia,
                "product_category_name_english": categoria, "payment_type_principal": pago,
                "customer_state": customer_state, "seller_state": seller_state,
                "hist_pct_demora_vendedor": hist_demora, "hist_pedidos_vendedor": hist_pedidos,
                "hist_resena_prom_vendedor": hist_resena, "hist_categorias_vendedor": hist_cats,
            }
            proba_pos = predecir_compra(bundle, entrada)
            proba_neg = 1 - proba_pos
            positiva = proba_pos >= 0.5

            col_r1, col_r2 = st.columns([1, 1.2])
            with col_r1:
                if positiva:
                    st.success(f"### Reseña probablemente POSITIVA\nProbabilidad: {proba_pos:.1%}")
                else:
                    st.error(f"### Reseña probablemente NEGATIVA\nProbabilidad: {proba_neg:.1%}")
                st.caption(
                    f"Cliente en {customer_state} · vendedor en {seller_state} "
                    f"({'mismo estado' if customer_state == seller_state else 'estados distintos'})"
                )
            with col_r2:
                fig = go.Figure(go.Bar(
                    x=[proba_neg, proba_pos], y=["Negativa", "Positiva"], orientation="h",
                    marker_color=[ROJO, VERDE],
                    text=[f"{proba_neg:.1%}", f"{proba_pos:.1%}"], textposition="auto",
                ))
                fig.update_xaxes(range=[0, 1], tickformat=".0%")
                estilizar(fig, height=200, showlegend=False)
                chart_card(fig, "Probabilidad estimada por el modelo")

            insight_box(
                "Recordá que este modelo predice **al momento de la compra**, sin conocer cómo "
                "saldrá la entrega (AUC ≈ 0.72). Es una estimación de riesgo, no una certeza: "
                "sirve para señalar qué pedidos vigilar, no para garantizar el resultado. La "
                "reseña histórica del vendedor es la variable que más influye en la predicción."
            )
