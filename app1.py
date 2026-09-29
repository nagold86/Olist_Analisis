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
