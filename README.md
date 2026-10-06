# Olist E-Commerce — EDA + modelado + dashboard

Proyecto integrador — Tecnicatura en Ciencia de Datos e Inteligencia Artificial.

Análisis completo del marketplace brasileño **Olist** sobre su dataset público (2016–2018): desde la
unificación de las 9 tablas originales hasta el modelado predictivo y un dashboard interactivo. Las 9
tablas van en `data/raw/` (bajadas de [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce),
sin token de API): alcanza con colocarlas ahí y ejecutar los notebooks.

Todo el análisis vive en un **pipeline de notebooks Jupyter (00 → 11)**, con el mismo criterio de
explicación en cada paso: antes de cada bloque de código, un comentario de **qué se hace** y **para
qué**; después, el **insight** que deja el resultado. El dashboard en Streamlit (`app.py`) presenta el
análisis y los modelos para alguien que no sabe de ciencia de datos, e incluye una pestaña de
predicción interactiva. El informe funcional (`Informe_Olist.docx`) resume todo en un solo documento.

## Estructura del proyecto

```
Olist_Analisis/
├── notebooks/                              # Todo el análisis, paso a paso (ejecutar en orden)
│   ├── 00_presentacion_del_pipeline.ipynb      # Presentación + verificación del entorno
│   ├── 01_configuracion_inicial.ipynb          # Configuración compartida (imports, rutas, paleta, estilo)
│   ├── 02_unificacion_de_las_9_tablas.ipynb    # Integración de las 9 tablas
│   ├── 03_carga_y_limpieza.ipynb               # Carga + limpieza
│   ├── 04_feature_engineering.ipynb            # Variables nuevas
│   ├── 05_estadisticas_descriptivas.ipynb      # Estadísticos y conteos
│   ├── 06_visualizaciones_exploratorias.ipynb  # Gráficos exploratorios
│   ├── 07_analisis_exhaustivo_logistica.ipynb  # Logística, precio y vendedores
│   ├── 08_mapa_geografico.ipynb                # Mapas (Plotly + Folium) y resumen por estado
│   ├── 09_nlp_sobre_las_resenas.ipynb          # NLP sobre reseñas en portugués
│   ├── 10_contraste_de_hipotesis.ipynb         # Tests estadísticos + conclusiones del EDA
│   └── 11_modelado.ipynb                       # Clasificación de reseña (LightGBM) + segmentación (K-Means)
├── app.py                                  # Dashboard en Streamlit (lee modelos_output/)
├── modelos_output/                         # Resultados del modelado (los genera el notebook 11)
├── data/
│   ├── raw/                                    # Las 9 tablas originales de Kaggle (colocarlas acá)
│   ├── pipeline/                               # Artefactos intermedios del pipeline
│   └── olist_dataset_unificado.csv             # Se genera en el notebook 02
├── figuras_eda/                            # PNG de los gráficos de los notebooks (prefijo nbXX_)
├── .streamlit/config.toml                  # Tema visual (paleta de Olist)
├── Informe_Olist.docx                      # Informe funcional final
├── Presentacion_Olist.pptx                 # Presentación final
├── requirements.txt
└── .gitignore
```

## Requisitos

- Python 3.10 o superior.

## Instalación

```bash
# 1. Entrá a la carpeta del proyecto
cd Olist_Analisis

# 2. Creá un entorno virtual (recomendado)
python -m venv .venv
.venv\Scripts\activate        # en Linux/macOS: source .venv/bin/activate

# 3. Instalá las dependencias
pip install -r requirements.txt
```

## Uso

### 1. Pipeline de notebooks

```bash
jupyter lab
```

Entrá a `notebooks/` y ejecutá los cuadernos **en orden** (00 → 11). Cada uno consume el artefacto que
genera el anterior:

```
data/olist_dataset_unificado.csv  (nb 02)
    → data/pipeline/01_df_limpio.csv  (nb 03)
        → data/pipeline/02_df_features.csv  (nb 04)
            → EDA (nb 05–10) y modelado (nb 11)
```

El **notebook 11** entrena los dos modelos y guarda sus resultados en `modelos_output/`, incluyendo el
modelo serializado (`modelo_final.joblib`) y el catálogo de vendedores (`catalogo_vendedores.csv`), de
modo que el dashboard no tenga que reentrenar nada.

### 2. Dashboard en Streamlit

```bash
streamlit run app.py
```

Ejecutá antes el pipeline al menos una vez (para tener `data/olist_dataset_unificado.csv` y
`modelos_output/`). El dashboard tiene siete pestañas — Resumen, Exploración de datos, Mapa de
clientes, Modelo de reseña, Predecir una compra, Segmentación de vendedores y Conclusiones — pensadas
como una presentación para alguien que no sabe de ciencia de datos: cada hallazgo técnico viene con una
explicación en criollo y, cuando corresponde, una oportunidad de negocio accionable. La pestaña
**Predecir una compra** usa el modelo entrenado para estimar, de forma interactiva, si una compra
terminará en reseña positiva o negativa.

El dashboard usa el logo de Olist y una paleta derivada de la identidad de olist.com (azul `#0A4EE4`).
El logo y las fotos ilustrativas están embebidos en base64 dentro de `app.py`, así que la app funciona
con un solo archivo, sin depender de una carpeta `assets/` aparte.

## Si el dataset se actualiza en Kaggle

Los CSV de `data/raw/` fueron bajados una sola vez desde
[Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). Para una versión más nueva:
entrar a esa página logueado (no requiere token de API), descargar, descomprimir y reemplazar los
archivos de `data/raw/`.

## Autora

Natalia — Tecnicatura en Ciencia de Datos e Inteligencia Artificial.
