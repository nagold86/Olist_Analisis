# Olist E-Commerce — EDA completo + modelos + dashboard

Proyecto integrador — Tecnicatura en Ciencia de Datos e Inteligencia Artificial.
Las 9 tablas originales van en `data/raw/` (bajadas de Kaggle): no hace falta crear cuenta ni token de la
API de Kaggle para correrlo, alcanza con clonar/descomprimir esta carpeta, colocar los CSV ahí y ejecutar
los scripts.

El análisis corre en dos formatos equivalentes: como **scripts de Python puro** (sin Jupyter, pensados
para andar directo en la terminal, sin depender de un servidor ni de un kernel) o como **pipeline de 11
Jupyter Notebooks** (carpeta `notebooks/`, ver más abajo), con el mismo análisis paso a paso y la
explicación expandida en celdas markdown. En ambos casos, cada bloque de código tiene, antes, un
comentario con **qué se hace** y **para qué**, y después (cuando corresponde) uno con el **insight**
que deja ese resultado — se puede leer de punta a punta, sin necesitar el informe al lado.

> El informe funcional (`Informe_Funcional_Olist_Natalia.docx`, que se entrega junto con este repo)
> resume todo esto en un solo documento, con el mismo criterio de "qué / para qué / insight" por
> cada fragmento de código, más el contexto de negocio y las conclusiones finales.

## Estructura del proyecto

```
olist-eda-completo/
├── eda_completo.py              # Script 1: unificación de las 9 tablas + EDA completo
├── notebooks/                   # Alternativa en Jupyter: el EDA paso a paso en 11 notebooks
│   ├── 00_presentacion_del_pipeline.ipynb      # Presentación del pipeline + verificación del entorno
│   ├── 01_configuracion_inicial.ipynb          # Configuración compartida (sección 0)
│   ├── 02_unificacion_de_las_9_tablas.ipynb    # Unificación de las 9 tablas (sección 1)
│   ├── 03_carga_y_limpieza.ipynb               # Carga + limpieza (secciones 2-3)
│   ├── 04_feature_engineering.ipynb            # 9 variables nuevas (sección 4)
│   ├── 05_estadisticas_descriptivas.ipynb      # Estadísticos y conteos (sección 5)
│   ├── 06_visualizaciones_exploratorias.ipynb  # 8 gráficos exploratorios (sección 6)
│   ├── 07_analisis_exhaustivo_logistica.ipynb  # Logística, precio y vendedores (sección 7)
│   ├── 08_mapa_geografico.ipynb                # Mapas Plotly/Folium + resumen por estado (sección 8)
│   ├── 09_nlp_sobre_las_resenas.ipynb          # NLP sobre reseñas en portugués (sección 9)
│   └── 10_contraste_de_hipotesis.ipynb         # Tests estadísticos + conclusiones (sección 10)
├── modelos_predictivos.py       # Script 2: clasificación (envío demorado) + clustering (vendedores)
├── app.py                       # Dashboard en Streamlit con todo el análisis y los modelos
├── .streamlit/
│   └── config.toml              # Tema visual (paleta de Olist)
├── data/
│   ├── raw/                            # Las 9 tablas originales de Kaggle (colocarlas acá)
│   ├── pipeline/                       # Artefactos intermedios del pipeline de notebooks
│   └── olist_dataset_unificado.csv     # Se genera la primera vez (script 1 / notebook 02)
├── figuras_eda/                  # PNG de cada gráfico (fig_XX del script 1, nbXX_ de los notebooks)
├── figuras_modelos/              # PNG de cada gráfico del script 2 (se generan al correrlo)
├── modelos_output/               # CSV con resultados de los modelos (los genera el script 2)
├── requirements.txt
└── .gitignore
```

## Requisitos

- Python 3.10 o superior.

## Instalación

```bash
# 1. Cloná o descomprimí este proyecto y entrá a la carpeta
cd olist-eda-completo

# 2. Creá un entorno virtual (recomendado)
python -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate

# 3. Instalá las dependencias
pip install -r requirements.txt
```

## Uso

### 1. Script de EDA

```bash
python eda_completo.py
```

Tarda unos segundos (no hay servidor ni kernel de por medio). La primera vez, la sección 1 lee las 9
tablas de `data/raw/`, las une y guarda `data/olist_dataset_unificado.csv`; si ese archivo ya existe (por
ejemplo en una corrida posterior), el script salta directo a cargarlo, sin volver a unificar nada.

Como un script no tiene dónde "mostrar" los gráficos en pantalla, cada uno se guarda automáticamente
como imagen en la carpeta `figuras_eda/` (`fig_01.png`, `fig_02.png`, ...) en el mismo orden en que se
generan, para poder revisarlos abriéndolos como cualquier imagen. Las estadísticas, tablas e insights se
imprimen en la terminal a medida que se ejecuta.

### 2. Script de modelos

```bash
python modelos_predictivos.py
```

Es un script aparte (autocontenido: vuelve a armar el dataset unificado si hace falta, sin depender de
haber corrido antes el script de EDA) con dos modelos:

- **Clasificación** — Decision Tree, Random Forest y XGBoost para predecir si una reseña va a ser
  negativa, neutral o positiva, comparados por F1 macro, con matriz de confusión e importancia de
  variables.
- **Clustering (K-Means)** — segmentación de vendedores según volumen, ticket promedio, demora y
  review score, eligiendo la cantidad de clusters con el método del codo + silhouette score.

Sus gráficos se guardan en `figuras_modelos/`, con el mismo criterio que el script de EDA. Al final
exporta los resultados a `modelos_output/` (CSV de métricas, importancia de variables, matriz de
confusión, predicciones, resumen y asignación de clusters) para que el dashboard de Streamlit los pueda
mostrar sin tener que reentrenar los modelos cada vez.

### 3. Dashboard en Streamlit

```bash
streamlit run app.py
```

Corré primero los dos scripts al menos una vez (para tener `data/olist_dataset_unificado.csv` y
`modelos_output/`). El dashboard tiene 6 pestañas — Resumen, Exploración de datos, Mapa, Modelo de
reseña del pedido, Modelo de segmentación de vendedores y Conclusiones — pensadas como una presentación
para alguien que no sabe de ciencia de datos: cada hallazgo técnico viene acompañado de una explicación
en criollo y, cuando corresponde, de una "oportunidad" concreta de negocio (con el ícono 🎯) que se
puede accionar a partir de ese hallazgo. La última pestaña resume las conclusiones principales y una
lista de próximos pasos sugeridos.

Tiene el logo de Olist en el encabezado, un tema oscuro y una paleta de colores derivada de la
identidad de olist.com (azul `#0A4EE4` como color principal), validada para buen contraste y para
personas con daltonismo. La tipografía usa dos fuentes de Google Fonts: **Blinker** (títulos, pestañas
y las "pastillas" que encabezan cada gráfico) y **Plus Jakarta Sans** (el resto del texto) — son una
aproximación, no se pudo confirmar la fuente exacta que usa olist.com. Cada gráfico va dentro de una
tarjeta con bordes redondeados, con su título en una pastilla separada arriba (en vez de ir como título
dentro del gráfico de Plotly), para que se lea más como una pieza de diseño que como un gráfico técnico
suelto. El logo y las fotos ilustrativas (vendedores, envíos, clientes, reseñas) están incrustadas en
base64 dentro del propio `app.py` — pasadas a un duotono en el azul de la paleta, recortadas a una
relación de aspecto pareja (4:3) para que no se vean cortadas, con los bordes difuminados hacia el
fondo oscuro — así que `app.py` funciona solo, sin depender de una carpeta `assets/` aparte.

### 4. Pipeline de Jupyter Notebooks (alternativa al script 1)

```bash
pip install jupyterlab        # si todavía no lo tenés (ya viene en requirements.txt)
jupyter lab
```

La carpeta `notebooks/` desarma `eda_completo.py` en **11 cuadernos ordenados (00 → 10)**, con la misma
lógica de "qué / para qué / insight" pero explicada en celdas markdown y con los resultados (tablas,
gráficos, tests) visibles dentro del cuaderno. Se ejecutan **en orden**: cada uno consume el archivo que
genera el anterior (`data/olist_dataset_unificado.csv` → `data/pipeline/01_df_limpio.csv` →
`data/pipeline/02_df_features.csv`). Los gráficos se guardan en `figuras_eda/` con prefijo `nbXX_`, así
no se pisan con los `fig_XX.png` del script. El notebook `00_presentacion_del_pipeline.ipynb` documenta
el mapa completo del pipeline y verifica el entorno antes de arrancar.

## Si el dataset se actualiza en Kaggle

Los CSV de `data/raw/` fueron bajados una sola vez desde
[Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). Si en algún momento hace falta
una versión más nueva: entrar a esa página logueado con una cuenta de Kaggle (no requiere token de API),
tocar "Download", descomprimir y reemplazar los archivos de `data/raw/`.

## Autora

Natalia — Tecnicatura en Ciencia de Datos e Inteligencia Artificial.
