# Pendientes — mejorar informe y presentación

> Nota de traspaso para una próxima sesión. El proyecto quedó unificado y funcionando;
> lo único a mejorar son los dos documentos finales (`.docx` y `.pptx`).

## Estado actual (lo que ya está hecho y OK)

- **Proyecto unificado** en la rama `experimento/smote-binario`, bajo un criterio único,
  sin referencias a scripts/versiones/intentos anteriores.
- **Pipeline de notebooks `00 → 11`** (todo el análisis vive acá, sin scripts `.py`):
  - `00`–`10`: EDA completo (unificación, limpieza, features, estadísticas, gráficos,
    logística, mapas, NLP, contraste de hipótesis).
  - `11_modelado.ipynb`: clasificación binaria de reseña (**LightGBM**) + segmentación de
    vendedores (**K-Means**). Entrena y **serializa** el modelo en `modelos_output/`.
- **`app.py`**: dashboard Streamlit único, rediseñado, 7 pestañas (incluye "Predecir una
  compra" interactiva). Lee de `modelos_output/`, no reentrena. Verificado: compila y predice.
- **Entregables generados** (los que hay que mejorar):
  - `Informe_Olist.docx` — generado con `python-docx`. 6 secciones + 3 tablas + portada.
  - `Presentacion_Olist.pptx` — generada con `python-pptx`. 9 diapositivas, tema Olist azul.
- `README.md` y `requirements.txt` actualizados. Documentos viejos eliminados.
- **No se hizo commit.** Todo está en el working tree.

## Resultados clave (para no recalcularlos)

- **Modelo de reseña (LightGBM binario, features al momento de la compra):**
  AUC ≈ 0,71 · F1 macro ≈ 0,62 · accuracy ≈ 0,72.
  Variable más influyente: `hist_resena_prom_vendedor` (reseña histórica del vendedor).
- **Clustering de vendedores (K-Means, 3 segmentos):**
  - Rápidos y chicos (~1.700): entrega veloz, sin demora, buena reseña.
  - Alto volumen (~800): muchos pedidos, desempeño sólido.
  - Rezagados (~460): ~20 días de entrega, ~20% demora, reseña ~3,5. **Grupo de riesgo prioritario.**
- **Hipótesis del EDA:** H1 (demora reduce satisfacción) confirmada; H2 (reseñas negativas
  hablan de demoras) evidenciada por NLP; H3 (fuera del Sudeste, más días y flete) confirmada;
  adicional (más distancia ⇒ más días, Spearman) confirmada.
- Casi 1 de cada 10 pedidos entregados llega tarde.
- Autora: Natalia.

## Qué falta mejorar

Los dos documentos hoy son **solo texto y tablas**: no incrustan ningún gráfico. Ese es el
principal salto de calidad pendiente.

- [ ] **Incrustar gráficos reales** en informe y presentación (hay 18 PNG en `figuras_eda/`
      una vez que se corren los notebooks; faltaría también exportar los del modelado del nb 11).
- [ ] Revisar redacción, jerarquía y extensión del informe.
- [ ] Mejorar el diseño visual de la presentación (plantillas, densidad de datos, imágenes).
- [ ] Definir con el usuario: ¿cuánta extensión? ¿qué gráficos sí/no? ¿algún estilo o plantilla?

## Herramientas / skills (importante)

- **Informe `.docx` → usar la skill `docx`** (es la única skill instalada que aplica).
  - **Falta tooling de verificación:** este entorno NO tiene `pandoc` ni LibreOffice (`soffice`),
    así que no se puede renderizar a PDF para revisar el resultado visual.
    **Antes de mejorar el informe, instalar pandoc o LibreOffice** para poder convertir a PDF,
    mirarlo y corregir sobre lo que realmente se ve.
  - `python-docx` 1.2.0 está instalado (es lo que se usó esta vez).
- **Presentación `.pptx` → NO hay skill de presentaciones.**
  - Opción A: seguir con `python-pptx` (1.0.2, ya instalado) — control total, diseño a mano.
  - Opción B: buscar una skill de PPT con la skill `find-skills` (o skills.sh) e instalarla.
- Las demás skills instaladas (`emil-design-eng`, `apple-design`, `animate`, `prototype`,
  `mobile-native`, etc.) son para **UI web/React**: NO aplican a `.docx`/`.pptx`.

## Entorno

- Windows · PowerShell · Python 3.14.7.
- Instalado: pandas, numpy, scipy, scikit-learn, lightgbm 4.7, joblib, matplotlib, seaborn,
  plotly, streamlit, nbformat, nbconvert, python-docx 1.2.0, python-pptx 1.0.2.
- Faltante para verificar documentos: **pandoc** y/o **LibreOffice**.

## Prompt sugerido para la próxima sesión

> "Mejorá `Informe_Olist.docx` y `Presentacion_Olist.pptx`. Activá la skill `docx` para el
> informe. Primero instalá pandoc o LibreOffice para poder verificar el render a PDF. Quiero
> que ambos incrusten los gráficos reales del análisis. Antes de empezar, mostrame qué cambios
> de contenido y diseño proponés."
