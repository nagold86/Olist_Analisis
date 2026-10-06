# Guion de la Presentación — Olist E-Commerce
**20 minutos · 4 presentadores · ~21 slides**

---

## Distribución de presentadores (sugerida)

| Bloque | Slides | Duración | Presentador |
|---|---|---|---|
| Apertura + datos + metodología | 1–5 | ~4 min | Presentador A |
| Análisis exploratorio | 6–10 | ~5 min | Presentador B |
| Contraste de hipótesis | 11–13 | ~3 min | Presentador C |
| Modelado + segmentación + cierre | 14–21 | ~8 min | Presentador D |

Cada presentador hace la transición al siguiente con una frase puente ("Ahora [nombre] les va a contar cómo…").

---

## SLIDE 1 — Portada
**En el slide:**
- Título grande: **Olist E-Commerce**
- Subtítulo: Análisis exploratorio, modelado predictivo y segmentación del marketplace brasileño
- Pie: Tecnicatura en Ciencia de Datos e IA · Proyecto Integrador · Octubre 2026
- Nombres de los 4 integrantes

**Speech (Presentador A, ~30 seg):**
> Buenas tardes. Somos [nombres] y vamos a presentar nuestro proyecto integrador sobre el marketplace brasileño Olist. En los próximos 20 minutos les vamos a contar qué encontramos analizando más de 100.000 pedidos reales, qué hipótesis pudimos confirmar y qué modelos construimos para anticipar problemas antes de que ocurran.

---

## SLIDE 2 — ¿Qué es Olist?
**En el slide:**
- Diagrama simple: Vendedor → Olist (plataforma + logística) → Comprador
- 3 cifras clave: ~100.000 pedidos · ~3.000 vendedores · 2016–2018
- Fuente: Kaggle (dataset público)

**Speech (~45 seg):**
> Olist es un marketplace brasileño: conecta a miles de pequeños comercios con compradores de todo Brasil y coordina el envío. No es como MercadoLibre donde cada vendedor gestiona su logística: acá Olist centraliza la experiencia. El dataset es público, viene de Kaggle, y cubre dos años de operación con 9 tablas relacionales: pedidos, productos, pagos, reseñas, vendedores, clientes y geolocalización. El anonimato está garantizado, las coordenadas son del código postal, no de direcciones reales.

---

## SLIDE 3 — Problema y objetivos
**En el slide:**
- Título: "¿Por qué importan las reseñas?"
- Frase destacada: "Un marketplace cuya reputación cae pierde compradores y, con ellos, vendedores."
- 3 objetivos con íconos simples:
  1. Explorar qué factores explican las reseñas negativas
  2. Contrastar hipótesis con tests estadísticos
  3. Construir un modelo predictivo + segmentación de vendedores

**Speech (~40 seg):**
> La métrica central de un marketplace es la satisfacción del cliente, que se mide con las reseñas. Si las reseñas caen, los compradores se van. Nuestro proyecto tiene tres objetivos: primero, explorar los datos para entender qué hace que un cliente deje una reseña negativa. Segundo, contrastar esas intuiciones con tests estadísticos formales. Y tercero, construir dos modelos: uno que anticipe si una compra va a terminar mal usando solo lo que se sabe al momento de comprar, y otro que agrupe a los vendedores por perfil para identificar a los de mayor riesgo.

---

## SLIDE 4 — Pipeline de trabajo
**En el slide:**
- Diagrama horizontal del pipeline: 9 tablas → Unificación → Limpieza → Feature engineering → EDA → Hipótesis → Modelado → Dashboard
- Debajo: "12 notebooks reproducibles (00 → 11)"

**Speech (~40 seg):**
> Todo el trabajo se organiza como un pipeline de 12 notebooks en Jupyter. Arrancamos con las 9 tablas crudas, las unificamos en un dataset de 112.650 filas a nivel de ítem de pedido, limpiamos, construimos 9 variables nuevas, hicimos el análisis exploratorio, contrastamos hipótesis y terminamos con los modelos. Cada notebook consume el artefacto del anterior, así que cualquiera puede reproducir todo desde cero. Al final, el modelo se serializa y alimenta un dashboard interactivo en Streamlit.

---

## SLIDE 5 — Los datos: estructura y limpieza
**En el slide:**
- Tabla compacta de las 9 tablas originales con filas y granularidad
- 3 cifras de limpieza destacadas: 0 duplicados · 42% con comentario escrito · 7 columnas eliminadas
- Resultado: 112.650 filas × 43 columnas

**Speech (Presentador A, ~50 seg, y cierra con transición):**
> Las 9 tablas se cruzan con left joins sobre la tabla de ítems. Pagos y reseñas se agregan primero a nivel pedido para no duplicar filas. La geolocalización se une dos veces: una para el cliente y otra para el vendedor. En la limpieza no se borró ni se inventó ningún dato: los nulos se mantienen donde corresponde, como en pedidos no entregados. Construimos 9 variables nuevas: tiempo de entrega, si llegó tarde, la distancia real cliente-vendedor por Haversine, el ratio flete/precio, y la reseña binarizada excluyendo los neutrales. El dataset final tiene 112.650 filas y 43 columnas. Ahora [Presentador B] les va a mostrar qué encontramos al explorar estos datos.

---

## SLIDE 6 — Evolución del negocio
**En el slide:**
- Gráfico: serie mensual de pedidos (nb06_02.png)
- Anotación visual en el pico de noviembre 2017: "Black Friday"

**Speech (Presentador B, ~40 seg):**
> El volumen de pedidos creció de forma sostenida desde finales de 2016 hasta un pico de 7.450 pedidos en noviembre de 2017, que coincide con el Black Friday. Después de eso se estabiliza alrededor de los 6.500 mensuales. Es un marketplace que ya alcanzó cierta madurez dentro del período que observamos.

---

## SLIDE 7 — Satisfacción y reseñas
**En el slide:**
- Gráfico de barras de distribución de review_score (nb06_04.png)
- Cifra destacada: "75% califican 4-5 ★, pero el 16% pone 1-2 ★"

**Speech (~40 seg):**
> Las reseñas están muy sesgadas hacia arriba: más de la mitad da 5 estrellas. Pero hay un 16% de reseñas negativas, que para un marketplace es un problema serio. Descartamos las de 3 estrellas porque son ambiguas: no son ni conformes ni disconformes. Esto nos deja con un target binario claro: positiva o negativa.

---

## SLIDE 8 — La logística es el problema central
**En el slide:**
- Dos elementos lado a lado:
  - Boxplot: review_score según envío demorado (nb06_06.png)
  - Cifra grande: "7,9% de envíos llegan tarde → caen de 4,21 a 2,55 ★"

**Speech (~50 seg):**
> Y acá está el hallazgo más importante del proyecto. Casi 1 de cada 10 pedidos llega después de la fecha prometida. Y cuando eso pasa, la calificación se desploma: de una mediana de 5 estrellas a una mediana de 2. La diferencia es de 1,7 estrellas en promedio. Esto no es un hallazgo menor: significa que la logística es el principal driver de la satisfacción, mucho más que el precio o la categoría del producto. Y lo confirmamos con los textos de las reseñas: cuando el cliente se queja, habla de plazos y entregas, no del producto.

---

## SLIDE 9 — Brecha geográfica
**En el slide:**
- Tabla comparativa: Sudeste vs. Resto del país (flete, días de entrega, % demorado, ticket promedio)
- Cifra destacada: "Fuera del Sudeste pagan más flete, esperan más y gastan lo mismo"

**Speech (~45 seg):**
> La geografía explica buena parte del problema. Los clientes del Sudeste reciben en 10 días promedio. Fuera del Sudeste, 16 días. Pagan R$ 26 de flete contra R$ 17. Y el porcentaje de demora es mayor. Pero el ticket de compra es similar o incluso más alto. O sea: el cliente periférico gasta lo mismo, pero recibe un servicio peor. Hay una brecha que se puede y se debe cerrar.

---

## SLIDE 10 — NLP: las palabras de la queja
**En el slide:**
- Nubes de palabras lado a lado: negativas (rojo) vs. positivas (azul) (nb09_02.png)
- 3 palabras destacadas de las negativas: "não · recebi · entregue · prazo"

**Speech (Presentador B, ~40 seg, transición al C):**
> Procesamos los comentarios en portugués con NLTK y los resultados son elocuentes. En las reseñas negativas dominan "não", "recebi", "entregue", "prazo", "chegou": todo vocabulario de entrega. En las positivas: "excelente", "qualidade", "recomendo", "perfeito". El cliente no se queja del producto, se queja de la logística. Esto es evidencia cualitativa que complementa los números. Ahora [Presentador C] va a contar cómo llevamos todo esto a tests estadísticos.

---

## SLIDE 11 — Hipótesis planteadas
**En el slide:**
- 4 hipótesis listadas con formato visual limpio:
  - H1: La demora reduce la satisfacción
  - H2: Las reseñas negativas hablan de demoras
  - H3: Fuera del Sudeste, más tiempo y flete
  - H4: Más distancia → más tiempo
- Método: Mann-Whitney U + Spearman (no paramétricos, α = 0,05)

**Speech (Presentador C, ~30 seg):**
> Formulamos cuatro hipótesis a partir del EDA. Usamos tests no paramétricos porque ni las calificaciones ni los tiempos de entrega cumplen los supuestos de normalidad que necesitan los tests clásicos. Mann-Whitney para comparar dos grupos y Spearman para medir correlación monótona. Todas las pruebas son de una cola, en la dirección de la hipótesis.

---

## SLIDE 12 — Resultados de las hipótesis
**En el slide:**
- Tabla resumen: Hipótesis | Test | Estadístico/ρ | p-valor | Veredicto (✓)
- Las 4 filas con sus datos concretos

**Speech (~50 seg):**
> H1: los pedidos a tiempo tienen una calificación promedio de 4,21 contra 2,55 de los demorados. Mann-Whitney con p-valor prácticamente cero. Confirmada. H2 se evidencia cualitativamente con el análisis de texto que ya vieron. H3: el Sudeste entrega en 10 días contra 16 del resto, con diferencia estadísticamente significativa. Y la hipótesis adicional, Spearman entre distancia y tiempo de entrega, da un ρ de 0,54 con p-valor cero: cada kilómetro extra se asocia a más días de espera. Las cuatro hipótesis se confirman.

---

## SLIDE 13 — Limitaciones metodológicas
**En el slide:**
- 3 puntos concisos (frases cortas):
  - "Tests a nivel de ítem → ítems del mismo pedido no son independientes"
  - "Datos 2016-2018 → snapshot histórico"
  - "H2 es cualitativa (no hay clasificador de sentimiento)"

**Speech (Presentador C, ~30 seg, transición al D):**
> Antes de pasar al modelado, vale aclarar las limitaciones. Los tests se corrieron a nivel de ítem, así que los ítems de un mismo pedido no son estrictamente independientes. Los datos son de hace varios años, así que el marketplace puede haber cambiado. Y la hipótesis H2 es cualitativa: no usamos un clasificador de sentimiento. Ahora [Presentador D] les va a contar los modelos.

---

## SLIDE 14 — Modelo 1: clasificación de la reseña
**En el slide:**
- Título: "¿Se puede anticipar una mala reseña antes de la entrega?"
- Esquema visual: Variables al comprar → LightGBM → Riesgo de reseña negativa
- Cifra: "117 variables · 76.749 train · 25.584 test"
- Destacado: "Sin fuga de información: no usamos nada que ocurra después de la compra"

**Speech (Presentador D, ~50 seg):**
> El primer modelo intenta responder una pregunta operativa: ¿se puede saber al momento de la compra si ese pedido va a terminar en una reseña negativa? Usamos solo variables disponibles al comprar: precio, flete, distancia, categoría, medio de pago, mes y el historial pasado del vendedor. Excluimos explícitamente todo lo que pasa después: la demora real, el texto de la reseña, la calificación. Entrenamos un LightGBM con 300 árboles y class_weight balanced porque el 82% de las reseñas son positivas. El historial del vendedor se calculó solo con datos de entrenamiento para evitar la fuga.

---

## SLIDE 15 — Resultados del clasificador
**En el slide:**
- Curva ROC (nb11_01_curva_roc.png) a la izquierda
- Métricas a la derecha: AUC = 0,72 · F1 macro = 0,62
- Frase: "Modesto pero legítimo: el factor que más influye (si llegó tarde) aún no ocurrió"

**Speech (~40 seg):**
> El AUC es 0,72 y el F1 macro 0,62. Son métricas modestas, pero hay que poner en contexto: estamos prediciendo la reseña sin saber lo más importante, que es si el envío va a llegar tarde. Esa información simplemente no existe al momento de la compra. Aun así, el modelo separa significativamente mejor que el azar, lo que lo hace útil como herramienta de priorización.

---

## SLIDE 16 — ¿Qué mira el modelo?
**En el slide:**
- Gráfico de importancia de variables (nb11_03_importancia_variables.png)
- Destacado: "El mejor predictor de si un cliente va a quedar conforme es cómo quedaron los clientes anteriores de ese vendedor"

**Speech (~40 seg):**
> La variable más influyente es la reseña histórica promedio del vendedor. O sea: el mejor predictor de la satisfacción de un cliente nuevo es cómo les fue a los clientes anteriores con ese mismo vendedor. Le siguen la distancia, el flete, el mes de compra y el ratio flete/precio. Las cuatro variables de historial del vendedor aparecen en el top 15. El mensaje para el negocio es claro: la trayectoria del vendedor es la señal más informativa que se tiene al momento de la compra.

---

## SLIDE 17 — Valor operativo: curva de captura
**En el slide:**
- Gráfico de curva de captura (nb11_05_curva_captura.png)
- 3 cifras destacadas:
  - Revisando el 10% → se detecta el 31% de las negativas
  - Revisando el 20% → se detecta el 47%
  - Revisando el 30% → se detecta el 57%

**Speech (~40 seg):**
> Esto es lo que le importa a un gerente de operaciones. Si usamos el modelo para priorizar y contactamos proactivamente al 20% de los pedidos de mayor riesgo, estamos cubriendo casi la mitad de las experiencias que van a terminar mal, antes de que la reseña esté escrita. Es un sistema de alerta temprana: no predice con certeza, pero focaliza el esfuerzo donde más rinde.

---

## SLIDE 18 — Modelo 2: segmentación de vendedores
**En el slide:**
- Gráfico de codo + silueta (nb11_06_codo_silhouette.png) pequeño arriba
- Tabla del perfil de los 3 segmentos (la misma del informe)
- Etiquetas: Ágiles · Consolidados · Rezagados

**Speech (~50 seg):**
> El segundo modelo agrupa a los vendedores según su perfil logístico y de negocio. Usamos K-Means con 8 variables: tiempo de entrega, porcentaje de demora, distancia, flete, pedidos, reseña, precio y categorías. Elegimos 3 segmentos por interpretabilidad. Los "Ágiles" son el 58%: pocos pedidos, entrega rápida, buena reseña. Los "Consolidados" son el 27% pero mueven el 85% de los ítems: alto volumen, desempeño sólido. Y los "Rezagados" son solo el 16%, pero tienen 20 días de entrega promedio, 24% de demora y reseña de 3,5.

---

## SLIDE 19 — Los rezagados: grupo de riesgo
**En el slide:**
- Gráfico de perfil comparativo (nb11_07_perfil_segmentos.png)
- Scatter de vendedores (nb11_08_mapa_vendedores.png) debajo
- Frase: "461 vendedores que concentran el riesgo"

**Speech (~30 seg):**
> Lo que hace valiosa esta segmentación es que identifica una lista concreta de 461 vendedores que concentran los peores indicadores. No son los que más venden: de hecho, tienen un volumen bajo. Pero cada pedido que pasa por ellos tiene el doble de probabilidad de llegar tarde. Son pocos, son identificables y son el primer lugar donde intervenir.

---

## SLIDE 20 — Oportunidades de negocio
**En el slide:**
- 4 oportunidades con ícono y frase corta:
  1. Alerta temprana: contactar al 20% de mayor riesgo antes de la entrega
  2. Programa para rezagados: acompañamiento con umbrales de desempeño
  3. Foco logístico por región: mejorar tiempos fuera del Sudeste
  4. Estrategia por categoría: negociar volumen y facturación por separado

**Speech (~40 seg):**
> Para cerrar, cuatro oportunidades concretas. Primera: usar el modelo como sistema de alerta temprana y contactar proactivamente los pedidos de riesgo. Segunda: armar un programa de acompañamiento para los 461 vendedores rezagados. Tercera: priorizar la mejora logística en las regiones fuera del Sudeste, donde la brecha es más grande. Y cuarta: tratar volumen y facturación como palancas distintas al negociar con las categorías de producto.

---

## SLIDE 21 — Cierre + preguntas
**En el slide:**
- Título: "Gracias"
- Subtítulo: "¿Preguntas?"
- Abajo: nombres de los 4 integrantes
- Opcional: link al dashboard de Streamlit

**Speech (Presentador D, ~20 seg):**
> Eso es todo. El análisis completo es reproducible ejecutando los notebooks del 00 al 11, y el dashboard está disponible para probar las predicciones en tiempo real. Quedamos a disposición para preguntas. Gracias.

---

## Notas de producción

- **Tiempo total de speech:** ~18 minutos (quedan 2 min para preguntas rápidas o un pequeño buffer)
- **Figuras que se usan:** nb06_02, nb06_04, nb06_06, nb09_02, nb11_01, nb11_03, nb11_05, nb11_06, nb11_07, nb11_08 (10 figuras)
- **Diseño sugerido:** fondo blanco, tipografía Arial/Calibri, acentos en azul oscuro (#1F3A5F), sin animaciones excesivas, una idea por slide
- **Recomendación de ensayo:** cada presentador practica su bloque 2-3 veces cronometrando; la transición entre presentadores es el momento donde más se pierde tiempo
