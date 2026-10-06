"""
Genera Informe_Olist.docx — informe formal (~25 páginas) con python-docx.
Carátula institucional, índice automático, prosa en 3.ª persona plural,
gráficos reales incrustados, tablas, y secciones completas.

Ejecutar: python generar_informe.py
Requiere: python-docx 1.x, datos en modelos_output/, figuras en figuras_eda/ y figuras_informe/.
"""
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_ORIENT
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml
import pandas as pd

# ──────────────────────────────────────────────
# Rutas
# ──────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
FIGS_EDA = BASE / "figuras_eda"
FIGS_INF = BASE / "figuras_informe"
MODELOS = BASE / "modelos_output"
SALIDA = BASE / "Informe_Olist.docx"

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
AZUL_OSCURO = RGBColor(0x1F, 0x3A, 0x5F)
GRIS_TEXTO = RGBColor(0x2B, 0x2B, 0x2B)
BLANCO = RGBColor(0xFF, 0xFF, 0xFF)


def set_cell_shading(cell, hex_color):
    """Aplica color de fondo a una celda de tabla."""
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}" w:val="clear"/>')
    cell._tc.get_or_add_tcPr().append(shading)


def add_styled_table(doc, df, header_color="1F3A5F", col_widths=None):
    """Agrega una tabla formateada al documento."""
    table = doc.add_table(rows=1 + len(df), cols=len(df.columns))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # encabezados
    for j, col_name in enumerate(df.columns):
        cell = table.rows[0].cells[j]
        cell.text = str(col_name)
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.bold = True
                run.font.color.rgb = BLANCO
                run.font.size = Pt(9)
        set_cell_shading(cell, header_color)

    # datos
    for i, (_, row) in enumerate(df.iterrows()):
        for j, val in enumerate(row):
            cell = table.rows[i + 1].cells[j]
            cell.text = str(val)
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    run.font.size = Pt(9)
                    run.font.color.rgb = GRIS_TEXTO
            if i % 2 == 1:
                set_cell_shading(cell, "F0F2F5")

    # anchos de columna si se proveen
    if col_widths:
        for i, w in enumerate(col_widths):
            for row in table.rows:
                row.cells[i].width = Cm(w)
    return table


def add_figure(doc, path, caption, width=Inches(5.8)):
    """Inserta una imagen centrada con epígrafe debajo."""
    if not path.exists():
        p = doc.add_paragraph(f"[Figura no encontrada: {path.name}]")
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(str(path), width=width)
    # epígrafe
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(14)
    r = cap.add_run(caption)
    r.italic = True
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)


def p_normal(doc, texto):
    """Párrafo normal con estilo consistente."""
    p = doc.add_paragraph(texto, style="Normal")
    return p


def p_empty(doc):
    doc.add_paragraph("", style="Normal")


# ──────────────────────────────────────────────
# Configuración de estilos del documento
# ──────────────────────────────────────────────
def configurar_estilos(doc):
    style = doc.styles["Normal"]
    font = style.font
    font.name = "Arial"
    font.size = Pt(11)
    font.color.rgb = GRIS_TEXTO
    pf = style.paragraph_format
    pf.space_after = Pt(8)
    pf.line_spacing = 1.15

    for level, size, color in [
        ("Heading 1", 16, AZUL_OSCURO),
        ("Heading 2", 13, AZUL_OSCURO),
        ("Heading 3", 11.5, AZUL_OSCURO),
    ]:
        h = doc.styles[level]
        h.font.name = "Arial"
        h.font.size = Pt(size)
        h.font.bold = True
        h.font.color.rgb = color
        h.paragraph_format.space_before = Pt(18 if level == "Heading 1" else 12)
        h.paragraph_format.space_after = Pt(6)

    # TOC
    toc_style = doc.styles.add_style("TOC Heading Custom", 1)  # paragraph
    toc_style.font.name = "Arial"
    toc_style.font.size = Pt(16)
    toc_style.font.bold = True
    toc_style.font.color.rgb = AZUL_OSCURO


# ──────────────────────────────────────────────
# Carátula
# ──────────────────────────────────────────────
def agregar_caratula(doc):
    for _ in range(6):
        doc.add_paragraph("")

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Tecnicatura Superior en Ciencia de Datos\ne Inteligencia Artificial")
    r.font.size = Pt(14)
    r.font.color.rgb = AZUL_OSCURO
    r.font.name = "Arial"

    p_empty(doc)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Proyecto Integrador")
    r.font.size = Pt(13)
    r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    r.font.name = "Arial"

    p_empty(doc)
    p_empty(doc)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Olist E-Commerce")
    r.font.size = Pt(26)
    r.font.bold = True
    r.font.color.rgb = AZUL_OSCURO
    r.font.name = "Arial"

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(
        "Análisis exploratorio, modelado predictivo\n"
        "y segmentación del marketplace brasileño"
    )
    r.font.size = Pt(14)
    r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    r.font.name = "Arial"

    for _ in range(6):
        doc.add_paragraph("")

    # datos de la autora y dataset
    datos = [
        ("Autora:", "Natalia"),
        ("Dataset:", "Brazilian E-Commerce Public Dataset by Olist (Kaggle)"),
        ("Período:", "Septiembre 2016 – Septiembre 2018"),
        ("Fecha:", "Octubre 2026"),
    ]
    for etiqueta, valor in datos:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r1 = p.add_run(etiqueta + " ")
        r1.bold = True
        r1.font.size = Pt(11)
        r1.font.color.rgb = AZUL_OSCURO
        r1.font.name = "Arial"
        r2 = p.add_run(valor)
        r2.font.size = Pt(11)
        r2.font.color.rgb = GRIS_TEXTO
        r2.font.name = "Arial"

    doc.add_page_break()


# ──────────────────────────────────────────────
# Índice (TOC placeholder — se actualiza con Word)
# ──────────────────────────────────────────────
def agregar_indice(doc):
    p = doc.add_paragraph("Índice", style="TOC Heading Custom")
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT

    # Campo TOC de Word — se actualiza al abrir el documento
    p2 = doc.add_paragraph()
    fld_begin = parse_xml(
        f'<w:r {nsdecls("w")}>'
        '  <w:fldChar w:fldCharType="begin"/>'
        '</w:r>'
    )
    fld_code = parse_xml(
        f'<w:r {nsdecls("w")}>'
        '  <w:instrText xml:space="preserve"> TOC \\o "1-3" \\h \\z \\u </w:instrText>'
        '</w:r>'
    )
    fld_sep = parse_xml(
        f'<w:r {nsdecls("w")}>'
        '  <w:fldChar w:fldCharType="separate"/>'
        '</w:r>'
    )
    fld_text = parse_xml(
        f'<w:r {nsdecls("w")}>'
        '  <w:t>[Actualizar este campo para ver el índice — clic derecho → Actualizar campo]</w:t>'
        '</w:r>'
    )
    fld_end = parse_xml(
        f'<w:r {nsdecls("w")}>'
        '  <w:fldChar w:fldCharType="end"/>'
        '</w:r>'
    )
    for elem in [fld_begin, fld_code, fld_sep, fld_text, fld_end]:
        p2._p.append(elem)

    doc.add_page_break()


# ──────────────────────────────────────────────
# Secciones del informe
# ──────────────────────────────────────────────

def sec_introduccion(doc):
    doc.add_heading("1. Introducción", level=1)

    doc.add_heading("1.1 Contexto y problema de negocio", level=2)
    p_normal(doc,
        "Olist es un marketplace brasileño que conecta a miles de pequeños y medianos comercios "
        "con compradores de todo el país, coordinando la publicación, la venta y el envío de los "
        "productos a través de una plataforma integrada. A diferencia de los modelos de comercio "
        "electrónico tradicional, en los que un único vendedor gestiona su propia logística, "
        "en Olist coexisten miles de vendedores con capacidades operativas muy dispares, lo que "
        "genera una fuerte heterogeneidad en la experiencia de compra."
    )
    p_normal(doc,
        "En este escenario, la satisfacción del cliente —medida a través de las reseñas que deja "
        "después de recibir el producto— se convierte en la métrica central del negocio: un "
        "marketplace cuya reputación cae pierde compradores y, con ellos, vendedores. Comprender "
        "qué factores explican las reseñas negativas y dónde se concentran los problemas es, por "
        "lo tanto, una necesidad estratégica."
    )

    doc.add_heading("1.2 Objetivos del análisis", level=2)
    p_normal(doc,
        "El presente trabajo se propone tres objetivos principales. En primer lugar, explorar y "
        "describir el comportamiento del marketplace a partir de los datos públicos de Olist, con "
        "foco en la relación entre logística, geografía y satisfacción del cliente. En segundo "
        "lugar, contrastar formalmente cuatro hipótesis de negocio mediante tests estadísticos no "
        "paramétricos. En tercer lugar, construir dos modelos analíticos: un clasificador que "
        "anticipe si una compra terminará en reseña negativa usando solamente información "
        "disponible al momento de la compra, y una segmentación de vendedores que permita "
        "identificar grupos de riesgo prioritario."
    )
    p_normal(doc,
        "Adicionalmente, se desarrolló un dashboard interactivo en Streamlit que presenta los "
        "resultados del análisis y permite predecir el riesgo de una compra nueva sin reentrenar "
        "el modelo."
    )

    doc.add_heading("1.3 Alcance y fuente de datos", level=2)
    p_normal(doc,
        "El análisis se basa en el dataset público «Brazilian E-Commerce Public Dataset by Olist», "
        "disponible en Kaggle. El dataset contiene información de aproximadamente 100.000 pedidos "
        "realizados entre septiembre de 2016 y septiembre de 2018, distribuida en nueve tablas "
        "relacionales que cubren pedidos, ítems, pagos, reseñas, clientes, productos, vendedores, "
        "geolocalización y traducción de categorías."
    )
    p_normal(doc,
        "Todos los datos son anónimos; los identificadores de clientes y vendedores están "
        "encriptados y las coordenadas geográficas se refieren al centroide del código postal, "
        "no a direcciones reales."
    )


def sec_datos_metodologia(doc):
    doc.add_heading("2. Datos y metodología", level=1)

    p_normal(doc,
        "El proyecto se organiza como un pipeline reproducible de notebooks Jupyter numerados "
        "del 00 al 11. Cada notebook consume el artefacto que generó el anterior y documenta, "
        "antes de cada bloque de código, qué se hace y para qué. A continuación se describe cada "
        "etapa del pipeline."
    )

    doc.add_heading("2.1 Unificación e integración de las nueve tablas", level=2)
    p_normal(doc,
        "Las nueve tablas originales se integran en un único dataset con granularidad de ítem de "
        "pedido (cada fila representa un producto dentro de un pedido), mediante una cadena de "
        "left joins sobre la tabla de ítems. Para evitar la duplicación de filas, las tablas de "
        "pagos y reseñas se agregan previamente a nivel de pedido: los pagos se resumen en el "
        "monto total (suma), las cuotas máximas y el medio de pago principal (moda); las reseñas "
        "se reducen a la última reseña por pedido, conservando el puntaje y el comentario."
    )
    p_normal(doc,
        "La geolocalización se incorpora promediando las coordenadas por código postal y uniendo "
        "el resultado dos veces: una como latitud y longitud del cliente y otra como latitud y "
        "longitud del vendedor. El dataset unificado resulta en 112.650 filas y 39 columnas, con "
        "la combinación order_id + order_item_id como clave única."
    )

    doc.add_heading("2.2 Limpieza", level=2)
    p_normal(doc,
        "El proceso de limpieza sigue un principio conservador: no se inventan ni se borran datos. "
        "Se verificó la ausencia de duplicados exactos (ninguno encontrado). Se convirtieron las "
        "seis columnas de fecha a formato datetime. Se reemplazaron las categorías de producto "
        "faltantes (1.627 registros) por la etiqueta «sin categoría». Se creó la variable "
        "tiene_comentario para marcar los ítems cuya reseña incluye texto escrito (el 42,1 % del "
        "total). Se anularon las coordenadas geográficas fuera del rango válido de Brasil (9 del "
        "lado del cliente), conservando las filas afectadas. Se calculó la cantidad de ítems por "
        "pedido y se eliminaron siete columnas redundantes o ya consolidadas."
    )
    p_normal(doc,
        "El dataset limpio resultante contiene 112.650 filas y 34 columnas, y se guarda como "
        "artefacto intermedio del pipeline."
    )

    doc.add_heading("2.3 Ingeniería de variables", level=2)
    p_normal(doc,
        "Sobre el dataset limpio se construyen nueve variables nuevas, diseñadas para habilitar "
        "tanto el análisis exploratorio como el modelado:"
    )

    # tabla de variables
    vars_data = pd.DataFrame({
        "Variable": [
            "tiempo_entrega_dias", "dias_vs_estimado", "envio_demorado",
            "flete_ratio", "reseña_positiva", "region_sudeste",
            "mes_compra", "distancia_km", "volumen_cm3",
        ],
        "Definición": [
            "Días entre la compra y la entrega real",
            "Días entre la entrega real y la fecha estimada (positivo = tarde)",
            "1 si la entrega fue posterior a la estimada, 0 si fue a tiempo, NaN si no fue entregado",
            "Costo del flete / precio del producto",
            "1 si el puntaje es 4-5 ★, 0 si es 1-2 ★; el puntaje 3 (neutral) se excluye",
            "«Sudeste» si el cliente está en SP, RJ, MG o ES; «Resto del país» en caso contrario",
            "Mes de la compra, normalizado al primer día",
            "Distancia geodésica cliente-vendedor, calculada con la fórmula de Haversine (R = 6.371 km)",
            "Largo × alto × ancho del producto (cm³)",
        ],
    })
    add_styled_table(doc, vars_data, col_widths=[4.5, 12.5])

    p_normal(doc,
        "El dataset resultante contiene 112.650 filas y 43 columnas. Los valores nulos se "
        "mantienen deliberadamente en los casos en que la información no existe (pedidos no "
        "entregados, coordenadas ausentes, reseñas neutrales): imputarlos distorsionaría las "
        "métricas de negocio."
    )

    doc.add_heading("2.4 Herramientas y entorno", level=2)
    p_normal(doc,
        "Todo el análisis se realizó en Python 3 con las bibliotecas pandas, NumPy, Matplotlib, "
        "Seaborn, Plotly, SciPy, scikit-learn y LightGBM. El dashboard interactivo se desarrolló "
        "en Streamlit. El modelo final se serializa con joblib para su consumo sin reentrenamiento."
    )


def sec_analisis_exploratorio(doc):
    doc.add_heading("3. Análisis exploratorio", level=1)

    p_normal(doc,
        "El análisis exploratorio recorre cuatro ejes: la evolución y composición del negocio, "
        "la logística y la satisfacción, la dimensión geográfica y el contenido textual de las "
        "reseñas."
    )

    # 3.1 Negocio
    doc.add_heading("3.1 Evolución y composición del negocio", level=2)
    p_normal(doc,
        "El volumen de pedidos creció de manera sostenida desde finales de 2016 hasta un pico de "
        "aproximadamente 7.450 pedidos mensuales en noviembre de 2017, coincidiendo con el Black "
        "Friday. A partir de enero de 2018 la tendencia se estabiliza en torno a los 6.500 pedidos "
        "mensuales, lo que sugiere una meseta de madurez dentro del período observado."
    )
    add_figure(doc, FIGS_EDA / "nb06_02.png",
               "Figura 1. Cantidad de pedidos únicos por mes (sept. 2016 – sept. 2018).")

    p_normal(doc,
        "La distribución de las reseñas está fuertemente sesgada hacia la calificación máxima: "
        "el 56 % de los ítems calificados recibe 5 estrellas y el 75 % recibe 4 o 5 estrellas. "
        "Sin embargo, las reseñas negativas (1-2 estrellas) representan el 16 % del total, una "
        "proporción no despreciable que justifica su análisis."
    )
    add_figure(doc, FIGS_EDA / "nb06_04.png",
               "Figura 2. Distribución de las calificaciones (review_score).", width=Inches(4.0))

    p_normal(doc,
        "La categoría con mayor volumen de ítems vendidos es «cama, baño y mesa» (11.115 ítems), "
        "seguida por «salud y belleza» y «deportes y ocio». Sin embargo, las categorías líderes "
        "en volumen no coinciden necesariamente con las líderes en facturación: las cinco "
        "categorías de mayor facturación concentran el 39,7 % del total, lo que indica una "
        "distribución razonablemente diversificada."
    )
    add_figure(doc, FIGS_EDA / "nb06_03.png",
               "Figura 3. Las diez categorías con mayor volumen de ítems vendidos.")
    add_figure(doc, FIGS_EDA / "nb07_07.png",
               "Figura 4. Diagrama de Pareto de la facturación por categoría (top 15).")

    p_normal(doc,
        "En cuanto a los medios de pago, la tarjeta de crédito domina con el 76 % de los pedidos, "
        "seguida por el boleto bancario con el 20 %. La mediana de cuotas elegidas es de 1, pero "
        "la distribución tiene una cola larga hacia los pagos en 10 y 12 cuotas, lo que sugiere "
        "que una fracción de las compras financia productos de ticket alto."
    )
    add_figure(doc, FIGS_EDA / "nb06_08.png",
               "Figura 5. Medio de pago principal y distribución de cuotas elegidas.")

    # 3.2 Logística y satisfacción
    doc.add_heading("3.2 Logística y satisfacción", level=2)
    p_normal(doc,
        "El tiempo de entrega promedio es de 12 días, con una mediana de 10 días. La distribución "
        "presenta una cola derecha pronunciada: aunque la gran mayoría de los pedidos se entrega "
        "en menos de 20 días, existen casos extremos que superan los 60 días. La variable "
        "dias_vs_estimado muestra que, en promedio, Olist entrega 12 días antes de la fecha "
        "prometida; las estimaciones son deliberadamente conservadoras, lo que amortigua el "
        "impacto de los atrasos sobre la percepción del cliente."
    )
    add_figure(doc, FIGS_EDA / "nb06_05.png",
               "Figura 6. Distribución del tiempo de entrega y de la desviación respecto de la fecha estimada.")

    p_normal(doc,
        "A pesar de ese colchón, el 7,9 % de los pedidos entregados llega después de la fecha "
        "estimada. La consecuencia sobre la satisfacción es drástica: los pedidos entregados a "
        "tiempo tienen una calificación promedio de 4,21 estrellas (mediana 5), mientras que los "
        "demorados caen a 2,55 estrellas (mediana 2). El boxplot de la Figura 7 ilustra la "
        "magnitud de esta brecha."
    )
    add_figure(doc, FIGS_EDA / "nb06_06.png",
               "Figura 7. Calificación según cumplimiento del plazo de entrega.", width=Inches(4.2))

    p_normal(doc,
        "Las categorías con mayor proporción de envíos demorados (entre aquellas con al menos "
        "30 pedidos) son «muebles, colchones y tapicería» (13,5 %), «audio» (12,7 %) y «moda "
        "playa» (12,6 %). Estas categorías no son las de mayor volumen, lo que sugiere problemas "
        "logísticos específicos más que un efecto de escala."
    )
    add_figure(doc, FIGS_EDA / "nb07_01.png",
               "Figura 8. Las diez categorías con mayor porcentaje de envíos demorados (≥ 30 pedidos).")

    # 3.3 Correlaciones y factores físicos
    doc.add_heading("3.3 Relaciones entre variables numéricas", level=2)
    p_normal(doc,
        "La matriz de correlación de Pearson entre las variables numéricas clave revela varios "
        "patrones relevantes. El tiempo de entrega se correlaciona fuertemente con la desviación "
        "respecto de la fecha estimada (r = 0,60) y con la distancia cliente-vendedor (r = 0,39). "
        "El peso y el volumen del producto están altamente correlacionados entre sí (r = 0,80), "
        "y ambos se asocian al costo de flete (r = 0,59 y r = 0,61 respectivamente), pero su "
        "relación con el tiempo de entrega es débil (r < 0,10). La calificación presenta una "
        "correlación negativa moderada con el tiempo de entrega (r = −0,30) y con la desviación "
        "respecto de la estimación (r = −0,23)."
    )
    add_figure(doc, FIGS_EDA / "nb07_03.png",
               "Figura 9. Matriz de correlación de Pearson entre las variables numéricas principales.")

    # 3.4 Geografía
    doc.add_heading("3.4 Dimensión geográfica", level=2)
    p_normal(doc,
        "Los clientes se concentran en el eje Sudeste-Sur del país, con São Paulo, Río de Janeiro "
        "y Minas Gerais como los tres estados dominantes. La distancia mediana entre cliente y "
        "vendedor es de 432 km, consistente con las dimensiones continentales de Brasil y con la "
        "concentración de vendedores en el Sudeste."
    )
    p_normal(doc,
        "La comparación entre las dos regiones (Sudeste vs. resto del país) muestra una brecha "
        "logística significativa: los clientes fuera del Sudeste esperan en promedio 15,9 días "
        "por su entrega frente a 10,2 días en el Sudeste, pagan un flete promedio de R$ 25,7 "
        "frente a R$ 17,4, y sufren una tasa de demora del 9 % frente al 7 %. Sin embargo, el "
        "gasto promedio por ítem es similar o incluso superior fuera del Sudeste (R$ 133,5 vs. "
        "R$ 114,8): la periferia geográfica paga más flete y espera más por un ticket comparable."
    )

    # tabla estados
    estados_data = pd.DataFrame({
        "Estado": ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "DF", "ES", "GO"],
        "Pedidos": ["41.375", "12.762", "11.544", "5.432", "4.998",
                     "3.612", "3.358", "2.125", "2.025", "2.007"],
        "Reseña prom.": ["4,13", "3,81", "4,09", "4,05", "4,11",
                          "4,00", "3,82", "4,00", "3,99", "3,99"],
        "% demorado": ["6 %", "13 %", "5 %", "7 %", "5 %",
                        "10 %", "14 %", "7 %", "12 %", "8 %"],
    })
    doc.add_paragraph("")
    add_styled_table(doc, estados_data, col_widths=[2.5, 3, 3, 3])
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(14)
    r = cap.add_run("Tabla 1. Los diez estados con mayor volumen de pedidos.")
    r.italic = True
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    p_normal(doc,
        "Cabe señalar que la brecha no es uniforme dentro del Sudeste: Río de Janeiro presenta "
        "una tasa de demora del 13 % y Espírito Santo del 12 %, ambas superiores al promedio "
        "general, lo que indica que la partición Sudeste/resto es una simplificación útil pero no "
        "perfecta."
    )

    # 3.5 NLP
    doc.add_heading("3.5 Análisis textual de las reseñas", level=2)
    p_normal(doc,
        "El 42,1 % de los ítems calificados incluye un comentario escrito en portugués. Se "
        "procesaron los textos eliminando acentos, caracteres no alfabéticos y stopwords del "
        "portugués (207 términos de la lista de NLTK), y se construyeron nubes de palabras "
        "separadas para las reseñas negativas (puntaje 1-2) y positivas (puntaje 4-5)."
    )
    p_normal(doc,
        "El vocabulario de las reseñas negativas está dominado por términos asociados al "
        "proceso de entrega: «não» (no), «recebi» (recibí), «entregue» (entregado), «prazo» "
        "(plazo), «chegou» (llegó), «entrega» y «pedido». La queja no es sobre el producto "
        "en sí, sino sobre la experiencia logística. En contraste, las reseñas positivas se "
        "caracterizan por palabras de satisfacción: «excelente», «qualidade» (calidad), «bom» "
        "(bueno), «recomendo» (recomiendo), «perfeito» (perfecto), «gostei» (me gustó), junto "
        "con referencias a una entrega exitosa: «chegou», «prazo», «rápido», «dentro» (dentro "
        "del plazo). Esta evidencia cualitativa refuerza la hipótesis de que la logística es el "
        "principal driver de la satisfacción."
    )
    add_figure(doc, FIGS_EDA / "nb09_02.png",
               "Figura 10. Nubes de palabras de las reseñas negativas (izquierda) y positivas (derecha).",
               width=Inches(6.0))


def sec_hipotesis(doc):
    doc.add_heading("4. Contraste de hipótesis", level=1)

    p_normal(doc,
        "Las observaciones del análisis exploratorio se sometieron a tests estadísticos formales "
        "para separar los hallazgos significativos del ruido. Se eligieron tests no paramétricos "
        "porque las variables involucradas (calificaciones ordinales, tiempos con distribución "
        "sesgada) no cumplen los supuestos de normalidad y homocedasticidad que requieren los "
        "tests clásicos. Todas las pruebas se realizaron con un nivel de significancia α = 0,05 "
        "y alternativas de una cola, en la dirección indicada por la hipótesis."
    )

    doc.add_heading("4.1 H1 — La demora reduce la satisfacción", level=2)
    p_normal(doc,
        "Se comparó la distribución de calificaciones entre los pedidos entregados a tiempo "
        "(n = 100.849, media = 4,21) y los demorados (n = 8.520, media = 2,55) mediante la "
        "prueba de Mann-Whitney U con alternativa «a tiempo > demorado». El estadístico U "
        "resultó 657.453.250 con un p-valor prácticamente igual a cero. La hipótesis se confirma: "
        "la diferencia de satisfacción entre ambos grupos es estadísticamente significativa y de "
        "gran magnitud."
    )

    doc.add_heading("4.2 H2 — Las reseñas negativas hablan de demoras", level=2)
    p_normal(doc,
        "Esta hipótesis se evaluó de forma cualitativa a partir de la evidencia producida en el "
        "análisis textual (sección 3.5). El ranking de los quince términos más frecuentes en las "
        "reseñas negativas muestra que los primeros lugares están ocupados por palabras directamente "
        "relacionadas con el proceso de entrega y la falta de cumplimiento del plazo. La hipótesis "
        "se considera evidenciada."
    )

    doc.add_heading("4.3 H3 — Fuera del Sudeste, más tiempo y más flete", level=2)
    p_normal(doc,
        "Se comparó el tiempo de entrega entre los clientes del Sudeste (n = 75.731, media = 10,2 "
        "días) y los del resto del país (n = 34.465, media = 15,9 días) mediante Mann-Whitney U "
        "con alternativa «Sudeste < resto». El p-valor resultó prácticamente cero. Adicionalmente, "
        "el flete promedio en el Sudeste es de R$ 17,4 frente a R$ 25,7 en el resto. La hipótesis "
        "se confirma: la ventaja logística del Sudeste es estadísticamente significativa."
    )

    doc.add_heading("4.4 Hipótesis adicional — Mayor distancia implica mayor tiempo", level=2)
    p_normal(doc,
        "Se calculó la correlación de Spearman entre la distancia cliente-vendedor y el tiempo de "
        "entrega, obteniendo un coeficiente ρ = 0,54 con p-valor prácticamente cero. La relación "
        "es positiva, monotónica y estadísticamente significativa: cada kilómetro adicional se "
        "asocia a más días de espera."
    )

    # tabla resumen
    doc.add_heading("4.5 Resumen de resultados", level=2)
    hip_data = pd.DataFrame({
        "Hipótesis": [
            "H1 — La demora reduce la satisfacción",
            "H2 — Las reseñas negativas hablan de demoras",
            "H3 — Fuera del Sudeste: más días y flete",
            "Adicional — Más km ⇒ más días",
        ],
        "Test": ["Mann-Whitney U", "Cualitativo (NLP)", "Mann-Whitney U", "Spearman"],
        "p-valor": ["≈ 0", "—", "≈ 0", "≈ 0"],
        "Veredicto": ["Confirmada", "Evidenciada", "Confirmada", "Confirmada"],
    })
    add_styled_table(doc, hip_data, col_widths=[6, 3.5, 2, 3])
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(14)
    r = cap.add_run("Tabla 2. Resumen del contraste de hipótesis.")
    r.italic = True
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    p_normal(doc,
        "Es importante notar que los tests se ejecutaron a nivel de ítem de pedido, de modo que "
        "los ítems de un mismo pedido multipedido no son estrictamente independientes. Esta "
        "limitación, habitual en datos transaccionales, se discute en la sección 8."
    )


def sec_modelado(doc):
    doc.add_heading("5. Modelado predictivo", level=1)

    # 5.1 Clasificación
    doc.add_heading("5.1 Clasificación de la reseña", level=2)
    doc.add_heading("5.1.1 Diseño del problema", level=3)
    p_normal(doc,
        "El objetivo es anticipar si una compra terminará en reseña positiva (4-5 estrellas) o "
        "negativa (1-2 estrellas) utilizando exclusivamente información disponible al momento de "
        "la compra. Las reseñas neutrales (3 estrellas) se excluyen del target para obtener un "
        "problema de clasificación binaria más nítido. El dataset etiquetado contiene 84.347 "
        "reseñas positivas y 17.986 negativas (proporción 82 %/18 %), lo que configura un "
        "problema desbalanceado."
    )
    p_normal(doc,
        "Se utilizan 117 variables agrupadas en cinco bloques: numéricas base (distancia, "
        "volumen, peso, precio, flete, ratio de flete, ítems por pedido, cuotas), historial del "
        "vendedor (porcentaje de demora, cantidad de pedidos, reseña promedio, categorías), "
        "temporales (mes, día de la semana, temporada alta), geográficas (mismo estado) y "
        "categóricas codificadas mediante one-hot (categoría de producto, región, medio de pago, "
        "estado del cliente). Las variables post-compra —texto de la reseña, calificación, "
        "demora real y fechas de entrega— se excluyen explícitamente para evitar la fuga de "
        "información."
    )

    doc.add_heading("5.1.2 Separación de datos e historial del vendedor", level=3)
    p_normal(doc,
        "El dataset se dividió en entrenamiento (76.749 filas, 75 %) y prueba (25.584 filas, "
        "25 %) mediante un split estratificado por el target. A partir de las filas de "
        "entrenamiento se calcularon cuatro variables de historial del vendedor: porcentaje de "
        "demora, cantidad de pedidos, reseña promedio y cantidad de categorías. Los vendedores "
        "que solo aparecen en el conjunto de prueba reciben el promedio global del entrenamiento. "
        "Este diseño garantiza que el historial no filtre información del propio pedido que se "
        "quiere predecir."
    )

    doc.add_heading("5.1.3 Entrenamiento y evaluación", level=3)
    p_normal(doc,
        "Se entrenó un modelo LightGBM con 300 árboles, learning rate de 0,05, 31 hojas por "
        "árbol y class_weight = balanced para compensar el desbalance de clases. La evaluación "
        "se realizó sobre el conjunto de prueba, priorizando las métricas que pesan por igual "
        "ambas clases."
    )

    # tabla de métricas
    met_data = pd.DataFrame({
        "Métrica": ["Accuracy", "Precision macro", "Recall macro", "F1 macro", "AUC"],
        "Valor": ["0,718", "0,612", "0,666", "0,618", "0,718"],
    })
    add_styled_table(doc, met_data, col_widths=[5, 3])
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(14)
    r = cap.add_run("Tabla 3. Métricas del clasificador sobre el conjunto de prueba.")
    r.italic = True
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    p_normal(doc,
        "El AUC de 0,72 y el F1 macro de 0,62 son modestos pero legítimos: al utilizar "
        "únicamente información previa a la entrega, el modelo no puede capturar el factor que "
        "más influye en la reseña (si el envío llegó tarde), porque ese dato aún no ocurrió. Aun "
        "así, el modelo separa significativamente mejor que el azar y es útil como herramienta de "
        "priorización de riesgo."
    )

    add_figure(doc, FIGS_INF / "nb11_01_curva_roc.png",
               "Figura 11. Curva ROC del clasificador sobre el conjunto de prueba.",
               width=Inches(4.6))

    add_figure(doc, FIGS_INF / "nb11_02_matriz_confusion.png",
               "Figura 12. Matriz de confusión normalizada por clase real.",
               width=Inches(4.2))

    p_normal(doc,
        "La matriz de confusión muestra que el modelo detecta correctamente el 59 % de las "
        "reseñas negativas (recall de la clase negativa), a costa de clasificar como negativo "
        "el 25 % de las reseñas que en realidad son positivas (falsos negativos desde la "
        "perspectiva de la clase positiva). En un contexto de negocio, este trade-off es "
        "aceptable: es preferible alertar sobre algunos pedidos que terminarán bien a dejar pasar "
        "sin aviso los que terminarán mal."
    )

    doc.add_heading("5.1.4 Variables más influyentes", level=3)
    p_normal(doc,
        "La variable con mayor peso en el modelo es la reseña histórica promedio del vendedor: "
        "el mejor predictor de si un cliente quedará conforme es cómo quedaron los clientes "
        "anteriores de ese mismo vendedor. Le siguen la distancia cliente-vendedor, el costo de "
        "flete, el mes de compra y el ratio flete/precio. Las cuatro variables de historial del "
        "vendedor aparecen entre las quince más importantes, lo que confirma que la trayectoria "
        "pasada del vendedor es la señal más informativa disponible al momento de la compra."
    )
    add_figure(doc, FIGS_INF / "nb11_03_importancia_variables.png",
               "Figura 13. Las quince variables más influyentes del modelo (por cantidad de divisiones).")

    doc.add_heading("5.1.5 Valor operativo del modelo", level=3)
    p_normal(doc,
        "La curva de captura muestra que, al priorizar los pedidos por riesgo según el modelo, "
        "se detecta el 31 % de las reseñas negativas revisando solo el 10 % de los pedidos, el "
        "47 % revisando el 20 % y el 57 % revisando el 30 %. Esto significa que un equipo de "
        "atención al cliente que contacte proactivamente al 20 % de mayor riesgo estaría "
        "cubriendo casi la mitad de las experiencias negativas antes de que se conviertan en "
        "reseñas publicadas."
    )
    add_figure(doc, FIGS_INF / "nb11_05_curva_captura.png",
               "Figura 14. Curva de captura: proporción de reseñas negativas detectadas al priorizar por riesgo.")

    # 5.2 Segmentación
    doc.add_heading("5.2 Segmentación de vendedores", level=2)
    doc.add_heading("5.2.1 Diseño y selección del número de segmentos", level=3)
    p_normal(doc,
        "Se agregó el dataset a nivel de vendedor calculando ocho variables de perfil: tiempo de "
        "entrega promedio, porcentaje de demora, distancia promedio al cliente, flete promedio, "
        "cantidad de pedidos, reseña promedio, precio promedio y cantidad de categorías vendidas. "
        "Se aplicó transformación logarítmica a las variables de pedidos y precio para reducir el "
        "efecto de los valores extremos, y se estandarizaron todas las variables con StandardScaler."
    )
    p_normal(doc,
        "Se evaluó K-Means con k de 2 a 8, seleccionando k = 3 por ofrecer el mejor equilibrio "
        "entre interpretabilidad y separación. El puntaje de silueta en torno a 0,18 es habitual "
        "en datos de negocio reales, donde los grupos no forman clusters perfectamente separados."
    )
    add_figure(doc, FIGS_INF / "nb11_06_codo_silhouette.png",
               "Figura 15. Método del codo e índice de silueta para la elección de k.")

    doc.add_heading("5.2.2 Perfil de los tres segmentos", level=3)
    p_normal(doc,
        "Los tres segmentos resultantes se etiquetaron según su perfil de negocio:"
    )

    seg_data = pd.DataFrame({
        "Segmento": ["Ágiles de bajo volumen", "Consolidados de alto volumen", "Rezagados"],
        "Vendedores": ["1.704 (58 %)", "795 (27 %)", "461 (16 %)"],
        "Entrega (días)": ["9,5", "11,8", "19,6"],
        "% demorado": ["4 %", "8 %", "24 %"],
        "Reseña prom.": ["4,25", "4,05", "3,48"],
        "Pedidos prom.": ["7", "107", "7"],
    })
    add_styled_table(doc, seg_data, col_widths=[4.5, 3, 2.5, 2.5, 2.5, 2.5])
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(14)
    r = cap.add_run("Tabla 4. Perfil promedio de los tres segmentos de vendedores.")
    r.italic = True
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    p_normal(doc,
        "El segmento «Ágiles de bajo volumen» agrupa al 58 % de los vendedores: operan con "
        "pocos pedidos pero entregan rápido (9,5 días en promedio), con baja demora y buena "
        "reseña. El segmento «Consolidados de alto volumen» concentra al 27 % de los vendedores "
        "pero al 85 % de los ítems vendidos: manejan un volumen alto con un desempeño logístico "
        "sólido. El segmento «Rezagados» es el más pequeño (16 % de los vendedores, 3 % de los "
        "ítems) pero presenta los peores indicadores: 19,6 días de entrega promedio, 24 % de "
        "demora y una reseña promedio de 3,48."
    )

    add_figure(doc, FIGS_INF / "nb11_07_perfil_segmentos.png",
               "Figura 16. Comparación de los indicadores clave por segmento de vendedores.")

    add_figure(doc, FIGS_INF / "nb11_08_mapa_vendedores.png",
               "Figura 17. Distribución de vendedores según tiempo de entrega y porcentaje de demora.")

    p_normal(doc,
        "El valor del segmento «Rezagados» reside en que constituye una lista de contactos "
        "prioritaria: son pocos vendedores que concentran una proporción desproporcionada del "
        "riesgo de mala experiencia. Una intervención focalizada sobre este grupo tendría el "
        "mayor retorno marginal en términos de satisfacción del cliente."
    )


def sec_conclusiones(doc):
    doc.add_heading("6. Conclusiones y oportunidades de negocio", level=1)

    p_normal(doc,
        "El análisis realizado permite extraer cinco conclusiones principales, cada una de las "
        "cuales se traduce en una oportunidad de negocio concreta."
    )

    doc.add_heading("6.1 La logística es el problema central de la experiencia", level=2)
    p_normal(doc,
        "El cumplimiento del plazo de entrega prometido es la variable más asociada a la "
        "calificación del cliente, por encima del precio, la categoría o el medio de pago. "
        "Los pedidos demorados reciben en promedio 1,7 estrellas menos que los entregados a "
        "tiempo. Las reseñas negativas no hablan del producto, sino de la entrega. Toda mejora "
        "en la cadena logística tiene un impacto directo y medible sobre la satisfacción."
    )

    doc.add_heading("6.2 La brecha geográfica es real y costosa", level=2)
    p_normal(doc,
        "Los clientes fuera del eje Sudeste pagan más flete, esperan más días y sufren más "
        "demoras, pese a gastar un ticket similar. La correlación entre distancia y tiempo de "
        "entrega es positiva y significativa (Spearman ρ = 0,54). Priorizar la mejora de tiempos "
        "y costos en las regiones periféricas reduciría la brecha de satisfacción sin sacrificar "
        "ingresos."
    )

    doc.add_heading("6.3 Se puede anticipar una mala reseña antes de la entrega", level=2)
    p_normal(doc,
        "El modelo de clasificación permite, con información disponible al momento de la compra, "
        "estimar el riesgo de reseña negativa. Revisando el 20 % de los pedidos de mayor riesgo "
        "se cubre casi la mitad de las experiencias negativas. Esto habilita un sistema de alerta "
        "temprana: contactar proactivamente al comprador, ofrecer seguimiento especial o coordinar "
        "con el vendedor antes de que la reseña esté escrita."
    )

    doc.add_heading("6.4 Los vendedores no son homogéneos", level=2)
    p_normal(doc,
        "La segmentación identifica un grupo de vendedores «Rezagados» que, pese a representar "
        "solo el 16 % de la base, concentra los peores indicadores logísticos. Un programa de "
        "acompañamiento focalizado en este segmento —con umbrales de desempeño y capacitación— "
        "tendría el mayor retorno marginal sobre la calidad del servicio."
    )

    doc.add_heading("6.5 Volumen y facturación son ejes distintos", level=2)
    p_normal(doc,
        "Las categorías líderes en cantidad de ítems no coinciden con las líderes en facturación: "
        "las primeras mueven alto volumen a ticket bajo y las segundas mueven menos unidades a "
        "ticket alto. Tratar volumen y facturación como palancas independientes permite negociar "
        "condiciones diferenciadas con cada tipo de categoría."
    )


def sec_trabajo_futuro(doc):
    doc.add_heading("7. Trabajo futuro", level=1)
    p_normal(doc,
        "El análisis presentado abre varias líneas de trabajo que podrían profundizar los "
        "hallazgos y mejorar los modelos."
    )
    p_normal(doc,
        "En primer lugar, el modelo de clasificación podría beneficiarse de un split temporal "
        "en lugar de aleatorio, de modo que el historial del vendedor refleje estrictamente el "
        "pasado y no incluya pedidos futuros. Asimismo, un esquema de target encoding con "
        "out-of-fold evitaría la potencial sobreestimación de la importancia de la reseña "
        "histórica del vendedor."
    )
    p_normal(doc,
        "En segundo lugar, la incorporación de variables externas —como la distancia a centros "
        "logísticos, datos de huelgas del transporte o eventos estacionales locales— podría "
        "mejorar la capacidad predictiva del modelo."
    )
    p_normal(doc,
        "En tercer lugar, un análisis de texto más sofisticado (topic modeling, análisis de "
        "sentimiento con modelos preentrenados en portugués) permitiría cuantificar con mayor "
        "precisión la proporción de reseñas negativas atribuibles a problemas logísticos frente "
        "a otros motivos."
    )
    p_normal(doc,
        "Finalmente, la segmentación de vendedores podría complementarse con un modelo de "
        "supervivencia que estime la probabilidad de que un vendedor «Rezagado» mejore o abandone "
        "la plataforma en los meses siguientes, permitiendo una intervención aún más proactiva."
    )


def sec_limitaciones(doc):
    doc.add_heading("8. Limitaciones", level=1)
    p_normal(doc,
        "El presente análisis tiene limitaciones que es necesario explicitar para una correcta "
        "interpretación de los resultados."
    )
    p_normal(doc,
        "Los datos cubren un período de dos años (2016-2018) y corresponden a un snapshot "
        "histórico; la dinámica del marketplace puede haber cambiado significativamente desde "
        "entonces. Las coordenadas geográficas se refieren al centroide del código postal, no a "
        "la ubicación exacta, lo que introduce un error en el cálculo de distancias que se "
        "atenúa con el uso de Haversine a escala de país."
    )
    p_normal(doc,
        "Los tests estadísticos se ejecutaron a nivel de ítem de pedido, donde los ítems de un "
        "mismo pedido no son independientes. Si bien el gran tamaño muestral (> 100.000) hace "
        "que este efecto sea menor en los p-valores, los tamaños de efecto deben interpretarse "
        "con cautela."
    )
    p_normal(doc,
        "El modelo de clasificación utiliza un split aleatorio en lugar de temporal, lo que "
        "significa que el historial del vendedor puede incluir pedidos cronológicamente "
        "posteriores al que se predice. Además, la imputación de medianas se calculó antes del "
        "split (un sesgo menor, dado que se refiere solo a las variables numéricas base). Estas "
        "decisiones se adoptaron por simplicidad y reproducibilidad, pero constituyen una fuente "
        "de optimismo en las métricas reportadas."
    )
    p_normal(doc,
        "Finalmente, el análisis textual se limitó a conteos de palabras y nubes de palabras; "
        "no se aplicaron técnicas de modelado de tópicos ni clasificadores de sentimiento, lo "
        "que limita el alcance de la hipótesis H2 a una evidencia cualitativa."
    )


def sec_glosario(doc):
    doc.add_heading("9. Glosario", level=1)

    terms = [
        ("AUC (Area Under the Curve)", "Área bajo la curva ROC; mide la capacidad "
         "discriminativa del modelo. Un valor de 0,5 equivale al azar; 1,0 es perfecto."),
        ("Boleto", "Método de pago brasileño equivalente a un recibo bancario impreso o "
         "digital que se paga en efectivo o por transferencia."),
        ("Class weight balanced", "Configuración del algoritmo que penaliza los errores sobre "
         "la clase minoritaria en proporción inversa a su frecuencia."),
        ("F1 macro", "Media armónica de precision y recall, promediada por igual entre las "
         "clases. Métrica adecuada para problemas desbalanceados."),
        ("Fórmula de Haversine", "Cálculo de la distancia geodésica entre dos puntos sobre la "
         "esfera terrestre a partir de sus coordenadas de latitud y longitud."),
        ("K-Means", "Algoritmo de clustering que agrupa observaciones en k clusters minimizando "
         "la inercia (suma de distancias al cuadrado al centroide de cada grupo)."),
        ("LightGBM", "Implementación eficiente de gradient boosting sobre árboles de decisión, "
         "desarrollada por Microsoft."),
        ("Mann-Whitney U", "Test no paramétrico para comparar dos muestras independientes sin "
         "asumir normalidad; compara rangos en lugar de medias."),
        ("One-hot encoding", "Técnica de codificación que transforma una variable categórica en "
         "tantas columnas binarias como categorías tiene."),
        ("Recall", "Proporción de positivos reales que el modelo identifica correctamente."),
        ("Silueta (silhouette)", "Medida de la calidad de un clustering; compara la cohesión "
         "interna de cada grupo con la separación respecto de los demás."),
        ("Spearman ρ", "Coeficiente de correlación no paramétrico que mide la relación "
         "monótona entre dos variables; robusto a no normalidad."),
    ]

    for term, defn in terms:
        p = doc.add_paragraph()
        r = p.add_run(term + ": ")
        r.bold = True
        r.font.size = Pt(10.5)
        r.font.color.rgb = AZUL_OSCURO
        r2 = p.add_run(defn)
        r2.font.size = Pt(10.5)
        r2.font.color.rgb = GRIS_TEXTO


def sec_reproducibilidad(doc):
    doc.add_heading("10. Reproducibilidad", level=1)
    p_normal(doc,
        "Todo el análisis es reproducible ejecutando los notebooks en orden (00 → 11): cada "
        "uno consume el artefacto que genera el anterior. El notebook 11 entrena y serializa el "
        "modelo en la carpeta modelos_output/, de modo que el dashboard de Streamlit (app.py) "
        "solo necesita leer los resultados, sin reentrenar nada. Los archivos requeridos son:"
    )

    repro_data = pd.DataFrame({
        "Archivo": [
            "data/pipeline/01_df_limpio.csv",
            "data/pipeline/02_df_features.csv",
            "modelos_output/modelo_final.joblib",
            "modelos_output/catalogo_vendedores.csv",
            "modelos_output/clustering_resumen.csv",
            "modelos_output/clustering_asignaciones.csv",
        ],
        "Generado por": [
            "Notebook 03", "Notebook 04", "Notebook 11",
            "Notebook 11", "Notebook 11", "Notebook 11",
        ],
        "Descripción": [
            "Dataset limpio (112.650 × 34)",
            "Dataset con variables derivadas (112.650 × 43)",
            "Clasificador LightGBM + artefactos de predicción",
            "Vendedores con historial (2.801 registros)",
            "Perfil promedio por segmento",
            "Segmento asignado a cada vendedor (2.960 registros)",
        ],
    })
    add_styled_table(doc, repro_data, col_widths=[5.5, 2.5, 7])
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(14)
    r = cap.add_run("Tabla 5. Artefactos principales del pipeline.")
    r.italic = True
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    p_normal(doc,
        "El entorno de ejecución requiere Python 3 con las bibliotecas listadas en "
        "requirements.txt. La semilla de aleatoriedad se fija en 42 en todos los notebooks "
        "que la requieren."
    )


# ──────────────────────────────────────────────
# Ensamblado principal
# ──────────────────────────────────────────────
def main():
    doc = Document()

    # márgenes
    for section in doc.sections:
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)

    configurar_estilos(doc)
    agregar_caratula(doc)
    agregar_indice(doc)
    sec_introduccion(doc)
    sec_datos_metodologia(doc)
    sec_analisis_exploratorio(doc)
    sec_hipotesis(doc)
    sec_modelado(doc)
    sec_conclusiones(doc)
    sec_trabajo_futuro(doc)
    sec_limitaciones(doc)
    sec_glosario(doc)
    sec_reproducibilidad(doc)

    doc.save(str(SALIDA))
    print(f"Informe generado: {SALIDA}")
    print(f"  → Abrilo en Word y actualizá el índice (clic derecho → Actualizar campo).")


if __name__ == "__main__":
    main()
