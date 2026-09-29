#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Utilidad: convierte los scripts .py de experimento a notebooks .ipynb.
#
# Respeta la estructura narrativa de los scripts: los bloques de comentarios "de sección"
# (líneas consecutivas que empiezan con '#', usadas como texto explicativo) se convierten
# en celdas markdown, y el código entre medio en celdas de código. Así los notebooks
# quedan legibles, en línea con los notebooks del proyecto (notebooks/).
#
# Se corre con: python convertir_a_notebooks.py
#

import re
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

# Scripts a convertir -> notebook destino
SCRIPTS = {
    "experimento_smote_binario.py": "notebooks/exp_01_smote_binario.ipynb",
    "experimento_features_compra.py": "notebooks/exp_02_features_compra.ipynb",
    "entrenar_modelo_final.py": "notebooks/exp_03_entrenar_modelo_final.ipynb",
}

# El shebang / encoding / módulo docstring inicial no aportan en un notebook
SALTAR_PREFIJOS = ("#!/usr/bin/env", "# -*- coding")


def _desescapar_comentario(linea):
    """Quita el '# ' inicial de una línea de comentario."""
    if linea.startswith("# "):
        return linea[2:]
    if linea == "#":
        return ""
    if linea.startswith("#"):
        return linea[1:]
    return linea


def es_comentario(linea):
    return linea.lstrip().startswith("#")


# Preámbulo: como los notebooks viven en notebooks/ pero los scripts usan rutas
# relativas a la raíz del proyecto (data/raw, modelos_output_experimento/...), la
# primera celda ubica el working dir en la raíz si hace falta.
PREAMBULO_MD = (
    "# " "{titulo}\n\n"
    "Notebook generado automáticamente a partir de `{origen}` "
    "(ver `convertir_a_notebooks.py`). Ejecutable de punta a punta.\n\n"
    "La celda siguiente ubica el directorio de trabajo en la raíz del proyecto, "
    "porque los scripts usan rutas relativas a esa raíz (`data/raw`, "
    "`modelos_output_experimento/`)."
)
PREAMBULO_CODE = (
    "import os\n"
    "from pathlib import Path\n"
    "# ubicar el cwd en la raíz del proyecto (donde está data/), venga de donde venga\n"
    "if not Path('data').exists() and Path('..') .joinpath('data').exists():\n"
    "    os.chdir('..')\n"
    "print('Directorio de trabajo:', Path.cwd())"
)


def convertir(ruta_py: Path, ruta_ipynb: Path):
    lineas = ruta_py.read_text(encoding="utf-8").splitlines()
    # descartar shebang y encoding del comienzo
    while lineas and lineas[0].startswith(SALTAR_PREFIJOS):
        lineas.pop(0)

    titulo = ruta_py.stem.replace("_", " ").capitalize()
    celdas = [
        new_markdown_cell(PREAMBULO_MD.format(titulo=titulo, origen=ruta_py.name)),
        new_code_cell(PREAMBULO_CODE),
    ]
    buffer = []
    modo = None  # "md" o "code"

    def volcar():
        nonlocal buffer, modo
        if not buffer:
            return
        contenido = "\n".join(buffer).strip("\n")
        if modo == "md":
            texto = "\n".join(_desescapar_comentario(l) for l in buffer).strip("\n")
            if texto.strip():
                celdas.append(new_markdown_cell(texto))
        else:
            if contenido.strip():
                celdas.append(new_code_cell(contenido))
        buffer = []

    for linea in lineas:
        linea_modo = "md" if es_comentario(linea) or linea.strip() == "" else "code"
        # una línea en blanco no cambia el modo por sí sola: se pega al bloque actual
        if linea.strip() == "" and modo is not None:
            buffer.append(linea)
            continue
        linea_modo = "md" if es_comentario(linea) else "code"
        if modo is None:
            modo = linea_modo
        if linea_modo != modo:
            volcar()
            modo = linea_modo
        buffer.append(linea)
    volcar()

    nb = new_notebook(cells=celdas)
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    ruta_ipynb.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, str(ruta_ipynb))
    n_md = sum(1 for c in celdas if c.cell_type == "markdown")
    n_code = sum(1 for c in celdas if c.cell_type == "code")
    print(f"{ruta_py.name} -> {ruta_ipynb}  ({n_md} md + {n_code} code)")


if __name__ == "__main__":
    for py, ipynb in SCRIPTS.items():
        convertir(Path(py), Path(ipynb))
    print("Conversión completa.")
