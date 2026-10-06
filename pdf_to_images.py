"""Resolve qualquer arquivo de entrada (imagem ou PDF) para uma lista de
caminhos de imagem PNG prontos pra usar. A maior parte do dataset real já
vem como imagem (PNG/JPEG) — nesse caso não há nada a renderizar, só
devolve o próprio caminho. PDF (raro no dataset) é renderizado via
PyMuPDF, sem depender de binário externo como poppler."""
import os

import fitz  # PyMuPDF

from config import IMAGE_EXTENSIONS, RENDER_DPI


def render_pdf_to_images(pdf_path: str, out_dir: str, dpi: int = RENDER_DPI) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)

    out_paths = []
    doc = fitz.open(pdf_path)
    try:
        for i, page in enumerate(doc):
            out_path = os.path.join(out_dir, f"{base}_p{i}.png")
            if not os.path.exists(out_path):
                pix = page.get_pixmap(matrix=mat)
                pix.save(out_path)
            out_paths.append(out_path)
    finally:
        doc.close()
    return out_paths


def get_page_images(path: str, out_dir: str, dpi: int = RENDER_DPI) -> list[str]:
    """Ponto de entrada único: devolve uma lista de caminhos de imagem PNG
    prontos pra abrir com PIL, não importa se a entrada era PDF ou imagem."""
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXTENSIONS:
        return [path]
    return render_pdf_to_images(path, out_dir, dpi)
