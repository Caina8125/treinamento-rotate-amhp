"""Utilitários de imagem: rotação (seguindo a convenção de label do config.py)
e redimensionamento com letterbox (evita distorcer a página ao forçar quadrado)."""
from PIL import Image, ImageOps

from config import IMG_SIZE


def apply_label_rotation(img: Image.Image, label_degrees: int) -> Image.Image:
    """Gera uma imagem cujo 'erro de rotação' é `label_degrees` (sentido horário).

    PIL's rotate() gira em sentido ANTI-horário para ângulos positivos, então
    girar +label aqui produz uma imagem que precisa de `label_degrees` no
    sentido horário para voltar a ficar correta — exatamente a convenção
    usada em todo o resto deste módulo.
    """
    if label_degrees == 0:
        return img.copy()
    return img.rotate(label_degrees, expand=True)


def undo_label_rotation(img: Image.Image, label_degrees: int) -> Image.Image:
    """Inverso de apply_label_rotation: recebe uma imagem cujo rótulo VERDADEIRO
    já é conhecido (ex: rotulada manualmente) e devolve a versão "em pé" dela.

    Como apply_label_rotation(upright, L) == upright.rotate(L), desfazer é só
    rotate(-L) — é exatamente essa simetria que permite, a partir de UM rótulo
    manual por documento, reconstruir a versão correta e então regenerar as
    4 rotações sintéticas (não precisa rotular as 4 na mão)."""
    if label_degrees == 0:
        return img.copy()
    return img.rotate(-label_degrees, expand=True)


def letterbox(img: Image.Image, size: int = IMG_SIZE) -> Image.Image:
    """Redimensiona mantendo proporção e preenche com branco até virar um
    quadrado size x size — preserva a orientação do texto sem distorcer."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    offset = ((size - img.width) // 2, (size - img.height) // 2)
    canvas.paste(img, offset)
    return canvas
