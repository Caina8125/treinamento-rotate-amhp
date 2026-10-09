"""Monta o dataset de treino a partir de rotation_data/labels.csv (rotulado
via label_dataset.py): pra cada documento rotulado, desfaz a rotação real
(chegando na versão "em pé") e regenera as 4 rotações sintéticas a partir
dela — ou seja, 1 rótulo manual por documento vira 4 exemplos de treino.
"""
import csv
import os
import sys

from PIL import Image
from tqdm import tqdm

from config import CLASSES, LABELS_CSV, PROCESSED_DIR, RAW_DIR, SYNTH_DIR, VALID_EXTENSIONS
from image_utils import apply_label_rotation, letterbox, undo_label_rotation
from pdf_to_images import get_page_images


def load_labels():
    if not os.path.exists(LABELS_CSV):
        return []
    with open(LABELS_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build():
    rows = load_labels()
    if not rows:
        print(f"Nenhum rótulo encontrado em {LABELS_CSV}. Rode label_dataset.py primeiro.")
        sys.exit(1)

    # avisa sobre documentos em raw/ que ainda não foram rotulados
    labeled_keys = {(r["tipo_documento"], r["filename"]) for r in rows}
    pendentes = 0
    if os.path.isdir(RAW_DIR):
        for tipo in os.listdir(RAW_DIR):
            tipo_dir = os.path.join(RAW_DIR, tipo)
            if not os.path.isdir(tipo_dir):
                continue
            for fname in os.listdir(tipo_dir):
                if fname.lower().endswith(VALID_EXTENSIONS) and (tipo, fname) not in labeled_keys:
                    pendentes += 1
    if pendentes:
        print(f"[aviso] {pendentes} documentos em raw/ ainda sem rótulo em labels.csv — rode label_dataset.py.")

    print(f"{len(rows)} documentos rotulados encontrados. Gerando dataset sintético...")

    total_gerado = 0
    for row in tqdm(rows, desc="Processando documentos rotulados"):
        tipo = row["tipo_documento"]
        fname = row["filename"]
        label_real = int(row["rotation_to_fix"])

        doc_path = os.path.join(RAW_DIR, tipo, fname)
        if not os.path.exists(doc_path):
            print(f"[aviso] {doc_path} não encontrado, pulando.")
            continue

        cache_dir = os.path.join(PROCESSED_DIR, tipo)
        page_images = get_page_images(doc_path, cache_dir)

        for img_path in page_images:
            base = os.path.splitext(os.path.basename(img_path))[0]
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                upright = undo_label_rotation(im, label_real)

                for synth_label in CLASSES:
                    rotated = apply_label_rotation(upright, synth_label)
                    rotated = letterbox(rotated)
                    out_dir = os.path.join(SYNTH_DIR, tipo, str(synth_label))
                    os.makedirs(out_dir, exist_ok=True)
                    rotated.save(os.path.join(out_dir, f"{base}.png"))
                    total_gerado += 1

    print(f"\nTotal de exemplos sintéticos gerados: {total_gerado} (a partir de {len(rows)} rótulos manuais)")
    print(f"Salvos em {SYNTH_DIR}")


if __name__ == "__main__":
    build()

