"""Avalia o checkpoint treinado por TIPO de documento, não só agregado —
usa exatamente o mesmo split de validação do train.py (mesmo SEED), então
os números aqui são sobre documentos nunca vistos no treino.
"""
import os
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from config import BATCH_SIZE, CHECKPOINT_DIR, CLASSES, NUM_WORKERS, SYNTH_DIR
from dataset import EVAL_TRANSFORM, RotationDataset, list_synthetic_samples, split_train_val
from model import build_model


def tipo_from_path(path: str) -> str:
    # path = SYNTH_DIR/<tipo>/<label>/<fname>
    rel = os.path.relpath(path, SYNTH_DIR)
    return rel.split(os.sep)[0]


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = list_synthetic_samples()
    _, val_samples = split_train_val(samples)

    val_ds = RotationDataset(val_samples, EVAL_TRANSFORM)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    model = build_model(num_classes=len(CLASSES), pretrained=False).to(device)
    checkpoint = torch.load(
        os.path.join(CHECKPOINT_DIR, "rotation_classifier_best.pt"), map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # tipo por item, na MESMA ordem que o DataLoader percorre (shuffle=False, preserva índice)
    tipos = [tipo_from_path(path) for path, _ in val_samples]

    all_preds, all_labels = [], []
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        preds = model(imgs).argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    por_tipo = defaultdict(lambda: [0, 0])  # tipo -> [acertos, total]
    for tipo, pred, label in zip(tipos, all_preds, all_labels):
        por_tipo[tipo][1] += 1
        if pred == label:
            por_tipo[tipo][0] += 1

    print(f"\n{'Tipo':<28}{'Acurácia':>10}{'N (val)':>10}")
    for tipo, (acertos, total) in sorted(por_tipo.items(), key=lambda kv: kv[1][0] / kv[1][1]):
        acc = acertos / total if total else 0
        print(f"{tipo:<28}{acc:>9.1%}{total:>10}")


if __name__ == "__main__":
    main()
