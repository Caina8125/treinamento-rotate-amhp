import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import BATCH_SIZE, CHECKPOINT_DIR, CLASSES, EPOCHS, IDX_TO_LABEL, LR, NUM_WORKERS
from dataset import EVAL_TRANSFORM, TRAIN_TRANSFORM, RotationDataset, list_synthetic_samples, split_train_val
from model import build_model


def confusion_matrix(preds, labels, n_classes=4):
    cm = [[0] * n_classes for _ in range(n_classes)]
    for p, y in zip(preds, labels):
        cm[y][p] += 1
    return cm


def print_confusion_matrix(cm):
    print("\nMatriz de confusão (linha=real, coluna=previsto, graus de correção):")
    header = "        " + "".join(f"{IDX_TO_LABEL[i]:>7}°" for i in range(len(CLASSES)))
    print(header)
    for i, row in enumerate(cm):
        print(f"{IDX_TO_LABEL[i]:>6}° " + "".join(f"{v:>8}" for v in row))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        preds = logits.argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
    acc = sum(p == y for p, y in zip(all_preds, all_labels)) / max(len(all_labels), 1)
    return acc, all_preds, all_labels


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    samples = list_synthetic_samples()
    if not samples:
        raise SystemExit(
            "Nenhum exemplo sintético encontrado. Rode label_dataset.py e depois build_synthetic_dataset.py primeiro."
        )
    train_samples, val_samples = split_train_val(samples)

    print(f"Treino: {len(train_samples)} exemplos")
    print(f"Validação: {len(val_samples)} exemplos (de documentos nunca vistos no treino)")

    train_ds = RotationDataset(train_samples, TRAIN_TRANSFORM)
    val_ds = RotationDataset(val_samples, EVAL_TRANSFORM)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    model = build_model(num_classes=len(CLASSES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_acc = -1.0
    best_path = os.path.join(CHECKPOINT_DIR, "rotation_classifier_best.pt")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        scheduler.step()

        train_loss = running_loss / len(train_ds)
        val_acc, _, _ = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:2d}/{EPOCHS} | loss={train_loss:.4f} | val_acc={val_acc:.2%}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model_state": model.state_dict(), "classes": CLASSES}, best_path)
            print(f"  -> novo melhor checkpoint salvo ({best_acc:.2%})")

    print(f"\nMelhor checkpoint: {best_path} (acurácia de validação: {best_acc:.2%})")

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    val_acc, val_preds, val_labels = evaluate(model, val_loader, device)
    print(f"\nAcurácia final de validação: {val_acc:.2%}")
    print_confusion_matrix(confusion_matrix(val_preds, val_labels))


if __name__ == "__main__":
    train()
