"""Exporta o melhor checkpoint para ONNX — inferência leve em produção, sem
precisar de PyTorch completo/CUDA no serviço que consome o modelo."""
import os

import torch

from config import CHECKPOINT_DIR, CLASSES, IMG_SIZE
from model import build_model


def export(checkpoint_name: str = "rotation_classifier_best.pt", onnx_name: str = "rotation_classifier.onnx"):
    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model = build_model(num_classes=len(CLASSES), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    onnx_path = os.path.join(CHECKPOINT_DIR, onnx_name)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,  # exportador "legado": evita depender do pacote onnxscript
    )
    print(f"Exportado para {onnx_path}")
    print(f"Classes (índice -> graus de correção): {dict(enumerate(CLASSES))}")


if __name__ == "__main__":
    export()
