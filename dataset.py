import os
import random
from collections import defaultdict

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import LABEL_TO_IDX, SEED, SYNTH_DIR, VAL_SPLIT

_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        _NORMALIZE,
    ]
)
EVAL_TRANSFORM = transforms.Compose([transforms.ToTensor(), _NORMALIZE])


class RotationDataset(Dataset):
    def __init__(self, samples: list[tuple[str, int]], transform):
        """samples: lista de (caminho_imagem, label_idx). As imagens do
        dataset sintético já foram salvas em IMG_SIZE, não precisa redimensionar de novo."""
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            tensor = self.transform(img)
        return tensor, label


def list_synthetic_samples() -> list[tuple[str, int, str]]:
    """Retorna (caminho, label_idx, doc_key). `doc_key` identifica o documento
    de origem (tipo + nome base) — as 4 rotações do mesmo documento compartilham
    o mesmo doc_key, usado pra nunca separar elas entre treino e validação."""
    samples = []
    if not os.path.isdir(SYNTH_DIR):
        return samples
    for tipo in sorted(os.listdir(SYNTH_DIR)):
        tipo_dir = os.path.join(SYNTH_DIR, tipo)
        if not os.path.isdir(tipo_dir):
            continue
        for label_str in sorted(os.listdir(tipo_dir)):
            label_dir = os.path.join(tipo_dir, label_str)
            if not os.path.isdir(label_dir):
                continue
            label_idx = LABEL_TO_IDX[int(label_str)]
            for fname in os.listdir(label_dir):
                if fname.lower().endswith(".png"):
                    doc_key = f"{tipo}/{fname}"  # mesmo nome de base em todo label_str
                    samples.append((os.path.join(label_dir, fname), label_idx, doc_key))
    return samples


def split_train_val(samples: list[tuple[str, int, str]], val_frac: float = VAL_SPLIT):
    """Divide por DOCUMENTO de origem (doc_key), não por arquivo — senão as 4
    rotações do mesmo documento vazam entre treino e validação e a acurácia
    de validação fica otimista demais (o modelo reconhece o documento, não a
    orientação)."""
    by_doc = defaultdict(list)
    for path, label, doc_key in samples:
        by_doc[doc_key].append((path, label))

    doc_keys = list(by_doc.keys())
    rng = random.Random(SEED)
    rng.shuffle(doc_keys)
    n_val_docs = int(len(doc_keys) * val_frac)
    val_keys = set(doc_keys[:n_val_docs])

    train_samples, val_samples = [], []
    for doc_key, items in by_doc.items():
        target = val_samples if doc_key in val_keys else train_samples
        target.extend(items)
    return train_samples, val_samples
