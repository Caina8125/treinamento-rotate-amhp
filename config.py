import os

# Convenção de rótulo (MUITO IMPORTANTE, não mudar sem atualizar build_synthetic_dataset.py,
# label_dataset.py e o README junto): label = quantos graus, NO SENTIDO HORÁRIO, a imagem
# precisa ser rotacionada para ficar em pé (0 = já está correta).
CLASSES = [0, 90, 180, 270]
LABEL_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_LABEL = {i: c for i, c in enumerate(CLASSES)}

IMG_SIZE = 384
RENDER_DPI = 150  # só usado para os raros arquivos .pdf (a maioria do dataset já é imagem)

# A maior parte do dataset real é PNG (bucket de treino de YOLO); alguns
# poucos arquivos aparecem como .jpeg/.pdf — suportamos os dois.
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
PDF_EXTENSIONS = (".pdf",)
VALID_EXTENSIONS = IMAGE_EXTENSIONS + PDF_EXTENSIONS

_HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(_HERE, "rotation_data")

# <tipo_documento>/*.pdf — baixados do S3, ORIENTAÇÃO DESCONHECIDA/ALEATÓRIA.
# Rotulados manualmente via label_dataset.py antes de virar dado de treino.
RAW_DIR = os.path.join(DATA_DIR, "raw")
LABELS_CSV = os.path.join(DATA_DIR, "labels.csv")

PROCESSED_DIR = os.path.join(DATA_DIR, "processed")   # cache: PDFs renderizados em PNG
SYNTH_DIR = os.path.join(DATA_DIR, "synthetic")         # dataset de treino (4 rotações por doc rotulado)

CHECKPOINT_DIR = os.path.join(_HERE, "checkpoints")

BATCH_SIZE = 32
EPOCHS = 15
LR = 3e-4
VAL_SPLIT = 0.15   # fração dos DOCUMENTOS (não dos arquivos sintéticos) reservada p/ validação
SEED = 42
NUM_WORKERS = 4
