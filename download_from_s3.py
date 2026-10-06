"""Baixa uma amostra aleatória de documentos do S3 pra rotation_data/raw/<tipo>/,
pronta pra rotular com label_dataset.py.

Uso:
  python download_from_s3.py --dry-run          # só lista pastas/formatos, não baixa nada
  python download_from_s3.py                     # baixa a amostra de verdade
  python download_from_s3.py --per-type 200       # muda a quantidade padrão por tipo
"""
import argparse
import os
import random

import boto3
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from config import RAW_DIR

BUCKET = "amhp-teste-ml"
PREFIX = "cid-yolo-dataset/"
DEFAULT_PER_TYPE = 150

# Ajuste aqui se quiser volume diferente pra tipos específicos
# (ex: os 2 que alimentam o Claude). Chave = nome exato da pasta no S3.
OVERRIDES: dict[str, int] = {
    "GUIA_SADT": 350,
    "AUTORIZACAO_SADT": 350,
}

VALID_EXTENSIONS = (".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff")


def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "us-east-1"),
    )


def list_type_folders(s3):
    paginator = s3.get_paginator("list_objects_v2")
    prefixes = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            prefixes.append(cp["Prefix"])
    return sorted(prefixes)


def list_files_in_type(s3, type_prefix):
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=type_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(VALID_EXTENSIONS):
                keys.append(key)
    return keys


def dry_run():
    s3 = get_s3_client()
    folders = list_type_folders(s3)
    print(f"{len(folders)} pastas encontradas em s3://{BUCKET}/{PREFIX}\n")
    for prefix in folders:
        tipo = prefix[len(PREFIX):].rstrip("/")
        keys = list_files_in_type(s3, prefix)
        exts = {}
        for k in keys:
            ext = os.path.splitext(k)[1].lower()
            exts[ext] = exts.get(ext, 0) + 1
        print(f"  {tipo:35s} {len(keys):6d} arquivos   formatos: {exts}")


def download(per_type_default=DEFAULT_PER_TYPE, seed=42):
    s3 = get_s3_client()
    random.seed(seed)
    folders = list_type_folders(s3)
    if not folders:
        print("Nenhuma pasta encontrada — confira bucket/prefixo.")
        return

    total_baixado = 0
    for prefix in folders:
        tipo = prefix[len(PREFIX):].rstrip("/")
        n = OVERRIDES.get(tipo, per_type_default)
        keys = list_files_in_type(s3, prefix)
        if not keys:
            print(f"[{tipo}] nenhum arquivo encontrado, pulando.")
            continue

        amostra = random.sample(keys, min(n, len(keys)))
        out_dir = os.path.join(RAW_DIR, tipo)
        os.makedirs(out_dir, exist_ok=True)

        baixados_agora = 0
        for key in amostra:
            fname = os.path.basename(key)
            dest = os.path.join(out_dir, fname)
            if os.path.exists(dest):
                continue  # idempotente: não baixa de novo se já rodou antes
            s3.download_file(BUCKET, key, dest)
            baixados_agora += 1

        print(f"[{tipo}] {len(amostra)} selecionados de {len(keys)} disponíveis ({baixados_agora} novos baixados)")
        total_baixado += baixados_agora

    print(f"\nTotal de arquivos novos baixados: {total_baixado}")
    print(f"Salvos em {RAW_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="só lista pastas/formatos, não baixa nada")
    parser.add_argument("--per-type", type=int, default=DEFAULT_PER_TYPE)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
    else:
        download(per_type_default=args.per_type, seed=args.seed)
