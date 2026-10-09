"""Converte todo PDF dentro de rotation_data/raw/<tipo>/ pra PNG, deixando o
dataset 100% imagem (consistente com o resto, que já é PNG). PDF de N
páginas vira N PNGs (<stem>_p0.png, <stem>_p1.png, ...).

Atualiza labels.csv junto: linha já rotulada do PDF original é substituída
por uma linha por página gerada (mesmo tipo_documento, mesma rotação — a
orientação do scan é a mesma em todas as páginas do mesmo documento).

NÃO rode isso com o label_dataset.py aberto (os dois escrevem no mesmo
labels.csv).
"""
import csv
import os

from config import LABELS_CSV, RAW_DIR, RENDER_DPI
from pdf_to_images import render_pdf_to_images


def load_labels():
    if not os.path.exists(LABELS_CSV):
        return []
    with open(LABELS_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_labels(rows):
    with open(LABELS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "tipo_documento", "rotation_to_fix"])
        writer.writeheader()
        writer.writerows(rows)


def find_pdfs():
    pdfs = []
    for tipo in sorted(os.listdir(RAW_DIR)):
        tipo_dir = os.path.join(RAW_DIR, tipo)
        if not os.path.isdir(tipo_dir):
            continue
        for fname in sorted(os.listdir(tipo_dir)):
            if fname.lower().endswith(".pdf"):
                pdfs.append((tipo, fname))
    return pdfs


def main():
    rows = load_labels()
    pdfs = find_pdfs()
    print(f"{len(pdfs)} PDF(s) encontrados em raw/.")

    convertidos = 0
    paginas_totais = 0
    for tipo, fname in pdfs:
        tipo_dir = os.path.join(RAW_DIR, tipo)
        pdf_path = os.path.join(tipo_dir, fname)

        novos_nomes = render_pdf_to_images(pdf_path, tipo_dir, dpi=RENDER_DPI)
        novos_nomes = [os.path.basename(p) for p in novos_nomes]

        if len(novos_nomes) == 1:
            # 1 página: renomeia pra <stem>.png (sem sufixo _p0)
            stem = os.path.splitext(fname)[0]
            final_name = f"{stem}.png"
            os.replace(os.path.join(tipo_dir, novos_nomes[0]), os.path.join(tipo_dir, final_name))
            novos_nomes = [final_name]

        # substitui a linha do PDF original (se já rotulado) por 1 linha por página nova
        linha_original = next((r for r in rows if r["tipo_documento"] == tipo and r["filename"] == fname), None)
        if linha_original:
            rows = [r for r in rows if not (r["tipo_documento"] == tipo and r["filename"] == fname)]
            for novo_nome in novos_nomes:
                rows.append({
                    "filename": novo_nome,
                    "tipo_documento": tipo,
                    "rotation_to_fix": linha_original["rotation_to_fix"],
                })

        os.remove(pdf_path)
        convertidos += 1
        paginas_totais += len(novos_nomes)

    write_labels(rows)
    print(f"{convertidos} PDF(s) convertidos em {paginas_totais} PNG(s) no total.")
    print(f"labels.csv atualizado — {len(rows)} linhas.")


if __name__ == "__main__":
    main()
