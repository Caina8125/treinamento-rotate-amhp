"""Ferramenta visual pra rotular rotation_data/raw/<tipo>/*: mostra cada
documento (imagem ou PDF), você clica (ou aperta a tecla) do grau de correção
VERDADEIRO, e ele escreve em rotation_data/labels.csv sozinho. Lembra onde
parou entre sessões.

Roda localmente (precisa de tela — não roda na instância GPU headless).
"""
import csv
import os

import customtkinter as ctk
from PIL import Image

from config import LABELS_CSV, PROCESSED_DIR, RAW_DIR, VALID_EXTENSIONS
from pdf_to_images import get_page_images

ctk.set_appearance_mode("dark")

FIELDNAMES = ["filename", "tipo_documento", "rotation_to_fix"]


def list_all_documents():
    """Varre raw/<tipo>/*. `filename` no CSV é o nome do arquivo (único dentro
    do tipo); a dupla (tipo, filename) é a chave única no dataset todo."""
    pairs = []
    if not os.path.isdir(RAW_DIR):
        return pairs
    for tipo in sorted(os.listdir(RAW_DIR)):
        tipo_dir = os.path.join(RAW_DIR, tipo)
        if not os.path.isdir(tipo_dir):
            continue
        for fname in sorted(os.listdir(tipo_dir)):
            if fname.lower().endswith(VALID_EXTENSIONS):
                pairs.append((tipo, fname))
    return pairs


def load_valid_rows():
    """Lê labels.csv e descarta linhas cujo arquivo não existe mais."""
    rows = []
    if os.path.exists(LABELS_CSV):
        with open(LABELS_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                full_path = os.path.join(RAW_DIR, row["tipo_documento"], row["filename"])
                if os.path.exists(full_path):
                    rows.append(row)
    return rows


def write_rows(rows):
    os.makedirs(os.path.dirname(LABELS_CSV), exist_ok=True)
    with open(LABELS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


class LabelApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Rotular documentos — rotação real")
        self.geometry("760x900")
        self.resizable(False, False)

        self.rows = load_valid_rows()
        write_rows(self.rows)  # já salva limpo
        labeled_keys = {(r["tipo_documento"], r["filename"]) for r in self.rows}

        all_pairs = list_all_documents()
        self.total_docs = len(all_pairs)
        self.queue = [p for p in all_pairs if p not in labeled_keys]
        self.index = 0
        self.tk_img = None
        self.last_labeled = None  # pra permitir desfazer

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=(16, 4))
        ctk.CTkLabel(header, text="Qual rotação corrige este documento?", font=("Segoe UI", 16, "bold")).pack(anchor="w")

        self.progress_bar = ctk.CTkProgressBar(self, width=680)
        self.progress_bar.pack(pady=(8, 4))

        self.tipo_label = ctk.CTkLabel(self, text="", font=("Segoe UI", 13, "bold"), text_color="#22d3ee")
        self.tipo_label.pack()

        self.image_label = ctk.CTkLabel(self, text="")
        self.image_label.pack(pady=10)

        self.status_label = ctk.CTkLabel(self, text="", font=("Segoe UI", 12), text_color="#8b96ad")
        self.status_label.pack()

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(pady=16)
        labels = [("0°\n(já está certo)", 0), ("90°\nhorário", 90), ("180°", 180), ("270°\nhorário", 270)]
        for text, deg in labels:
            ctk.CTkButton(
                btn_frame, text=text, width=150, height=60, font=("Segoe UI", 13),
                command=lambda d=deg: self.label_current(d),
            ).pack(side="left", padx=6)

        bottom_frame = ctk.CTkFrame(self, fg_color="transparent")
        bottom_frame.pack(pady=(4, 10))
        ctk.CTkButton(
            bottom_frame, text="Pular (não sei)", fg_color="gray30", hover_color="gray40",
            command=self.skip_current,
        ).pack(side="left", padx=6)
        self.undo_btn = ctk.CTkButton(
            bottom_frame, text="↶ Desfazer último", fg_color="gray30", hover_color="gray40",
            command=self.undo_last,
        )
        self.undo_btn.pack(side="left", padx=6)

        ctk.CTkLabel(
            self, text="Atalhos: 0 / 9 / 1 / 2 para rotular  ·  S para pular  ·  Backspace para desfazer",
            font=("Segoe UI", 11), text_color="#8b96ad",
        ).pack()

        self.bind("0", lambda e: self.label_current(0))
        self.bind("9", lambda e: self.label_current(90))
        self.bind("1", lambda e: self.label_current(180))
        self.bind("2", lambda e: self.label_current(270))
        self.bind("s", lambda e: self.skip_current())
        self.bind("S", lambda e: self.skip_current())
        self.bind("<BackSpace>", lambda e: self.undo_last())

        self.show_current()

    def show_current(self):
        done = self.total_docs - len(self.queue)
        self.progress_bar.set(done / self.total_docs if self.total_docs else 0)

        if self.index >= len(self.queue):
            self.image_label.configure(image=None, text="✓ Tudo rotulado — pode fechar.")
            self.tipo_label.configure(text="")
            self.status_label.configure(text=f"{len(self.rows)} de {self.total_docs} documentos rotulados")
            return

        tipo, fname = self.queue[self.index]
        doc_path = os.path.join(RAW_DIR, tipo, fname)
        cache_dir = os.path.join(PROCESSED_DIR, tipo)
        rendered = get_page_images(doc_path, cache_dir)
        img = Image.open(rendered[0]).convert("RGB")
        img.thumbnail((640, 640))
        self.tk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
        self.image_label.configure(image=self.tk_img, text="")
        self.tipo_label.configure(text=tipo)
        self.status_label.configure(
            text=f"{fname}  —  {done + 1} de {self.total_docs} no total"
        )

    def label_current(self, degrees):
        if self.index >= len(self.queue):
            return
        tipo, fname = self.queue[self.index]
        row = {"filename": fname, "tipo_documento": tipo, "rotation_to_fix": str(degrees)}
        self.rows.append(row)
        write_rows(self.rows)
        self.last_labeled = (tipo, fname)
        self.index += 1
        self.show_current()

    def skip_current(self):
        self.index += 1
        self.show_current()

    def undo_last(self):
        if not self.rows or self.last_labeled is None:
            return
        # remove a última linha rotulada e recoloca na fila, na posição atual
        if self.rows[-1]["tipo_documento"] == self.last_labeled[0] and self.rows[-1]["filename"] == self.last_labeled[1]:
            self.rows.pop()
            write_rows(self.rows)
            self.queue.insert(self.index, self.last_labeled)
            self.last_labeled = None
            self.show_current()


if __name__ == "__main__":
    os.makedirs(RAW_DIR, exist_ok=True)
    app = LabelApp()
    app.mainloop()
