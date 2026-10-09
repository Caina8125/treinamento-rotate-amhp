"""Ferramenta visual pra classificar documentos de um lote externo (ex: erros
de rotação encontrados em produção) dentro das categorias já existentes em
rotation_data/raw/<tipo>/. Mostra cada documento, você clica no tipo certo,
e ele COPIA (não move — o lote original fica intacto) pra
rotation_data/raw/<tipo_escolhido>/<arquivo>.

Depois de classificar, use o label_dataset.py normalmente — ele vai detectar
esses arquivos novos dentro das pastas de tipo e pedir o rótulo de rotação.

Uso:
  python sort_erros_producao.py "C:\\Erros_Rotacao"
"""
import os
import shutil
import sys

import customtkinter as ctk
from PIL import Image

from config import RAW_DIR, VALID_EXTENSIONS
from pdf_to_images import get_page_images

ctk.set_appearance_mode("dark")

_TIPOS_POR_LINHA = 4


def list_tipos():
    os.makedirs(RAW_DIR, exist_ok=True)
    return sorted(d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d)))


def list_pendentes(origem_dir):
    return sorted(f for f in os.listdir(origem_dir) if f.lower().endswith(VALID_EXTENSIONS))


class SortApp(ctk.CTk):
    def __init__(self, origem_dir):
        super().__init__()
        self.origem_dir = origem_dir
        self.title("Classificar documentos por tipo")
        self.geometry("820x900")
        self.resizable(True, True)

        self.tipos = list_tipos()
        if not self.tipos:
            raise SystemExit(f"Nenhuma pasta de tipo encontrada em {RAW_DIR}.")

        self.queue = list_pendentes(origem_dir)
        self.index = 0
        self.tk_img = None
        self.last_sorted = None  # (nome_arquivo, tipo) pra desfazer

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=(16, 4))
        ctk.CTkLabel(header, text="Qual é o tipo deste documento?", font=("Segoe UI", 16, "bold")).pack(anchor="w")

        self.image_label = ctk.CTkLabel(self, text="")
        self.image_label.pack(pady=10)

        self.status_label = ctk.CTkLabel(self, text="", font=("Segoe UI", 12), text_color="#8b96ad")
        self.status_label.pack()

        grid_frame = ctk.CTkScrollableFrame(self, width=760, height=220, fg_color="transparent")
        grid_frame.pack(pady=14)
        for i, tipo in enumerate(self.tipos):
            row, col = divmod(i, _TIPOS_POR_LINHA)
            ctk.CTkButton(
                grid_frame, text=tipo, width=180, height=40, font=("Segoe UI", 11),
                command=lambda t=tipo: self.classify_current(t),
            ).grid(row=row, column=col, padx=4, pady=4)

        bottom_frame = ctk.CTkFrame(self, fg_color="transparent")
        bottom_frame.pack(pady=(10, 10))
        ctk.CTkButton(
            bottom_frame, text="Pular (não sei)", fg_color="gray30", hover_color="gray40",
            command=self.skip_current,
        ).pack(side="left", padx=6)
        self.undo_btn = ctk.CTkButton(
            bottom_frame, text="↶ Desfazer último", fg_color="gray30", hover_color="gray40",
            command=self.undo_last,
        )
        self.undo_btn.pack(side="left", padx=6)

        self.show_current()

    def show_current(self):
        total = len(self.queue) + (self.index if self.last_sorted is None else 0)
        if self.index >= len(self.queue):
            self.image_label.configure(image=None, text="✓ Tudo classificado — pode fechar.")
            self.status_label.configure(text="")
            return

        fname = self.queue[self.index]
        doc_path = os.path.join(self.origem_dir, fname)
        cache_dir = os.path.join(self.origem_dir, "_preview_cache")
        rendered = get_page_images(doc_path, cache_dir)
        img = Image.open(rendered[0]).convert("RGB")
        img.thumbnail((480, 480))
        self.tk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
        self.image_label.configure(image=self.tk_img, text="")
        self.status_label.configure(text=f"{fname}  —  {self.index + 1} de {len(self.queue)}")

    def classify_current(self, tipo):
        if self.index >= len(self.queue):
            return
        fname = self.queue[self.index]
        destino_dir = os.path.join(RAW_DIR, tipo)
        os.makedirs(destino_dir, exist_ok=True)
        shutil.copy2(os.path.join(self.origem_dir, fname), os.path.join(destino_dir, fname))
        self.last_sorted = (fname, tipo)
        self.index += 1
        self.show_current()

    def skip_current(self):
        self.index += 1
        self.show_current()

    def undo_last(self):
        if self.last_sorted is None:
            return
        fname, tipo = self.last_sorted
        destino = os.path.join(RAW_DIR, tipo, fname)
        if os.path.exists(destino):
            os.remove(destino)
        self.index -= 1
        self.last_sorted = None
        self.show_current()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(r'Uso: python sort_erros_producao.py "C:\Erros_Rotacao"')
        sys.exit(1)
    app = SortApp(sys.argv[1])
    app.mainloop()
