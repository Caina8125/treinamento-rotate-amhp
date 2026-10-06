# Classificador de rotação de documentos

Substitui o uso do Textract só-para-saber-a-rotação por uma CNN própria
(4 classes: 0°/90°/180°/270°).

**Os documentos baixados do S3 têm orientação desconhecida/aleatória** — não
existe um lote "já correto" pra gerar rótulo de graça. Por isso, cada
documento passa por uma rotulagem manual rápida (você olha e diz qual é a
rotação real). A partir **desse único rótulo**, o pipeline reconstrói a
versão "em pé" do documento e regenera as 4 rotações sintéticas sozinho — ou
seja, é 1 clique por documento, não 4, mesmo que o dataset de treino final
tenha 4x mais exemplos.

## Convenção de rótulo — IMPORTANTE

**`label` = quantos graus, no SENTIDO HORÁRIO, a imagem precisa ser rotacionada
para ficar em pé.** `0` significa "já está correta".

Usada em todo o código (`config.py`, `label_dataset.py`,
`build_synthetic_dataset.py`).

## 0. Baixar a amostra do S3

```
python download_from_s3.py --dry-run     # só lista as 27 pastas e formatos, não baixa nada
python download_from_s3.py                # baixa 150 de cada tipo em rotation_data/raw/<tipo>/
```

Usa `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` do `.env` na raiz do projeto
(`C:\metrificar_extracao\.env`, o mesmo usado no projeto de custo do Claude).
Bucket: `amhp-teste-ml/cid-yolo-dataset`. É idempotente — rodar de novo não
baixa os arquivos que já estão em disco.

**Formato real descoberto no dry-run: é imagem (PNG), não PDF** — só uns
poucos arquivos avulsos vieram em `.jpeg`/`.pdf`. O pipeline trata os dois
formatos automaticamente (`pdf_to_images.get_page_images`).

**Quantidade por tipo**: por padrão baixa **150 de cada um dos 27 tipos**.
Se vocês quiserem mais volume em `GUIA_SADT`/`AUTORIZACAO_SADT` (os que
aparentam ser os 2 que alimentam o Claude, a julgar pelo nome — confirmar)
como discutido antes (300-400 cada), edite o dicionário `OVERRIDES` no topo
de `download_from_s3.py`:

```python
OVERRIDES = {
    "GUIA_SADT": 350,
    "AUTORIZACAO_SADT": 350,
}
```

Alguns tipos têm bem menos de 150 disponíveis no bucket (ex: `ASSINATURAS_GDF`,
`AUTORIZACAO_TEXTO`, `GUIA_SISTEMA_CONSULTA`, `PAGINA_BRANCO` têm exatamente
150 ou perto disso) — o script baixa o que tiver disponível, sem erro.

## 1. Estrutura de dados

```
rotation_data/
  raw/
    GUIA_SADT/                 <- um subdiretório por tipo (nome = pasta no S3)
      arquivo001.png            <- baixado do S3, rotação desconhecida
      arquivo002.png
      ...
    AUTORIZACAO_SADT/
      ...
    <outros 25 tipos>/
      ...
  labels.csv                    <- gerado automaticamente por label_dataset.py, não editar à mão
```

## 2. Rotular (local, com tela — não roda na instância GPU headless)

```
pip install -r requirements.txt
python label_dataset.py
```

Abre uma janela mostrando cada documento de `raw/<tipo>/` um por vez. Você:
- Clica no grau certo (ou aperta `0` / `9` / `1` / `2` no teclado)
- `S` pra pular um que você não tem certeza
- `Backspace` pra desfazer o último, se errar o clique

Salva em `labels.csv` a cada rótulo — pode fechar e reabrir a qualquer
momento, ele lembra o que já foi feito e só mostra o que falta.

**Isso é trabalho real, mas rápido por item** (um olhar + um clique, não
leitura/transcrição) — pros ~4.050 documentos do volume padrão (150×27),
estimar algo entre 3-6 horas no total, dá pra dividir em sessões.

## 3. Gerar o dataset sintético de treino

```
python build_synthetic_dataset.py
```

Lê `labels.csv`, desfaz a rotação real de cada documento (reconstruindo a
versão em pé) e gera as 4 rotações sintéticas a partir dela. Avisa se sobrou
algum documento em `raw/` que ainda não foi rotulado.

## 4. Treinar

```
python train.py
```

- Separa treino/validação **por documento de origem** (não por arquivo
  sintético) — garante que as 4 rotações do mesmo documento nunca fiquem
  split entre treino e validação, senão a acurácia de validação fica
  otimista demais.
- Imprime acurácia por época e, ao final, a matriz de confusão 4x4. Uma
  confusão sistemática entre 90°↔270° é sinal clássico de bug de convenção
  de sentido (horário/anti-horário) em algum lugar — não necessariamente do
  modelo.

## 5. Exportar pra produção

```
python export_onnx.py
```

## Próximos passos depois de treinado

- Avaliar acurácia por **tipo de documento** separadamente, não só agregada —
  tipos com pouco texto/muita imagem tendem a ser mais difíceis.
- Se algum tipo específico performar mal, rotular mais exemplos daquele tipo
  especificamente, em vez de aumentar tudo proporcionalmente.
- Depois de validado, o `.onnx` em `checkpoints/` substitui a chamada ao
  Textract no passo de rotação do pipeline de produção.
