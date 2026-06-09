# Uputstvo za reprodukciju rezultata i generisanje slika za izveštaj

> Ovaj fajl je namenjen da se prosledi drugoj Claude instanci (ili da se ručno
> izvrši). Cilj je: (1) istrenirati model, (2) izračunati numeričke metrike i
> (3) generisati slike koje u `report/main.tex` trenutno stoje kao **rezervisana
> mesta** (`[MESTO ZA SLIKU]`) ili kao `[XX,X%]` / `[0,XX]` brojevi.
>
> Sve skripte iz repozitorijuma (`train.py`, `dataset.py`, `model.py`,
> `gradcam_viz.py`) već postoje i rade. Treba samo dodati skripte za evaluaciju
> i za par dodatnih slika, pokrenuti ih, i ubaciti rezultate u LaTeX.

## 0. Okruženje

- GPU: NVIDIA RTX 4090 (24 GB), iznajmljen preko **vast.ai**.
- Skup podataka raspakovan u:
  `DATA_ROOT = /root/dataset/LumbarSpinalStenosis/LumbarSpinalStenosis`
  sa podfolderima `train/` i `test/`, a u svakom klase
  `Herniated Disc`, `No Stenosis`, `Thecal Sac`.
- Klasa `Thecal Sac` se **izostavlja** (`EXCLUDE_CLASSES = {"Thecal Sac"}`),
  problem je binarni: *Herniated Disc* vs *No Stenosis*.
- Zavisnosti: `torch`, `torchvision`, `scikit-learn`, `matplotlib`, `numpy`,
  `Pillow`. Dodatno za evaluaciju ispod: `scikit-learn` (već potreban).

## 1. Treniranje (već implementirano) — DVE arhitekture

U izveštaju se porede dva modela: **ResNet-50** (od nule) i **ViT-Base/16**
(pretreniran na ImageNet-21k, fino podešen). Oba se treniraju **istom skriptom**
`train.py` — model se bira parametrom `MODEL_TYPE` na vrhu fajla
(`"resnet50"` ili `"vit"`), a hiperparametri (LR, weight decay, optimizator,
putanje) se automatski biraju u `_MODEL_CFG`. (Raniji zaseban `train_vit.py`
je obrisan i objedinjen u `train.py`.)

### 1a. ResNet-50
```bash
# u train.py postaviti: MODEL_TYPE = "resnet50"
python3 train.py
```
Rezultat:
- `best_model_resnet50.pth` — težine sa najboljom tačnošću na validaciji.
- `best_model_resnet50_curves.png` — krive treniranja. **Ovo je Slika 3**
  (`report/figures/learning_curves.png`). Posle ponovnog treniranja:
  ```bash
  cp best_model_resnet50_curves.png report/figures/learning_curves.png
  ```

### 1b. ViT-Base/16
Zahteva `timm` (AdamW, LR=1e-4, weight_decay=1e-2, pretrained=True).
```bash
pip install timm
# u train.py postaviti: MODEL_TYPE = "vit"
python3 train.py
```
Rezultat:
- `best_model_vit.pth` — težine ViT modela.
- `best_model_vit_curves.png` — krive treniranja. **Ovo je Slika 4**
  (`fig:krive-vit`, trenutno placeholder). Posle treniranja:
  ```bash
  cp best_model_vit_curves.png report/figures/vit_curves.png
  ```
  pa u `main.tex` zameniti placeholder za `fig:krive-vit` sa
  `\includegraphics[width=\linewidth]{figures/vit_curves.png}`.

> Napomene:
> - Treniranje ide do budžeta `EPOCHS`, ali se čuva model sa najboljom
>   validacionom tačnošću (efektivno rano zaustavljanje) + međučekpoint na
>   `CKPT_INTERVAL` sekundi. Tok se loguje u **wandb** (`WANDB_PROJECT`), a za
>   ResNet se na `GRADCAM_INTERVAL` epoha loguju i Grad-CAM slike.
> - ViT **ne podržava Grad-CAM** (nema prostorne mape obeležja). Grad-CAM se
>   radi samo na ResNet-u (`gradcam_viz.py`); za ViT koristiti attention
>   rollout (budući rad).
> - wandb linkovi runova (`[wandb-link-resnet]`, `[wandb-link-vit]`) ubacuju se
>   u `main.tex`, odeljak Treniranje.

## 2. Grad-CAM slike (već implementirano)

```bash
python3 gradcam_viz.py
```

Konfiguracija na vrhu `gradcam_viz.py` (`MODEL_PATH`, `N_TRAIN`, `N_VAL`, ...).
Rezultat je `gradcam_viz/summary.png` + pojedinačne slike.
**`summary.png` je već slika `report/figures/gradcam_summary.png`** (Slika 5).
Posle novog pokretanja:
```bash
cp gradcam_viz/summary.png report/figures/gradcam_summary.png
```

## 3. Evaluacija + matrica konfuzije + metrike  (TREBA NAPISATI)

Napisati `eval.py` koji učitava model, prolazi kroz **test** skup (ceo, ne
uzorkovan), i računa metrike. Pozitivna klasa = `Herniated Disc`.

**Pokrenuti za OBA modela** i popuniti po jednu kolonu u Tabeli 3 i po jedan red
u Tabeli 4:
- ResNet-50: `MODEL_PATH = "best_model_resnet50.pth"`, `build_model(...)`.
- ViT-B/16:  `MODEL_PATH = "best_model_vit.pth"`, `build_vit(..., pretrained=False)`
  (arhitektura se pravi prazna pa se učitava `state_dict`).

Skelet (oslanja se na postojeće module `dataset.py` i `model.py`):

```python
# eval.py
import numpy as np, torch
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import datasets
from sklearn.metrics import (confusion_matrix, roc_auc_score, roc_curve,
                             accuracy_score, precision_score, recall_score, f1_score)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from dataset import get_transform, IMG_SIZE if False else None  # vidi dole
from model import build_model

DATA_ROOT = "/root/dataset/LumbarSpinalStenosis/LumbarSpinalStenosis"
MODEL_PATH = "best_model.pth"
EXCLUDE = {"Thecal Sac"}
IMG_SIZE = 224
POS_CLASS = "Herniated Disc"   # pozitivna klasa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- test skup, bez izostavljene klase ---
from dataset import get_transform
tf = get_transform(IMG_SIZE)
raw = datasets.ImageFolder(f"{DATA_ROOT}/test")
class_names = [c for c in raw.classes if c not in EXCLUDE]
keep = [i for i,(_,l) in enumerate(raw.samples) if raw.classes[l] not in EXCLUDE]
ds = Subset(datasets.ImageFolder(f"{DATA_ROOT}/test", transform=tf), keep)
loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=8)

# PAŽNJA: indeksi klasa. ImageFolder mapira po abecedi nad SVE tri klase.
# Posle izbacivanja "Thecal Sac" zadrži originalni label->ime preko raw.classes,
# pa pozitivnu klasu definiši kao raw.class_to_idx[POS_CLASS].
pos_idx = raw.class_to_idx[POS_CLASS]

model = build_model(len(class_names), pretrained=False).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device)); model.eval()

# VAŽNO: build_model pravi 2 izlaza čiji redosled odgovara `class_names`
# (alfabetski, bez izostavljenih). Uskladi mapiranje predikcija sa imenima klasa
# pre poređenja sa pos_idx (npr. preko class_names.index(POS_CLASS)).
pos_out = class_names.index(POS_CLASS)

y_true, y_score, y_pred = [], [], []
with torch.no_grad():
    for imgs, labels in loader:
        # labels su indeksi nad ImageFolder(test) BEZ izostavljanja? Ne — Subset
        # zadržava originalne labele iz pune ImageFolder mape. Konvertuj u
        # binarno: 1 ako je uzorak Herniated Disc.
        logits = model(imgs.to(device))
        prob = torch.softmax(logits, 1)[:, pos_out].cpu().numpy()
        pred = logits.argmax(1).cpu().numpy()            # indeks u class_names
        # mapiraj true labele (ImageFolder full mapa) -> binarno
        # ... popuniti prema stvarnoj mapi ...

# Izračunaj: accuracy_score, recall_score (sens), specificity (= TN/(TN+FP) iz CM),
# precision_score, f1_score, roc_auc_score(y_true, y_score).
```

> Napomena za izvršioca: **najsigurnije** je da `eval.py` ponovo iskoristi
> `make_loaders(...)` iz `dataset.py` sa `max_val=0`? Ne — `make_loaders` meša
> train-split i test folder u validaciju. Za čistu evaluaciju na test folderu
> napravi loader direktno iz `test/` foldera kao gore i pažljivo uskladi
> indekse klasa (to je jedino mesto gde se lako greši).

Skripta treba da:
1. **Ispiše** brojeve i njima popuni LaTeX:
   - `Sažetak`: tačnost / senzitivnost / specifičnost.
   - `Tabela 3` (`tab:rezultati`): Accuracy, Sensitivity, Specificity,
     Precision, F1, AUC — zameniti `[XX,X%]` i `[0,XX]`.
   - `Tabela 4` (`tab:poredjenje`): red "Naš model".
   - **Decimalni zarez**, ne tačka (srpski) — npr. `88,3\%`, `0,94`.
2. **Sačuva matricu konfuzije** kao `report/figures/confusion_matrix.png`:
   ```python
   cm = confusion_matrix(y_true, y_pred)   # red=true, kol=pred, [No Stenosis, Herniated]
   fig, ax = plt.subplots(figsize=(4,4))
   im = ax.imshow(cm, cmap="Blues")
   ax.set_xticks([0,1]); ax.set_yticks([0,1])
   ax.set_xticklabels(["Uredan", "Hernija"]); ax.set_yticklabels(["Uredan","Hernija"])
   ax.set_xlabel("Predviđeno"); ax.set_ylabel("Stvarno")
   for i in range(2):
       for j in range(2):
           ax.text(j, i, f"{cm[i,j]}\n{cm[i,j]/cm.sum()*100:.1f}%",
                   ha="center", va="center")
   plt.tight_layout(); plt.savefig("report/figures/confusion_matrix.png", dpi=150)
   ```

Zatim u `main.tex` zameniti placeholder za Sliku 4:
```latex
% bilo:
\placeholderbox{6cm}{Matrica konfuzije ...}
% novo:
\includegraphics[width=0.5\linewidth]{figures/confusion_matrix.png}
```

## 4. Primeri snimaka iz dve klase  (Slika 1 / `fig:primeri`)  (TREBA NAPISATI)

Mali montaž od npr. 2 snimka (po jedan iz svake klase). Skripta `make_examples.py`:

```python
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from torchvision import datasets
from dataset import get_transform, IMG_MEAN, IMG_STD
import numpy as np, torch

DATA_ROOT="/root/dataset/LumbarSpinalStenosis/LumbarSpinalStenosis"
def first_of(folder, cls):
    ds = datasets.ImageFolder(folder, transform=get_transform(224))
    idx = ds.class_to_idx[cls]
    i = next(k for k,(_,l) in enumerate(ds.samples) if l==idx)
    img,_ = ds[i]
    return np.clip(img.permute(1,2,0).numpy()*np.array(IMG_STD)+np.array(IMG_MEAN),0,1)

fig,ax=plt.subplots(1,2,figsize=(8,4))
ax[0].imshow(first_of(f"{DATA_ROOT}/train","Herniated Disc")); ax[0].set_title("Hernija diska"); ax[0].axis("off")
ax[1].imshow(first_of(f"{DATA_ROOT}/train","No Stenosis"));   ax[1].set_title("Uredan nalaz");  ax[1].axis("off")
plt.tight_layout(); plt.savefig("report/figures/examples.png", dpi=150)
```

Zatim zameniti placeholder za Sliku 1 sa
`\includegraphics[width=0.8\linewidth]{figures/examples.png}`.

## 5. Primeri promašaja  (Slika 6 / `fig:promasaj`)  (TREBA NAPISATI)

Najlakše: modifikovati `gradcam_viz.py` da prolazi kroz veći uzorak test skupa i
**čuva samo slučajeve gde je `pred != true`**, pa od prva 2–3 takva napraviti
montažu (isti 3-panel format: ulaz / Grad-CAM / superpozicija). Snimiti kao
`report/figures/misclassified.png` i zameniti placeholder za Sliku 6.

## 6. Finalni korak

Posle ubacivanja svih slika i brojeva, prevesti izveštaj:

```bash
cd report
latexmk -pdf -interaction=nonstopmode main.tex
# čišćenje pomoćnih fajlova:
latexmk -c
```

Izlaz: `report/main.pdf`.

### Spisak rezervisanih mesta koja treba popuniti
**Brojevi / linkovi:**
- [ ] `Sažetak` i `Abstract` — tačnost, senzitivnost, specifičnost najboljeg modela.
- [ ] Tabela 3 (`tab:rezultati`) — svih 6 metrika, **dve kolone** (ResNet i ViT).
- [ ] Tabela `tab:poklasa` — rezultati po klasama (preciznost/odziv/F1).
- [ ] Tabela `tab:poredjenje` — redovi "ResNet-50" i "ViT-B/16".
- [ ] Odeljak `sec:jedanprimer` — verovatnoće klasa $P(\cdot)$ za izabrani primer.
- [ ] `[wandb-link-resnet]` i `[wandb-link-vit]` — URL-ovi wandb runova (odeljak Treniranje).
- [ ] `[github-link]` — URL GitHub repozitorijuma (odeljak Dostupnost koda).
- [ ] (opciono) `[Ime mentora]` na naslovnoj strani.

**Slike:**
- [ ] `fig:primeri` — `examples.png` (primeri iz dve klase).
- [ ] `fig:konfuzija` — `confusion_matrix.png`.
- [ ] `fig:krive-vit` — `vit_curves.png` (krive treniranja ViT).
- [ ] `fig:jedanprimer` — montaža međukoraka na jednom primeru (iz `gradcam_viz.py`).
- [ ] `fig:promasaj` — `misclassified.png`.
- [x] `fig:krive` — `learning_curves.png` (već postoji).
- [x] `fig:gradcam` — `gradcam_summary.png` (već postoji, samo ResNet).
