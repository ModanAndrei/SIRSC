# 📘 README – Etapa 5: Configurarea și Antrenarea Modelului RN

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** [Nume Prenume]  
**Link Repository GitHub:** [URL complet]  
**Data predării:** 2026-09-03

---

## Scopul Etapei 5

Această etapă corespunde punctului **6. Configurarea și antrenarea modelului RN** din lista de 9 etape - slide 2 **RN Specificatii proiect.pdf**.

**Obiectiv principal:** Antrenarea efectivă a modelului RN definit în Etapa 4, evaluarea performanței și integrarea în aplicația completă.

---

## PREREQUISITE – Verificare Etapa 4 (OBLIGATORIU)

- [x] **State Machine** definit și documentat în `docs/state_machine.*`
- [x] **Contribuție ≥40% date originale** în `data/generated/` (verificabil)
- [x] **Modul 1 (Data Logging)** funcțional - produce CSV-uri / formate YOLO
- [x] **Modul 2 (RN)** cu arhitectură YOLO v11 definită
- [x] **Modul 3 (UI/Web Service)** funcțional cu model antrenat

---

## Pregătire Date pentru Antrenare 

Datasetul a fost combinat folosind setul original GTSRB și datele sintetice/augmentate generate pentru a asigura o distribuție echilibrată a celor 15 clase selectate.

- **Split stratificat:** 80% train / 10% validation / 10% test (optimizat pentru YOLO)
- **Total imagini:** ~15.300 imagini în setul combinat.
- **Preprocesare:** Conversie în format YOLO, redimensionare și normalizare.

---

## Nivel 1 – Obligatoriu pentru Toți (70% din punctaj)

1. **Antrenare model:** YOLO v11 nano antrenat de la zero (pretrained=false).
2. **Epoci:** 50 epoci planificate (20 epoci în ultima rulare de fine-tuning).
3. **Metrici calculate pe test set:**
   - **Acuratețe:** 94.73%
   - **F1-score (macro):** 0.9216
4. **Salvare model antrenat:** `models/trained_model.pt`
5. **Integrare în UI:** Interfața Streamlit (`src/app/app.py`) încarcă modelul antrenat pentru inferență reală.

#### Tabel Hiperparametri și Justificări (OBLIGATORIU - Nivel 1)

| **Hiperparametru** | **Valoare Aleasă** | **Justificare** |
|--------------------|-------------------|-----------------|
| Learning rate | 0.001 | Valoare standard pentru AdamW, asigură convergență stabilă. |
| Batch size | 16 | Echilibru între stabilitatea gradientului și limitările de memorie pe CPU. |
| Number of epochs | 50 | Suficient pentru a permite modelului să învețe caracteristicile de bază de la zero. |
| Optimizer | AdamW | Optimizator robust cu weight decay pentru a preveni overfitting-ul. |
| Loss function | YOLO Loss (Box, Cls, DFL) | Optimizat pentru detecția obiectelor și localizarea precisă. |
| Image size | 96 | Compromis între viteza de procesare pe CPU și păstrarea detaliilor semnelor mici. |

---

## Nivel 2 – Recomandat (85-90% din punctaj)

1. **Early Stopping:** Patience 0 (rulat până la capăt pentru a maximiza învățarea pe CPU).
2. **Learning Rate Scheduler:** Cosine scheduler utilizat pentru o scădere lină a LR.
3. **Augmentări relevante:** Perspective, HSV, mosaic, mixup și blur pentru a simula condiții industriale/rutiere variate.
4. **Grafic loss:** Salvat în `docs/loss_curve.png` (extras din `runs/results.png`).
5. **Analiză erori context industrial:**

### 1. Pe ce clase greșește cel mai mult modelul?
Deși acuratețea este ridicată, confuziile apar în principal între clasele cu forme geometrice identice și culori similare, cum ar fi semnele de obligație (fond albastru, săgeți albe: 34, 36, 37). De asemenea, limita de 20 km/h și 80 km/h pot fi confundate la rezoluții mici din cauza formei cifrelor.

### 2. Ce caracteristici ale datelor cauzează erori?
Condițiile de iluminare extreme (simulate prin HSV augmentation) și perspectiva (unghiuri oblice) sunt factorii principali. De asemenea, redimensionarea la 96px poate face ca detaliile fine ale cifrelor din limitele de viteză să devină neclare.

### 3. Ce implicații are pentru aplicația industrială?
Într-un sistem de asistență la conducere (ADAS), un **False Negative** la semnul "STOP" sau "Acces Interzis" este critic. Modelul actual prioritizează o rechemare (recall) ridicată pentru a evita ratarea semnelor critice, chiar dacă introduce mici erori de clasificare între limitele de viteză.

### 4. Ce măsuri corective propuneți?
1. Creșterea rezoluției de intrare la 320px pentru a distinge mai bine cifrele.
2. Utilizarea Transfer Learning pentru a beneficia de caracteristici pre-antrenate pe volume mari de date.
3. Echilibrarea dataset-ului pentru clasele minoritare (ex. limita de 20 km/h).

---

## Nivel 3 – Bonus (până la 100%)

- **Confusion Matrix:** Generată automat și salvată în `docs/confusion_matrix.png`.
- **Benchmark latență:** Inferență sub 100ms pe CPU pentru imagini de 96px.

---

## Verificare Consistență cu State Machine (Etapa 4)

| **Stare din Etapa 4** | **Implementare în Etapa 5** |
|-----------------------|-----------------------------|
| `ACQUIRE_DATA` | Încărcare imagine prin Streamlit sau citire din setul de test. |
| `PREPROCESS` | Redimensionare la 96x96 și normalizare (standard YOLO). |
| `RN_INFERENCE` | Rulare model `trained_model.pt` folosind Ultralytics. |
| `THRESHOLD_CHECK` | Verificare confidence threshold (0.25) pentru afișarea detecțiilor. |
| `ALERT` | Afișare bounding box și clasă în interfața UI. |

---

## Structura Repository-ului la Finalul Etapei 5

```
proiect-rn-trafficsign/
├── README.md                           # Overview general proiect
├── docs/
│   ├── etapa5_antrenare_model.md      # ACEST FIȘIER
│   ├── loss_curve.png                 # Grafic antrenare
│   ├── confusion_matrix.png           # Matricea de confuzie
│   └── screenshots/
│       └── inference_real.png         # Screenshot UI
├── models/
│   ├── untrained_model.pt             # Arhitectura inițială
│   └── trained_model.pt               # Modelul antrenat final
├── results/
│   ├── training_history.csv           # Istoric epoci
│   ├── test_metrics.json              # Metrici finale test
│   └── hyperparameters.yaml           # Configurația antrenării
└── src/
    ├── app/
    │   └── app.py                     # UI actualizat
    └── neural_network/
        ├── train.py                   # Script antrenare
        └── evaluate.py                # Script evaluare
```
