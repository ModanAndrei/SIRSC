# README – Etapa 6: Analiza Performanței, Optimizarea și Concluzii Finale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** [Nume Prenume]  
**Link Repository GitHub:** [URL complet]  
**Data predării:** 2026-09-03

---

## Scopul Etapei 6

Această etapă corespunde punctelor **7. Analiza performanței și optimizarea parametrilor**, **8. Analiza și agregarea rezultatelor** și **9. Formularea concluziilor finale** din lista de 9 etape.

**Obiectiv principal:** Maturizarea completă a Sistemului cu Inteligență Artificială (SIA) prin optimizarea modelului RN, analiza detaliată a performanței și integrarea îmbunătățirilor în aplicația software completă.

---

## PREREQUISITE – Verificare Etapa 5 (OBLIGATORIU)

- [x] **Model antrenat** salvat în `models/trained_model.pt`
- [x] **Metrici baseline** raportate: Accuracy = 94.73%, F1-score = 0.9216
- [x] **Tabel hiperparametri** cu justificări completat în Etapa 5
- [x] **`results/training_history.csv`** cu toate epoch-urile
- [x] **UI funcțional** care încarcă modelul antrenat și face inferență reală
- [x] **Screenshot inferență** în `docs/screenshots/inference_real.png`
- [x] **State Machine** implementat conform definiției din Etapa 4

---

## Cerințe

1. **Minimum 4 experimente de optimizare** - Realizate (vezi tabel)
2. **Tabel comparativ experimente** - Completat
3. **Confusion Matrix** - Generată în `docs/confusion_matrix_optimized.png`
4. **Analiza detaliată a 5 exemple greșite** - Completată
5. **Metrici finali pe test set:**
   - **Acuratețe: 97.21%** (îmbunătățire față de baseline)
   - **F1-score (macro): 0.9534**
6. **Salvare model optimizat** în `models/optimized_model.pt`
7. **Actualizare aplicație software:**
   - UI încarcă modelul OPTIMIZAT
   - Screenshot demonstrativ în `docs/screenshots/inference_optimized.png`
8. **Concluzii tehnice** - Documentate mai jos

#### Tabel Experimente de Optimizare

| **Exp#** | **Modificare față de Baseline (Etapa 5)** | **Accuracy** | **F1-score** | **Timp antrenare** | **Observații** |
|----------|------------------------------------------|--------------|--------------|-------------------|----------------|
| Baseline | Configurația din Etapa 5 (LR 0.001, Batch 16) | 0.9473 | 0.9216 | 45 min | Referință YOLOv11n |
| Exp 1 | Learning rate 0.001 → 0.0005 | 0.9512 | 0.9284 | 48 min | Convergență mai stabilă |
| Exp 2 | Batch size 16 → 32 | 0.9385 | 0.9105 | 38 min | Mai rapid, dar scădere ușoară |
| Exp 3 | Imgsz 96 → 128 | 0.9654 | 0.9412 | 75 min | Detecție mai bună a detaliilor |
| Exp 4 | Augmentări aditițonale (Blur & Noise) | 0.9721 | 0.9534 | 80 min | **BEST** - Optimizat industrial |

**Justificare alegere configurație finală:**
```text
Am ales Exp 4 ca model final deoarece:
1. Oferă cel mai bun F1-score (0.9534), esențial pentru siguranța rutieră.
2. Augmentările de tip zgomot și blur simulează condiții meteorologice adverse și vibrații ale camerei, făcând sistemul mai robust.
3. Creșterea rezoluției la 128px a ajutat semnificativ la distingerea cifrelor pe semnele de limitare a vitezei.
```

---

## 1. Actualizarea Aplicației Software în Etapa 6 

### Tabel Modificări Aplicație Software

| **Componenta** | **Stare Etapa 5** | **Modificare Etapa 6** | **Justificare** |
|----------------|-------------------|------------------------|-----------------|
| **Model încărcat** | `trained_model.pt` | `optimized_model.pt` | +2.48% accuracy, robustețe sporită |
| **Threshold alertă** | 0.25 (default) | 0.35 | Reducerea detecțiilor false (FP) |
| **Stare nouă State Machine** | N/A | `CONFIDENCE_CHECK` | Diferențiere între decizii automate și review uman |
| **UI - afișare confidence** | Metric simplu | Metric cu delta (Confidence High/Low) | Feedback vizual îmbunătățit pentru operator |
| **Logging** | Minimal | Avertizări în UI pentru review uman | Creșterea siguranței în operare |

### Modificări concrete aduse în Etapa 6:

1. **Model înlocuit:** `models/trained_model.pt` → `models/optimized_model.pt`
   - Îmbunătățire: Accuracy +2.48%, F1 +3.18%
   - Motivație: Modelul optimizat este mai rezistent la zgomot industrial.

2. **State Machine actualizat:**
   - Pragul de încredere a fost ridicat la 0.35 pentru a filtra zgomotul.
   - A fost introdusă starea `CONFIDENCE_CHECK` care marchează predicțiile sub 60% ca având nevoie de intervenție umană.

3. **UI îmbunătățit:**
   - Adăugarea indicatoarelor de încredere ("High Confidence" / "Low Confidence").
   - Avertismente textuale pentru semnele detectate cu scor mic.
   - Screenshot: `docs/screenshots/inference_optimized.png`

---

## 2. Analiza Detaliată a Performanței

### 2.1 Confusion Matrix și Interpretare

**Locație:** `docs/confusion_matrix_optimized.png`

### Interpretare Confusion Matrix:

**Clasa cu cea mai bună performanță:** Semnul "STOP" (ClassId 14)
- Precision: 99%
- Recall: 98%
- Explicație: Formă octogonală unică și contrast ridicat (roșu/alb).

**Clasa cu cea mai slabă performanță:** Limită 80 km/h (ClassId 05)
- Precision: 91%
- Recall: 89%
- Explicație: Confuzie frecventă cu limitele de 30 și 50 km/h din cauza similitudinii cifrelor la distanță.

**Confuzii principale:**
1. Clasa "Limită 80" confundată cu "Limită 30" în 5% din cazuri.
   - Cauză: Cifra '8' și '3' au trăsături curbe similare.
   - Impact industrial: Risc de nerespectare a vitezei legale.
   
2. Semnele de obligație (34, 36) confundate între ele.
   - Cauză: Ambele au fond albastru circular și săgeți albe, diferind doar prin orientare.

### 2.2 Analiza Detaliată a 5 Exemple Greșite

| **Index** | **True Label** | **Predicted** | **Confidence** | **Cauză probabilă** | **Soluție propusă** |
|-----------|----------------|---------------|----------------|---------------------|---------------------|
| #45 | Limita 80 | Limita 30 | 0.52 | Imagine pixelată | Creștere imgsz la 256px |
| #112 | Obligatoriu Dreapta | Obligatoriu Inainte | 0.48 | Unghi de vizualizare oblic | Augmentare perspective |
| #205 | Prioritate | Drum cu prioritate | 0.61 | Expunere solară excesivă | Augmentare brightness/contrast |
| #330 | Acces Interzis | STOP | 0.45 | Ocluzie parțială (crengi) | Augmentare Random Erasing |
| #410 | Limita 20 | Normal (Background) | 0.38 | Semn de dimensiuni mici | Utilizare modele cu input multiscale |

---

## 3. Optimizarea Parametrilor și Experimentare

### 3.1 Strategia de Optimizare

**Abordare:** Căutare manuală (Manual Search) ghidată de rezultatele inițiale.

**Axe de optimizare explorate:**
1. **Rezoluție:** Testarea impactului măririi imaginii de la 96 la 128px.
2. **Learning rate:** Ajustare pentru a evita minimele locale.
3. **Batch size:** Testarea stabilității gradientului la 16 vs 32.
4. **Augmentări:** Introducerea Gaussian Blur și Zgomot pentru robustețe industrială.

**Criteriu de selecție model final:** F1-score maxim cu menținerea latenței sub 50ms pe CPU.

---

## 4. Agregarea Rezultatelor și Vizualizări

### 4.1 Tabel Sumar Rezultate Finale

| **Metrică** | **Etapa 4** | **Etapa 5** | **Etapa 6** | **Target Industrial** | **Status** |
|-------------|-------------|-------------|-------------|----------------------|------------|
| Accuracy | ~5% | 94.7% | 97.2% | ≥85% | OK |
| F1-score (macro) | ~0.02 | 0.92 | 0.95 | ≥0.80 | OK |
| Precision (macro) | N/A | 0.93 | 0.95 | ≥0.85 | OK |
| Recall (macro) | N/A | 0.93 | 0.94 | ≥0.90 | OK |
| Latență inferență | 100ms | 38ms | 42ms | ≤50ms | OK |

### 4.2 Vizualizări Obligatorii

Salvate în `docs/results/` și `docs/optimization/`:

- [x] `docs/confusion_matrix_optimized.png` - Confusion matrix model final
- [x] `docs/results/learning_curves_final.png` - Loss și accuracy vs. epochs
- [x] `docs/results/metrics_evolution.png` - Evoluție metrici Etapa 4 → 5 → 6
- [x] `docs/optimization/accuracy_comparison.png` - Grafic comparativ acuratețe

---

## 5. Concluzii Finale și Lecții Învățate

### 5.1 Evaluarea Performanței Finale

**Obiective atinse:**
- [x] Model YOLO v11 funcțional cu acuratețe 97.2% pe test set.
- [x] Integrare completă în aplicația Streamlit.
- [x] State Machine actualizat cu verificare de încredere (Confidence Check).
- [x] Performanță în timp real pe CPU (42ms/imagine).

### 5.2 Limitări Identificate

1. **Limitări date:** Dataset-ul GTSRB este colectat în condiții diurne; performanța pe timp de noapte este necunoscută.
2. **Limitări model:** Dificultăți în distingerea cifrelor fine la rezoluții foarte mici (<64px).
3. **Limitări infrastructură:** Dependența de CPU pentru inferență limitează throughput-ul la ~24 FPS.

### 5.3 Direcții de Cercetare și Dezvoltare

**Pe termen scurt:**
1. Colectarea de date în condiții de noapte și ploaie.
2. Exportul modelului în format OpenVINO pentru accelerare pe procesoare Intel.

### 5.4 Lecții Învățate

1. **Rezoluția contează:** Trecerea de la 96 la 128px a adus cel mai mare salt de performanță pentru clasele de viteze.
2. **Augmentarea specifică:** Simularea zgomotului senzorului prin augmentare a îmbunătățit generalizarea pe date reale, necurățate.
3. **Importanța F1-score:** Acuratețea poate fi înșelătoare dacă clasele sunt ușor dezechilibrate; F1-score oferă o imagine mai clară asupra siguranței sistemului.

### 5.5 Plan Post-Feedback

1. Implementarea unui sistem de logging automat al predicțiilor incerte.
2. Refactorizarea codului de preprocesare pentru a suporta multithreading.
3. Documentarea API-ului pentru integrare cu alte sisteme ADAS.

---

## Structura Repository-ului la Finalul Etapei 6

```
proiect-rn-trafficsign/
├── README.md                               # Overview general proiect (FINAL)
├── etapa3_analiza_date.md                  # Din Etapa 3
├── etapa4_arhitectura_sia.md               # Din Etapa 4
├── etapa5_antrenare_model.md               # Din Etapa 5
├── docs/
│   ├── etapa6_optimizare_concluzii.md          # ACEST FIȘIER
│   ├── state_machine.svg                   # Din Etapa 4
│   ├── confusion_matrix_optimized.png      # OBLIGATORIU
│   ├── results/
│   │   ├── metrics_evolution.png           # Evoluție Etapa 4→5→6
│   │   ├── learning_curves_final.png       # Model optimizat
│   │   └── example_predictions.png         # Grid exemple
│   ├── optimization/
│   │   ├── accuracy_comparison.png
│   │   └── f1_comparison.png
│   └── screenshots/
│       ├── inference_real.png              # Din Etapa 5
│       └── inference_optimized.png         # OBLIGATORIU
├── models/
│   ├── trained_model.pt                    # Din Etapa 5
│   └── optimized_model.pt                  # OBLIGATORIU
├── results/
│   ├── training_history.csv                # Din Etapa 5
│   ├── optimization_experiments.csv        # OBLIGATORIU
│   ├── final_metrics.json                  # Metrici model optimizat
├── src/
│   ├── app/
│   │   └── app.py                         # ACTUALIZAT
│   └── neural_network/
│       ├── optimize.py                     # Script experimente
│       └── visualize_stage6.py            # Script vizualizari
└── requirements.txt
```
