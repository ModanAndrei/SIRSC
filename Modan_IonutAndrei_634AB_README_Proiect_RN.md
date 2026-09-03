# Proiect Rețele Neuronale - Recunoașterea Semnelor Rutiere (GTSRB)

**Student:** Modan Ionuț Andrei  
**Grupa:** 634AB  
**Disciplina:** Rețele Neuronale (RN)  
**Instituție:** Universitatea Politehnica București - FIIR  
**Link Repository:** https://github.com/ModanAndrei/SIRSC.git  
**Tag Versiune Finală:** `v0.6-optimized-final`

---

## 1. Declarația de Originalitate și Politica de Utilizare AI

### Declarație de Originalitate
Subsemnatul Modan Ionuț Andrei, declar pe propria răspundere că acest proiect este rezultatul muncii mele proprii, realizat în cadrul disciplinei Rețele Neuronale. Toate sursele externe utilizate au fost citate corespunzător.

### Politica de Utilizare AI
În realizarea acestui proiect:
- [x] Am utilizat instrumente AI (ex: ChatGPT, Claude, Junie) pentru asistență la scrierea codului de vizualizare și a documentației.
- [x] Am verificat și validat manual orice fragment de cod generat de AI.
- [x] Am conceput personal arhitectura sistemului și am condus experimentele de optimizare.
- **Contribuție Date Originale:** Peste 40% din datele utilizate în antrenare/validare provin din augmentări specifice (Gaussian Blur, Noise, Perspective Transform) generate manual pentru a simula condiții industriale reale.

---

## 2. Rezumat Performanță SIA (Sistem cu Inteligență Artificială)

La finalul Etapei 6, sistemul a atins următoarele performanțe pe setul de test (GTSRB filtrat pe 15 clase):

| Metrică | Valoare Obținută | Prag Minim Cerut | Status |
|---------|------------------|------------------|--------|
| **Accuracy** | **97.21%** | ≥ 70% | ✅ OK |
| **F1-Score (macro)** | **0.9534** | ≥ 0.65 | ✅ OK |
| **Latență Inferență** | **42ms / imagine** | ≤ 100ms | ✅ OK |

---

## 3. Module Funcționale ale Proiectului

Proiectul este compus din trei module principale integrate conform specificațiilor:

1.  **Modulul Data Logging & Acquisition (`src/data_acquisition`):**
    - Aici m-am ocupat de tot ce ține de dataset-ul GTSRB: de la transformarea formatelor (PPM -> JPG), la generarea de noi imagini prin augmentări pentru a face modelul mai robust.
    - Am folosit un split stratificat de 80/10/10 pentru a fi sigur că antrenarea e corectă și că nu există suprapuneri între seturi.

2.  **Modulul RN (Rețeaua Neuronală) (`src/neural_network`):**
    - Am ales arhitectura YOLO v11 de la Ultralytics.
    - Am integrat scripturi pentru tot fluxul de lucru: antrenarea de bază, evaluarea pe test set și optimizarea finală a hiperparametrilor.
    - Modelele finale (baseline și cel optimizat) sunt salvate în folderul `models/`.

3.  **Modulul Web Service / UI (`src/app`):**
    - Am făcut o interfață cu Streamlit ca să pot testa vizual performanța modelului.
    - Poți încărca orice imagine cu semne rutiere și vezi imediat ce a detectat rețeaua.
    - Am implementat și logica de State Machine: sistemul trece prin starea de inferență (`RN_INFERENCE`) și apoi verifică dacă e destul de sigur pe rezultat (`CONFIDENCE_CHECK`).

---

## 4. Analiza Performanței și Optimizări

### Experimente de Optimizare (Min. 4)
| Exp# | Modificare | Accuracy | F1-Score | Observații |
|------|------------|----------|----------|------------|
| Baseline | LR 0.001, Batch 16, Imgsz 96 | 94.73% | 0.9216 | Configurația inițială |
| Exp 1 | LR 0.001 -> 0.0005 | 95.12% | 0.9284 | Convergență mai stabilă |
| Exp 2 | Batch size 16 -> 32 | 93.85% | 0.9105 | Viteză mai mare, precizie scăzută |
| Exp 3 | Imgsz 96 -> 128 | 96.54% | 0.9412 | Detecție net îmbunătățită pentru cifre |
| **Exp 4** | **Augmentări (Blur & Noise)** | **97.21%** | **0.9534** | **Modelul Final Optimizat** |

### Analiza Erorilor
- **Confusion Matrix:** Disponibilă în `docs/confusion_matrix_optimized.png`.
- **Exemple Greșite:** Analiza a 5 cazuri critice (ex: confuzia între limită de 30 și 80 km/h) este documentată în `docs/etapa6_optimizare_concluzii.md`.
- **Stare State Machine:** A fost adăugată starea `CONFIDENCE_CHECK` pentru a alerta operatorul uman la predicții cu încredere < 60%.

---

## 5. Demonstrație End-to-End

O demonstrație completă a funcționării sistemului (input -> RN -> UI -> Decision) se găsește în:
👉 **`docs/demo/`** (conține screenshot-uri ale pașilor de inferență și interfeței).

---

## 6. Structura Repository-ului

Conform cerinței Etapei 6, structura proiectului este:

```text
TrafficSignRecognition/
├── Modan_IonutAndrei_634AB_README_Proiect_RN.md  <-- ACEST FIȘIER
├── README.md                                     <-- Ghid rapid
├── config/                                       <-- Clase și config YOLO
├── data/                                         <-- Seturi de date (raw, processed)
├── docs/                                         <-- Documentația etapelor 3-6
│   ├── etapa3_analiza_date.md
│   ├── etapa5_antrenare_model.md
│   ├── etapa6_optimizare_concluzii.md
│   ├── demo/                                     <-- Demo funcțional
│   └── screenshots/                              <-- Vizualizări metrici
├── models/                                       <-- Modele .pt (trained, optimized)
├── results/                                      <-- CSV-uri training, experimente
├── src/                                          <-- Codul sursă
│   ├── app/                                      <-- Modul UI (Streamlit)
│   ├── data_acquisition/                         <-- Modul Logging/Data
│   └── neural_network/                           <-- Modul Antrenare/Optimizare
└── requirements.txt
```
