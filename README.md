# Detectarea semnelor rutiere cu YOLO v11 (GTSRB)

Acesta este repository-ul oficial pentru proiectul de Rețele Neuronale.
**Student:** Modan Ionuț Andrei (Grupa 634AB)

### 🚀 Status Proiect
- **Livrabil Final:** [Modan_IonutAndrei_634AB_README_Proiect_RN.md](./Modan_IonutAndrei_634AB_README_Proiect_RN.md)
- **Tag:** `v0.6-optimized-final`
- **Performanță:** Accuracy **97.21%**, F1-Score **0.95**

### 📂 Structură
- `src/`: Codul sursă (Data Acquisition, RN, UI)
- `docs/`: Documentația etapelor 3-6
- `models/`: Modelele antrenate și optimizate
- `results/`: Metrici și rezultate experimente
- `data/`: Dataset (link-uri către raw data și procesate)

### 🛠️ Instalare și Rulare rapidă
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run src/app/app.py
```

### 📋 Documentație pe etape
1. [Etapa 3 - Analiza datelor](./docs/etapa3_analiza_date.md)
2. [Etapa 5 - Antrenare model](./docs/etapa5_antrenare_model.md)
3. [Etapa 6 - Optimizare și Concluzii](./docs/etapa6_optimizare_concluzii.md)

