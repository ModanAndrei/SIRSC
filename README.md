##  1. Structura Repository-ului Github (versiunea Etapei 3)

```
# SIRSC
SIRSC/
├── README.md
├── docs/
│   └── datasets/          # descriere seturi de date, surse, diagrame
├── data/
│   ├── raw/               # date brute
│   ├── processed/         # date curățate și transformate
│   ├── train/             # set de instruire
│   ├── validation/        # set de validare
│   └── test/              # set de testare
├── src/
│   ├── preprocessing/     # funcții pentru preprocesare
│   ├── data_acquisition/  # generare / achiziție date (dacă există)
│   └── neural_network/    # implementarea RN (în etapa următoare)
├── config/                # fișiere de configurare
└── requirements.txt       # dependențe Python (dacă aplicabil)

# Descrierea Setului de Date

## Sursa datelor
Am utilizat datasetul **GTSRB – German Traffic Sign Recognition Benchmark**, un set de date public ce conține imagini reale cu semne de circulație din Germania. Este unul dintre cele mai folosite seturi în proiecte de clasificare vizuală.

Link: https://benchmark.ini.rub.de/

## Structura datelor
- imagini color (RGB)
- dimensiuni variabile, care vor fi redimensionate la 48x48
- fiecare imagine aparține unei clase (0–42)

## Folosire în proiect
- imaginile brute sunt plasate în `data/raw/`
- după preprocesare, vor fi salvate în `data/processed/`
- împărțirea în `train/validation/test` va fi realizată ulterior.

