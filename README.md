# SIRSC
SIRSC/
├── README.md
├── data/
│   ├── raw/
│   ├── processed/
│   ├── train/
│   ├── validation/
│   └── test/
├── docs/
│   └── datasets/
├── src/
│   ├── preprocessing/
│   └── neural_network/
├── config/
└── requirements.txt

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

