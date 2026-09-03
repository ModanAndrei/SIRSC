Aici gestionăm datele proiectului.

Folosim dataset-ul **GTSRB** (German Traffic Sign Recognition Benchmark). 
Ce trebuie să știi:
- Avem imagini cu semne rutiere și fișiere CSV cu coordonatele lor (ROI).
- Din cele 43 de clase originale, am ales **15 clase** pe care să lucrăm (folosind un seed fix `2026` pentru repetabilitate).
- Am convertit imaginile din formatul lor ciudat (.ppm) în .jpg și le-am pregătit pentru YOLO.
- Împărțirea datelor (Train/Val/Test) am făcut-o pe "track-uri", adică am avut grijă ca poze diferite ale aceluiași semn fizic să nu fie amestecate între seturi (ca să nu "trișeze" rețeaua).

Dacă vrei să vezi statistici despre date, poți rula:
```powershell
python src/preprocessing/analyze_dataset.py
```
