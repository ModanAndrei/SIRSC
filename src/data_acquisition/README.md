# Modul 1: Data Logging / Acquisition

`generate.py` produce 6.120 observații originale, câte 408 pentru fiecare dintre cele 15 clase. Simularea folosește un fundal de drum, forme geometrice de semne, poziție și scară variabile, rotație de +/-12 grade și iluminare 0.72-1.22. Manifestul CSV include ROI, clasă, dimensiuni și sursa observației.

```powershell
.venv\Scripts\python.exe src\data_acquisition\generate.py
```

Datele sunt salvate în `data/generated/` și sunt distincte de cadrele publice GTSRB.
