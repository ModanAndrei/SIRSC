# Etapa 4: Arhitectura completa a aplicatiei SIA

**Disciplina:** Retele neuronale  
**Institutie:** POLITEHNICA Bucuresti - FIIR  
**Proiect:** Detectarea semnelor rutiere cu YOLO v11 si Python  

## 1. Nevoie reala -> solutie SIA -> modul software

| Nevoie reala concreta | Cum o rezolva SIA-ul | Modul software responsabil |
|---|---|---|
| Identificarea rapida a semnelor rutiere pentru asistenta soferului | Detectare cu bounding box si denumire in romana, cu tinta de inferenta sub 200 ms pe GPU | RN YOLO v11 + Web UI |
| Verificarea automata a imaginilor cu semne din inspectii de drum | Upload imagine si raport vizual cu toate detectiile peste pragul de incredere | Data Acquisition + RN + UI |
| Testarea sistemului in conditii variabile de iluminare si perspectiva | Date sintetice controlate, minimum 6.120 observatii originale si acoperire a 15 clase | Data Acquisition + Preprocessing |

## 2. Contributia originala la setul de date

**Total observatii finale:** 15.300 (9.180 publice + 6.120 generate)  
**Observatii originale:** 6.120 (40%)  

**Tipul contributiei:**

- [x] Date generate prin simulare fizica simplificata a camerei
- [ ] Date achizitionate cu senzori proprii
- [ ] Etichetare/adnotare manuala
- [x] Date sintetice parametrizate cu variatie de perspectiva si iluminare

Generatorul din `src/data_acquisition/generate.py` produce semne parametrice, nu duplicate GTSRB: fundal de drum, geometrie de semn, pozitie si scara variabile, rotatie de +/-12 grade si iluminare 0.72-1.22. ROI-ul este scris simultan cu geometria randata, iar fiecare observatie are `source=synthetic_physical_camera`.

Cele 6.120 imagini sunt distribuite egal, 408 pentru fiecare din cele 15 clase si sunt adaugate in `data/processed/train` cu etichete YOLO. Parametrii sunt controlabili prin seed `2026`; manifestul CSV permite reproducerea si auditarea fiecarei observatii.

**Locatii:** cod `src/data_acquisition/generate.py`, date `data/generated/`, manifest `data/generated/annotations.csv`.  
**Dovezi:** [grafic comparativ](docs/generated_vs_real.svg), [tabel statistici](docs/data_statistics.csv), manifest CSV.

## 3. State Machine

Diagrama este in [docs/state_machine.svg](docs/state_machine.svg). Am ales o arhitectura de clasificare/detectare interactiva, deoarece sistemul primeste fie o imagine, fie un frame webcam, il valideaza si intoarce imediat bounding box-ul si denumirea semnului.

Stari principale: `IDLE` asteapta sursa, `ACQUIRE_DATA` primeste upload/frame, `PREPROCESS` converteste RGB si pregateste intrarea, `INFERENCE` ruleaza YOLO v11, `DISPLAY / LOG` afiseaza cutiile si increderea, iar `WAIT / LOOP` permite urmatorul frame. Orice input invalid sau model indisponibil duce la `ERROR`, nu la un rezultat ascuns.

Tranzitia critica este `ACQUIRE_DATA -> PREPROCESS` cand imaginea poate fi deschisa, iar `INFERENCE -> ERROR` cand modelul nu poate fi incarcat. Bucla `WAIT / LOOP -> ACQUIRE_DATA` sustine detectia repetata din webcam sau procesarea mai multor imagini.

## 4. Modulele SIA

1. **Data Logging / Acquisition:** `src/data_acquisition/generate.py` produce CSV, imagini originale si etichete YOLO.
2. **Neural Network:** `src/neural_network/model.py` defineste un CNN cu 15 iesiri, face forward pass si salveaza/reincarca weights neantrenate.
3. **Web Service / UI:** `src/app/app.py` primeste upload si flux video webcam live si afiseaza rezultatul YOLO.

Screenshot demonstrativ: [docs/screenshots/ui_demo.png](docs/screenshots/ui_demo.png).

Comenzi de verificare (din mediul virtual al proiectului):

```powershell
.venv\Scripts\python.exe src\data_acquisition\generate.py
.venv\Scripts\python.exe src\neural_network\model.py
.venv\Scripts\python.exe -m streamlit run src\app\app.py
```

Modelul este intentionat neantrenat in aceasta etapa; antrenarea serioasa se face ulterior cu `.venv\Scripts\python.exe train.py`.

## Checklist

- [x] Tabel nevoie-solutie-modul cu metrici
- [x] Contributie originala de 40% documentata si masurata
- [x] Generator CSV cu minimum 100 samples si parametri documentati
- [x] State machine cu 7 stari si tratare ERROR
- [x] Modul RN definit, compilabil, salvabil si reincarcabil
- [x] UI cu upload si webcam
- [x] Structura `data/generated`, `src/data_acquisition`, `src/neural_network`, `src/app`, `models`, `docs`