# 📘 README – Etapa 3: Analiza și Pregătirea Setului de Date pentru Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Iordache Robert Georgian
**Data:** 06.12.2025  

---

## Introducere

Acest document descrie activitățile realizate în Etapa 3, concentrându-se pe procesarea semnalelor EMG din baza de date NinaPro DB2. Scopul a fost transformarea datelor brute (serii de timp) într-un format compatibil cu arhitectura Deep Learning (ResNet 1D), aplicând tehnici avansate de ferestruire, normalizare și augmentare sintetică pentru a asigura robustețea modelului.

---

##  1. Structura Repository-ului Github (versiunea Etapei 3)

```
project-name/
├── README.md
├── README_Etapa3.md       # <-- Acest fișier
├── docs/
│   └── datasets/          # Diagrame distribuție clase, grafice semnal brut vs filtrat
├── data/
│   ├── raw/               # Fișierele originale .mat (S1_E2_A1.mat ... S14_E2_A1.mat)
│   ├── processed/         # Datele ferestruite și normalizate (în memorie/binar)
│   ├── train/             # Setul de antrenare (inclusiv date augmentate)
│   ├── validation/        # Setul de validare (stratificat)
│   └── test/              # Setul de testare (stratificat)
├── src/
│   ├── preprocessing/     # Scripturi pentru Windowing și Normalizare Z-Score
│   ├── data_acquisition/  # Scriptul de generare date sintetice (Augmentare)
│   └── neural_network/    # Definiția modelului ResNet (pregătire pentru Etapa 4)
├── config/                # Parametri (window_size=150, step=20)
└── requirements.txt       # tensorflow, scipy, sklearn, numpy
```

---

##  2. Descrierea Setului de Date

### 2.1 Sursa datelor

* **Origine:** NinaPro DB2 (Non-Invasive Adaptive Prosthetics Database), una dintre cele mai utilizate baze de date academice pentru proteze mioelectrice.
* **Modul de achiziție:**
[X] Senzori reali: Electrozi Delsys Trigno Wireless (frecvență eșantionare 2000 Hz).
[X] Generare programatică: Augmentare date (Zgomot Gaussian + Scalare).
* **Perioada / condițiile colectării:** Datele provin de la 14 subiecți sănătoși care execută mișcări repetitive ale mâinii și încheieturii (Exercițiul 2).

### 2.2 Caracteristicile dataset-ului

* **Număr total de observații:** ~163.841 ferestre reale + ~65.500 ferestre sintetice (Total > 229.000 instanțe).
* **Număr de caracteristici (features):** 12 (Canale EMG).
* **Tipuri de date:** [X] Numerice (Serii de timp) / [X] Categoriale (Etichete mișcare).
* **Format fișiere:** [X] .mat (Sursă) / [X] NumPy Arrays (Procesat).

### 2.3 Descrierea fiecărei caracteristici

| **Caracteristică** | **Tip** | **Unitate** | **Descriere** | **Domeniu valori** |
|-------------------|---------|-------------|---------------|--------------------|
| emg_ch[1-12] | numeric | μV (norm) | Semnal electric muscular (12 electrozi) normalizat Z-score. | ~ -3.0 ... +3.0 (după norm) |
| stimulus | categorial | - | Eticheta mișcării (Ground Truth). | 0–6 (după grupare) |
| window_time | temporal | m/s | Durata unei ferestre de analiză. | 150 ms (fereastră fixă) |
| subject_id | categorial | - | Identificatorul subiectului (S1-S14). | 1–14 |

**Fișier recomandat:**  `data/README.md`

---

##  3. Analiza Exploratorie a Datelor (EDA) – Sintetic

### 3.1 Statistici descriptive aplicate


* **Distribuții** S-a analizat histograma claselor originale (23 clase). S-a observat o distribuție inegală (mișcările de apucare durează mai mult decât cele de extensie).
* **Semnal** S-a calculat media și deviația standard per fereastră pentru a verifica calitatea contactului electrozilor.

### 3.2 Analiza calității datelor

* **Detectarea valorilor lipsă** Nu există valori NaN în fișierele .mat, dar există discrepanțe de lungime între vectorii emg și stimulus (eroare de 1 sample).
* **Zgomot** S-au identificat segmente de "Rest" (Repaus) care conțin zgomot de fond irelevant pentru clasificare.


### 3.3 Probleme identificate

* Problemă: Etichetele din fișierele E2 conțineau indici din setul E3 (18-40) în loc de standardul E2 (13-29).
Soluție: Remapare completă a dicționarului de etichete (label_map).
* Problemă: Confuzie ridicată între mișcările fine (ex: Pinch 2 fingers vs Pinch 3 fingers) pe 14 subiecți.
* Soluție: Implementarea strategiei de Grupare Funcțională (reducere de la 23 la 7 clase robuste).

---

##  4. Preprocesarea Datelor

### 4.1 Curățarea datelor

* **Filtrare:** Eliminarea eșantioanelor de repaus (stimulus=0) și a celor marcate ca pauză între repetiții.
* **Corecție:** Trunchierea vectorilor la lungimea minimă comună (min_len) pentru a evita erori de indexare.
* **Eliminare clase rare** Clasele cu mai puțin de 20 de ferestre per fișier au fost excluse.

### 4.2 Transformarea caracteristicilor

* **Windowing (Ferestruire):** 
Tehnică: Sliding Window.

Dimensiune: 150 samples (timp de reacție rapid).

Suprapunere (Step Size): 20 samples (pentru maximizarea datelor).
* **Mapping (Grupare):** Transformarea celor 23 de mișcări anatomice în 7 comenzi de control:
Wrist Flexion, Wrist Extension, Pronation, Supination, Power Grip, Hand Open, Precision Pinch.
* **Normalizare** Standardizare Z-Score aplicată individual pe fiecare fereastră

### 4.3 Structurarea seturilor de date

**Împărțire:**
70% Train: Folosit pentru optimizarea ponderilor. Include datele reale + 40% date sintetice.

15% Validation: Folosit pentru Early Stopping și ReduceLROnPlateau.

15% Test: Date complet neatinse, folosite doar pentru Matricea de Confuzie finală.

Principiu: train_test_split cu opțiunea stratify=y pentru a menține proporția fiecărei mișcări.

### 4.4 Salvarea rezultatelor preprocesării
Deoarece volumul de date este mare (~230.000 de ferestre x 12 canale), am optat pentru o procesare în memorie (In-Memory Processing) pentru a maximiza viteza de antrenare, evitând scrierea intermediară lentă pe disc (I/O Bottleneck).

Format date procesate: Datele finale sunt stocate în variabile de tip numpy.ndarray (float32) direct în memoria RAM (x_train_final, y_train_final), optimizate pentru ingestia în TensorFlow/Keras.

Normalizare: Nu este necesară salvarea unui fișier de tip "Scaler" (ex: .pkl), deoarece normalizarea este implementată dinamic, per fereastră (calcularea mediei și deviației standard se face independent pentru fiecare eșantion de 150ms). Aceasta asigură că modelul poate funcționa pe date noi fără a depinde de statistici globale pre-calculate.

Artefacte salvate: Rezultatul final al pipeline-ului de preprocesare și antrenare este salvat în format binar standard Keras:

models/model_proteza_final_resnet.keras – Modelul complet (arhitectură + ponderi).

models/model_proteza_final.tflite – Versiunea cuantizată pentru inferență rapidă în timp real (pentru modulul LabVIEW).

---

##  5. Fișiere Generate în Această Etapă
data/raw/*.mat – Fișierele originale NinaPro (S1-S14).
prelucrare_date.py – Scriptul principal care conține pipeline-ul complet (Load -> Preprocess -> Augment -> Train).
model_proteza_final_resnet.keras – Modelul antrenat rezultat.
docs/confusion_matrix.png – Dovada performanței pe setul de test.

---

##  6. Stare Etapă (de completat de student)

[x] Structură repository configurată
[x] Dataset analizat (Identificat probleme etichete și variabilitate)
[x] Date preprocesate (Windowing, Normalizare, Grupare 7 clase)
[x] Date augmentate (40% contribuție proprie)
[x] Seturi train/val/test generate (Stratified Split)
[x] Documentație actualizată în README + README_Etapa3.md

---
