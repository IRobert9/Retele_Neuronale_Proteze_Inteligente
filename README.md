# Retele_Neuronale_Proteze_Inteligente
Proiect desfasurat in cadrul materiei Retele Neuronale

# 📘 README – Etapa 3: Analiza și Pregătirea Setului de Date pentru Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Iordache Robert Georgian  
**Data:** 21.11.2025 

---

## Introducere

Acest document descrie activitățile realizate în **Etapa 3**, în care se analizează și se preprocesează setul de date necesar proiectului „Rețele Neuronale". Scopul etapei este pregătirea corectă a datelor pentru instruirea modelului RN, respectând bunele practici privind calitatea, consistența și reproductibilitatea datelor.

---

##  1. Structura Repository-ului Github (versiunea Etapei 3)

```
project-name/
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
```

---

##  2. Descrierea Setului de Date

### 2.1 Sursa datelor

* **Origine:** 
Dataset public - NINAPRO Database 2, unul dintre cele mai cunoscute seturi publice de date EMG din domeniul cercetării pentru controlul protezelor mioelectrice. Datasetul este disponibil pentru uz academic.

* **Modul de achiziție:** 
Senzori EMG folosiți:
12 electrozi EMG de suprafață, așezați circumferențial pe antebraț
rata de eșantionare: 2.000 Hz
rezoluție ridicată pentru captarea semnalelor musculare

Configurația hardware:
sistem Ottobock 13E200 EMG
12 canale electromiografice sincronizate
semnal analogic amplificat și digitalizat

Procedura de achiziție:
Participanților li s-a cerut să execute repetat un set de mișcări standardizate:
flexie încheietură
extensie încheietură
închidere palmă (grasp)
deschidere palmă
pronație/supinație
relaxare (activitate minimă EMG)

În timpul fiecărei mișcări:
participantul urmează un video cu ritm fix (pentru a menține uniformitatea)
fiecare gest este repetat de 6–10 ori
durata unei mișcări este de 5–7 secunde

Structurarea datelor în dataset:
semnalele sunt împărțite pe fișiere .mat
pentru fiecare mișcare există etichete numerice
datele sunt sincronizate între canale
semnalele brute sunt puse la dispoziție pentru procesare

* **Perioada / condițiile colectării:**
Datele au fost colectate în perioada 2012–2014

subiecții sănătoși cu mobilitate normală
efectuarea mișcărilor în poziție șezând pe scaun
temperatura camerei controlată 21–24 °C
repaus muscular între secvențe pentru evitarea oboselii
electrozi aplicați pe pielea curată, degresată cu alcool
mișcările sunt efectuate conform unui protocol video standardizat
toți participanții au repetat aceleași mișcări în aceleași condiții
### 2.2 Caracteristicile dataset-ului

* **Număr total de observații:** 19.600 instanțe complete de date asociată cu o etichetă (o mișcare).
Se foloseste una dintre bazele de date din NinaPro (DB2) unde se regasesc:
40 subiecți sănătoși intacți * 49 miscari ale mainii * 10 repetari

* **Număr de caracteristici (features):** 5
* **Tipuri de date:**  Subiect (ID subiect) / 12 canale EMG / Stimulus (Eticehta) / Restimulus (perioade de repaus/activ) / Repetition (repetari)
* **Format fișiere:** .mat (Mathlab)

### 2.3 Descrierea fiecărei caracteristici

| **Caracteristică** | **Tip** | **Unitate** | **Descriere** | **Domeniu valori** |
|-------------------|---------|-------------|---------------|--------------------|
| feature_1 | numeric | mm | [...] | 0–150 |
| feature_2 | categorial | – | [...] | {A, B, C} |
| feature_3 | numeric | m/s | [...] | 0–2.5 |
| ... | ... | ... | ... | ... |

**Fișier recomandat:**  `data/README.md`

---

##  3. Analiza Exploratorie a Datelor (EDA) – Sintetic

### 3.1 Statistici descriptive aplicate

* **Medie, mediană, deviație standard**
* **Min–max și quartile**
* **Distribuții pe caracteristici** (histograme)
* **Identificarea outlierilor** (IQR / percentile)

### 3.2 Analiza calității datelor

* **Detectarea valorilor lipsă** (% pe coloană)
* **Detectarea valorilor inconsistente sau eronate**
* **Identificarea caracteristicilor redundante sau puternic corelate**

### 3.3 Probleme identificate

* [exemplu] Feature X are 8% valori lipsă
* [exemplu] Distribuția feature Y este puternic neuniformă
* [exemplu] Variabilitate ridicată în clase (class imbalance)

---

##  4. Preprocesarea Datelor

### 4.1 Curățarea datelor

* **Eliminare duplicatelor**
* **Tratarea valorilor lipsă:**
  * Feature A: imputare cu mediană
  * Feature B: eliminare (30% valori lipsă)
* **Tratarea outlierilor:** IQR / limitare percentile

### 4.2 Transformarea caracteristicilor

* **Normalizare:** Min–Max / Standardizare
* **Encoding pentru variabile categoriale**
* **Ajustarea dezechilibrului de clasă** (dacă este cazul)

### 4.3 Structurarea seturilor de date

**Împărțire recomandată:**
* 70–80% – train
* 10–15% – validation
* 10–15% – test

**Principii respectate:**
* Stratificare pentru clasificare
* Fără scurgere de informație (data leakage)
* Statistici calculate DOAR pe train și aplicate pe celelalte seturi

### 4.4 Salvarea rezultatelor preprocesării

* Date preprocesate în `data/processed/`
* Seturi train/val/test în foldere dedicate
* Parametrii de preprocesare în `config/preprocessing_config.*` (opțional)

---

##  5. Fișiere Generate în Această Etapă

* `data/raw/` – date brute
* `data/processed/` – date curățate & transformate
* `data/train/`, `data/validation/`, `data/test/` – seturi finale
* `src/preprocessing/` – codul de preprocesare
* `data/README.md` – descrierea dataset-ului

---

##  6. Stare Etapă (de completat de student)

- [ ] Structură repository configurată
- [ ] Dataset analizat (EDA realizată)
- [ ] Date preprocesate
- [ ] Seturi train/val/test generate
- [ ] Documentație actualizată în README + `data/README.md`

---
