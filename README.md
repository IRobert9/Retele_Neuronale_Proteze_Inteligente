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

* **Număr total de observații:** ~580.000+ ferestre
* Notă: O observație = o fereastră de timp de 150 ms (sliding window).

Se foloseste una dintre bazele de date din NinaPro (DB2) unde se regasesc:
40 subiecți sănătoși intacți * 49 miscari ale mainii * 10 repetari

* **Număr de caracteristici (features):** 12 (cele 12 canale ale senzorilor EMG).
* **Tipuri de date:**  Serii de timp numerice (Matrice 150 eșantioane x 12 canale)
* **Format fișiere:** .mat (Structuri MATLAB proprietare NinaPro)

### 2.3 Descrierea fiecărei caracteristici

| **Caracteristică** | **Tip** | **Unitate** | **Descriere** | **Domeniu valori** |
|-------------------|---------|-------------|---------------|--------------------|
| emg (ch 1-12) | numeric (float) | μV/a.u | Semnalul electric captat de cei 12 electrozi Delsys Trigno | Valori brute mici |
| stimulus | categorial (int) | – | Eticheta mișcării pe care subiectul trebuie să o execute. | 0 (Rest) ... 40 (Diverse mișcări) |
| restimulus | categorial (int) | - | Indică dacă subiectul este în pauză sau execută mișcarea. | 0 (Mișcare), 1 (Repaus) |
| repetition | categorial (int) | - | Indexul repetării curente a mișcării. | 1 – 6 |
| subject | categorial (int) | - | ID-ul unic al subiectului (extras din numele fișierului). | 1 – 40 |

**Fișier recomandat:**  `data/README.md`

---

##  3. Analiza Exploratorie a Datelor (EDA) – Sintetic

### 3.1 Statistici descriptive aplicate

* **Medie și deviație standard:** Calculate per fereastră (axis=1) în etapa de normalizare pentru a centra semnalul în 0.
  
* **Distribuții pe clase:** S-a observat o variație a numărului de ferestre per mișcare (datorată duratei variabile de execuție a subiecților).
Exemplu: Mișcările complexe (Grasps) tind să dureze mai mult decât mișcările simple (Wrist Flexion), generând mai multe ferestre.

### 3.2 Analiza calității datelor

* **Sincronizare:** S-au detectat discrepanțe minore (de 1 eșantion) între lungimea vectorilor emg și stimulus în fișierele brute .mat.
* **Clase rare:** Anumite mișcări au avut un număr insuficient de exemple (< 20 ferestre) în cazul unor subiecți, riscând să destabilizeze antrenarea.
* **Zgomot de repaus:** Semnalele marcate cu stimulus=0 (Repaus) conțin zgomot de fond care nu este relevant pentru clasificarea mișcărilor active.

### 3.3 Probleme identificate

* Mismatch de dimensiuni: Vectori inegali (ex: 877073 vs 877072) -> Soluționat prin tăiere la lungimea minimă comună.
* Class Imbalance: Clasele au număr diferit de exemple -> Soluționat prin ponderarea claselor (implicit prin date masive) și Stratified Split.
* Nepotrivire Etichete: Fișierele E2 conțin etichete începând de la 13, nu de la 1 -> Soluționat prin remaparea dicționarului label_map.

---

##  4. Preprocesarea Datelor

### 4.1 Curățarea datelor

* **Filtrare Activă:** S-au eliminat toate eșantioanele unde stimulus == 0 sau restimulus != 0. Se păstrează doar momentele de contracție musculară activă.
* **Eliminarea claselor rare:** S-a implementat un prag (min_windows = 20). Clasele cu mai puțin de 20 de ferestre sunt eliminate automat din setul de date.
* **Corecție lungimi:** Trunchierea automată a vectorilor la min(len(emg), len(stimulus)).

### 4.2 Transformarea caracteristicilor

* **Windowing (Ferestruire):**
Tehnică: Sliding Window (Fereastră Glisantă).
Dimensiune fereastră: 150 eșantioane (cca. 150-200ms).
Suprapunere (Step Size): 20 eșantioane (pentru augmentarea datelor și continuitate).

* **Normalizare (Z-Score per Window):**

Se aplică individual pe fiecare fereastră pentru a reduce efectul variațiilor de amplitudine (oboseală musculară, conductivitate piele).

* **Encoding:**
Etichetele (13, 14, ...) au fost mapate la indici consecutivi (0, 1, ...) și transformate în vectori One-Hot pentru antrenare (to_categorical).

### 4.3 Structurarea seturilor de date
S-a folosit funcția train_test_split cu stratificare (stratify=y) pentru a menține proporția mișcărilor în toate seturile:
**Împărțire recomandată:**
* 70% – train
* 15% – validation
* 15% – test

**Principii respectate:**
* Stratificare pentru clasificare
* Fără scurgere de informație (data leakage)
* Statistici calculate DOAR pe train și aplicate pe celelalte seturi

### 4.4 Salvarea rezultatelor preprocesării

* Modelul antrenat este salvat în format .keras și .tflite.
* Artifactele (codul, logurile) sunt salvate local.

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
