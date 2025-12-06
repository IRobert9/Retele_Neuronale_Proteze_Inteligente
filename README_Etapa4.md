# 📘 README – Etapa 4: Arhitectura Completă a Aplicației SIA bazată pe Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** [Nume Prenume]  
**Link Repository GitHub**
**Data:** [Data]  
---

## Scopul Etapei 4

Această etapă corespunde punctului **5. Dezvoltarea arhitecturii aplicației software bazată pe RN** din lista de 9 etape - slide 2 **RN Specificatii proiect.pdf**.

**Trebuie să livrați un SCHELET COMPLET și FUNCȚIONAL al întregului Sistem cu Inteligență Artificială (SIA). In acest stadiu modelul RN este doar definit și compilat (fără antrenare serioasă).**

### IMPORTANT - Ce înseamnă "schelet funcțional":

 **CE TREBUIE SĂ FUNCȚIONEZE:**
- Toate modulele pornesc fără erori
- Pipeline-ul complet rulează end-to-end (de la date → până la output UI)
- Modelul RN este definit și compilat (arhitectura există)
- Web Service/UI primește input și returnează output

 **CE NU E NECESAR ÎN ETAPA 4:**
- Model RN antrenat cu performanță bună
- Hiperparametri optimizați
- Acuratețe mare pe test set
- Web Service/UI cu funcționalități avansate

**Scopul anti-plagiat:** Nu puteți copia un notebook + model pre-antrenat de pe internet, pentru că modelul vostru este NEANTRENAT în această etapă. Demonstrați că înțelegeți arhitectura și că ați construit sistemul de la zero.

---

##  Livrabile Obligatorii

### 1. Tabelul Nevoie Reală → Soluție SIA → Modul Software (max ½ pagină)
Completați in acest readme tabelul următor cu **minimum 2-3 rânduri** care leagă nevoia identificată în Etapa 1-2 cu modulele software pe care le construiți (metrici măsurabile obligatoriu):

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul vostru** | **Modul software responsabil** |
|---------------------------|--------------------------------|--------------------------------|
| Controlul intuitiv al unei proteze mioelectrice pentru amputați transradiali | Clasificare semnal EMG în 7 mișcări funcționale (Pumn, Palmă, etc.) cu acuratețe > 95% | Modul 2 (Neural Network) (ResNet 1D) |
| Rejecția mișcărilor involuntare și a zgomotului de senzor | Nod decizional bazat pe "Confidence Threshold" (prag > 60%) pentru siguranță | Modul 3 (Server Python) + Logică pre-procesare |
| Vizualizarea în timp real a intenției de mișcare și a semnalului brut | Interfață grafică (GUI) cu latență mică (< 50ms) via TCP/IP | Modul 3 (LabVIEW UI) |

**Instrucțiuni:**
- Fiți concreti (nu vagi): "detectare fisuri sudură" ✓, "îmbunătățire proces" ✗
- Specificați metrici măsurabile: "< 2 secunde", "> 95% acuratețe", "reducere 20%"
- Legați fiecare nevoie de modulele software pe care le dezvoltați

---

### 2. Contribuția Voastră Originală la Setul de Date – MINIM 40% din Totalul Observațiilor Finale

**Regula generală:** Din totalul de **N observații finale** în `data/processed/`, **minimum 40%** trebuie să fie **contribuția voastră originală**.

#### Cum se calculează 40%:

**Exemplu 1 - Dataset DOAR public în Etapa 3:**
```
Etapa 3: Ați folosit 10,000 samples dintr-o sursa externa (ex: Kaggle)
Etapa 4: Trebuie să generați/achiziționați date astfel încât:
  
Opțiune A: Adăugați 6,666 samples noi → Total 16,666 (6,666/16,666 = 40%)
Opțiune B: Păstrați 6,000 publice + 4,000 generate → Total 10,000 (4,000/10,000 = 40%)
```

**Exemplu 2 - Dataset parțial original în Etapa 3:**
```
Etapa 3: Ați avut deja 3,000 samples generate + 7,000 publice = 10,000 total
Etapa 4: 3,000 samples existente numără ca "originale"
        Dacă 3,000/10,000 = 30% < 40% → trebuie să generați încă ~1,700 samples
        pentru a ajunge la 4,700/10,000 = 47% > 40% ✓
```

**Exemplu 3 - Dataset complet original:**
```
Etapa 3-4: Generați toate datele (simulare, senzori proprii, etichetare manuală - varianta recomandata)
           → 100% original ✓ (depășește cu mult 40% - FOARTE BINE!)
```

#### Tipuri de contribuții acceptate (exemple din inginerie):

| **Date sintetice prin metode avansate** | • Deoarece accesul la pacienți reali pentru achiziție nouă este limitat, am dezvoltat un modul software de generare a datelor sintetice pentru a crește robustețea modelului la condiții reale imperfecte. Am aplicat două tehnici de augmentare asupra datelor NinaPro DB2:

Injectare de Zgomot Gaussian (White Noise): Pentru a simula senzori EMG low-cost sau interferențe electromagnetice ambientale.

Scalare Dinamică a Amplitudinii: Am multiplicat semnalele cu factori aleatori (0.85 - 1.15) pentru a simula variația forței musculare și oboseala (atenuarea semnalului). Aceste date au fost generate static și adăugate la setul de antrenare înainte de procesul de învățare.

#### Declarație obligatorie în README:
### Contribuția originală la setul de date:

**Total observații finale:** Aprox. 230.000 ferestre (163k reale + 67k generate).
**Observații originale:** Observații originale: ~67.000 (40% din totalul de antrenare).

**Tipul contribuției:**
[ ] Date generate prin simulare fizică  
[ ] Date achiziționate cu senzori proprii  
[ ] Etichetare/adnotare manuală  
[X] Date sintetice prin metode avansate  

**Descriere detaliată:**
Deoarece accesul la pacienți reali pentru achiziție nouă este limitat, am dezvoltat un modul software de generare a datelor sintetice pentru a crește robustețea modelului la condiții reale imperfecte. Am aplicat două tehnici de augmentare asupra datelor NinaPro DB2:

Injectare de Zgomot Gaussian (White Noise): Pentru a simula senzori EMG low-cost sau interferențe electromagnetice ambientale.

Scalare Dinamică a Amplitudinii: Am multiplicat semnalele cu factori aleatori (0.85 - 1.15) pentru a simula variația forței musculare și oboseala (atenuarea semnalului). Aceste date au fost generate static și adăugate la setul de antrenare înainte de procesul de învățare.

**Locația codului:** src/data_acquisition/data_generator.py (sau secțiunea dedicată din prelucrare_date.py)
**Locația datelor:** data/generated/ (sau integrate în pipeline)

**Dovezi:**
- Grafic comparativ: `docs/generated_vs_real.png`
- Setup experimental: `docs/acquisition_setup.jpg` (dacă aplicabil)
- Tabel statistici: `docs/data_statistics.csv`
```

#### Exemple pentru "contribuție originală":
-Simulări fizice realiste cu ecuații și parametri justificați  
-Date reale achiziționate cu senzori proprii (setup documentat)  
-Augmentări avansate cu justificare fizică (ex: simulare perspective camera industrială)  


#### Atenție - Ce NU este considerat "contribuție originală":

- Augmentări simple (rotații, flips, crop) pe date publice  
- Aplicare filtre standard (Gaussian blur, contrast) pe imagini publice  
- Normalizare/standardizare (aceasta e preprocesare, nu generare)  
- Subset dintr-un dataset public (ex: selectat 40% din ImageNet)


---

### 3. Diagrama State Machine a Întregului Sistem (OBLIGATORIE)

**Cerințe:**
- **Minimum 4-6 stări clare** cu tranziții între ele
- **Formate acceptate:** PNG/SVG, pptx, draw.io 
- **Locație:** `docs/state_machine.*` (orice extensie)
- **Legendă obligatorie:** 1-2 paragrafe în acest README: "De ce ați ales acest State Machine pentru nevoia voastră?"

**Stări tipice pentru un SIA:**

[IDLE / START] 
      ↓
[ACQUIRE_WINDOW] (Citește 150ms de semnal brut)
      ↓
[PREPROCESS] (Normalizare Z-Score)
      ↓
[INFERENCE_RN] (Modelul ResNet prezice clasa)
      ↓
[DECISION_LOGIC] (Verifică Confidence > 60%)
      │
      ├─ [Low Confidence] → [SEND_REST] (Trimite comandă "Stai")
      │
      └─ [High Confidence] → [SEND_MOVEMENT] (Trimite ID-ul mișcării)
            ↓
[TCP_TRANSMIT] (Trimite JSON către LabVIEW)
      ↓
[UPDATE_UI] (LabVIEW afișează mișcarea)
      ↓
(Înapoi la ACQUIRE_WINDOW)

**Legendă obligatorie (scrieți în README):**
### Justificarea State Machine-ului ales:

Am ales o arhitectură de tip Procesare Continuă în Timp Real (Streaming) deoarece o proteză trebuie să răspundă instantaneu la comenzile utilizatorului.

Stările principale sunt:
1. ACQUIRE_WINDOW: Simularea senzorului care umple un buffer de 150ms.
2. INFERENCE_RN: Pasul critic unde rețeaua neuronală clasifică intenția.
3. DECISION_LOGIC: Un filtru de siguranță esențial. Dacă rețeaua nu este sigură (probabilitate mică), proteza nu trebuie să se miște haotic, ci să intre în starea de siguranță (REST).

Sistemul include o stare de eroare (TCP_ERROR) care gestionează pierderea conexiunii cu interfața LabVIEW, asigurând reconectarea automată fără a opri procesul de analiză.

Tranzițiile critice sunt:
- [STARE_A] → [STARE_B]: [când se întâmplă - ex: "când buffer-ul atinge 1024 samples"]
- [STARE_X] → [ERROR]: [condiții - ex: "când senzorul nu răspunde > 100ms"]

Starea ERROR este esențială pentru că [explicați ce erori pot apărea în contextul 
aplicației voastre industriale - ex: "senzorul se poate deconecta în mediul industrial 
cu vibrații și temperatură variabilă, trebuie să gestionăm reconnect automat"].

Bucla de feedback [dacă există] funcționează astfel: [ex: "rezultatul inferenței 
actualizează parametrii controlerului PID pentru reglarea vitezei motorului"].
```

---

### 4. Scheletul Complet al celor 3 Module Cerute la Curs (slide 7)

Toate cele 3 module trebuie să **pornească și să ruleze fără erori** la predare. Nu trebuie să fie perfecte, dar trebuie să demonstreze că înțelegeți arhitectura.

| **Modul** | **Python (exemple tehnologii)** / **LabVIEW** | **Cerință minimă funcțională (la predare)** |
|-----------|----------------------------------|-------------|----------------------------------------------|
| **1. Data Logging / Acquisition** | prelucrare_date.py (partea de încărcare și generare) | Citește fișierele .mat, aplică filtre, generează datele sintetice (40%) și creează ferestrele de timp (Windowing). |
| **2. Neural Network Module** | src/neural_network/resnet_model.py | LLB cu VI-uri RN | Definirea arhitecturii ResNet 1D, compilarea modelului și procesul de antrenare. Modelul salvat este model_proteza_final.keras. |
| **3. Web Service / UI** | server_proteza.py + LabVIEW VI | Serverul Python preia datele, rulează inferența și trimite rezultatele prin TCP către aplicația client dezvoltată în LabVIEW.

#### Detalii per modul:

#### **Modul 1: Data Logging / Acquisition**

**Funcționalități obligatorii:**
- [X] Cod rulează fără erori: `python src/data_acquisition/generate.py` sau echivalent LabVIEW
- [X] Generează CSV în format compatibil cu preprocesarea din Etapa 3
- [X] Include minimum 40% date originale în dataset-ul final
- [X] Documentație în cod: ce date generează, cu ce parametri

#### **Modul 2: Neural Network Module**

**Funcționalități obligatorii:**
- [X] Arhitectură RN definită și compilată fără erori
- [ ] Model poate fi salvat și reîncărcat
- [X] Include justificare pentru arhitectura aleasă (în docstring sau README)
- [ ] **NU trebuie antrenat** cu performanță bună (weights pot fi random)


#### **Modul 3: Web Service / UI**

**Funcționalități MINIME obligatorii:**
- [ ] Propunere Interfață ce primește input de la user (formular, file upload, sau API endpoint)
- [ ] Includeți un screenshot demonstrativ în `docs/screenshots/`

**Ce NU e necesar în Etapa 4:**
- UI frumos/profesionist cu grafică avansată
- Funcționalități multiple (istorice, comparații, statistici)
- Predicții corecte (modelul e neantrenat, e normal să fie incorect)
- Deployment în cloud sau server de producție

**Scop:** Prima demonstrație că pipeline-ul end-to-end funcționează: input user → preprocess → model → output.


## Structura Repository-ului la Finalul Etapei 4 (OBLIGATORIE)

**Verificare consistență cu Etapa 3:**

```
proiect-rn-[nume-prenume]/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── generated/  # Date originale
│   ├── train/
│   ├── validation/
│   └── test/
├── src/
│   ├── data_acquisition/
│   ├── preprocessing/  # Din Etapa 3
│   ├── neural_network/
│   └── app/  # UI schelet
├── docs/
│   ├── state_machine.*           #(state_machine.png sau state_machine.pptx sau state_machine.drawio)
│   └── [alte dovezi]
├── models/  # Untrained model
├── config/
├── README.md
├── README_Etapa3.md              # (deja existent)
├── README_Etapa4_Arhitectura_SIA.md              # ← acest fișier completat (în rădăcină)
└── requirements.txt  # Sau .lvproj
```

**Diferențe față de Etapa 3:**
- Adăugat `data/generated/` pentru contribuția dvs originală
- Adăugat `src/data_acquisition/` - MODUL 1
- Adăugat `src/neural_network/` - MODUL 2
- Adăugat `src/app/` - MODUL 3
- Adăugat `models/` pentru model neantrenat
- Adăugat `docs/state_machine.png` - OBLIGATORIU
- Adăugat `docs/screenshots/` pentru demonstrație UI

---

## Checklist Final – Bifați Totul Înainte de Predare

### Documentație și Structură
- [ ] Tabelul Nevoie → Soluție → Modul complet (minimum 2 rânduri cu exemple concrete completate in README_Etapa4_Arhitectura_SIA.md)
- [ ] Declarație contribuție 40% date originale completată în README_Etapa4_Arhitectura_SIA.md
- [ ] Cod generare/achiziție date funcțional și documentat
- [ ] Dovezi contribuție originală: grafice + log + statistici în `docs/`
- [ ] Diagrama State Machine creată și salvată în `docs/state_machine.*`
- [ ] Legendă State Machine scrisă în README_Etapa4_Arhitectura_SIA.md (minimum 1-2 paragrafe cu justificare)
- [ ] Repository structurat conform modelului de mai sus (verificat consistență cu Etapa 3)

### Modul 1: Data Logging / Acquisition
- [ ] Cod rulează fără erori (`python src/data_acquisition/...` sau echivalent LabVIEW)
- [ ] Produce minimum 40% date originale din dataset-ul final
- [ ] CSV generat în format compatibil cu preprocesarea din Etapa 3
- [ ] Documentație în `src/data_acquisition/README.md` cu:
  - [ ] Metodă de generare/achiziție explicată
  - [ ] Parametri folosiți (frecvență, durată, zgomot, etc.)
  - [ ] Justificare relevanță date pentru problema voastră
- [ ] Fișiere în `data/generated/` conform structurii

### Modul 2: Neural Network
- [ ] Arhitectură RN definită și documentată în cod (docstring detaliat) - versiunea inițială 
- [ ] README în `src/neural_network/` cu detalii arhitectură curentă

### Modul 3: Web Service / UI
- [ ] Propunere Interfață ce pornește fără erori (comanda de lansare testată)
- [ ] Screenshot demonstrativ în `docs/screenshots/ui_demo.png`
- [ ] README în `src/app/` cu instrucțiuni lansare (comenzi exacte)

---

**Predarea se face prin commit pe GitHub cu mesajul:**  
`"Etapa 4 completă - Arhitectură SIA funcțională"`

**Tag obligatoriu:**  
`git tag -a v0.4-architecture -m "Etapa 4 - Skeleton complet SIA"`


