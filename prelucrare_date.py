# -------------------------------
# 1.a. Importuri
# -------------------------------
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import time
import os
import glob

# -------------------------------
# 1.b. Maparea Etichetelor ACTUALIZATĂ (NinaPro DB2 Complet)
# -------------------------------
label_map = {
    # --- Mișcări de Bază & Încheietură ---
    13: "Wrist flexion",
    14: "Wrist extension",
    15: "Radial deviation",
    16: "Ulnar deviation",
    17: "Wrist pronation",
    18: "Wrist supination",
    19: "Hand close",       # <--- Critic
    20: "Hand open",        # <--- Critic
    
    # --- Prize Funcționale (Set A - E2) ---
    21: "Power grasp",      # Apucare de forță
    22: "Lateral grasp",    # Apucare cheie/laterală
    23: "Tripod grasp",     # Apucare cu 3 degete
    24: "Index pointer",    # Arătător întins
    25: "Thumb pointer",    # Deget mare întins
    26: "Pinch 2 fingers",  # Ciupire 2 degete
    27: "Pinch 3 fingers",
    28: "Pinch 4 fingers",
    29: "Stick grasp",      # Apucare băț
    
    # --- Prize Funcționale Avansate (Set B - E3/Extins) ---
    # Acestea sunt clasele care apăreau ca "Class 30-40"
    30: "Large Dia. Grasp", # Apucare obiect gros
    31: "Small Dia. Grasp", # Apucare obiect subțire
    32: "Fixed Hook",       # Cârlig (ex: geantă)
    33: "Palmar Pinch",     # Ciupire spre palmă 
    34: "Tip Pinch",        # Pensetă fină
    35: "Sphere Grasp",     # Minge
    36: "Plate/Disk Grasp", # Disc/Farfurie
    37: "Hook Grasp",       # Cârlig simplu
    38: "Parallel Ext.",    # Extensie paralelă
    39: "Palm Ext.",        # Extensie palmă
    40: "Tip Pinch Open",   # Pensetă deschisă
    
    # --- Repaus ---
    0: "Rest" 
}

# 2. Configurare Căutare
# Setează calea către folderul principal unde ai toate datele
data_root_path = r'D:\Proiect_ReteleNeuronale' 

# Căutăm recursiv toate fișierele care se termină cu '_E2_A1.mat'
print(f"Caut fișiere E2_A1 în: {data_root_path} ...")
search_pattern = os.path.join(data_root_path, "**", "*_E2_A1.mat")
file_list = glob.glob(search_pattern, recursive=True)

if not file_list:
    raise ValueError("Nu am găsit niciun fișier *_E2_A1.mat! Verifică calea.")

print(f"Am găsit {len(file_list)} fișiere. Încep procesarea...")

# 3. Configurare Ferestre
window_size = 150
step_size = 20  # <--- Sfat: Pune 20 sau 30 pentru a economisi memorie RAM cu mulți subiecți
all_x_windows = []
all_y_labels = []

# 4. Procesare fiecare fișier în parte
for file_path in file_list:
    filename = os.path.basename(file_path)
    # print(f"  -> Procesez: {filename}") # Decomentează dacă vrei să vezi progresul detaliat
    
    try:
        data = sio.loadmat(file_path)
        emg = data['emg']
        stimulus = data['stimulus'].flatten()
        restimulus = data['restimulus'].flatten()

        # --- FIX: Tăiem la lungimea minimă (pentru siguranță) ---
        min_len = min(len(stimulus), len(restimulus), emg.shape[0])
        emg = emg[:min_len, :]
        stimulus = stimulus[:min_len]
        restimulus = restimulus[:min_len]

        # --- Selectare eșantioane active ---
        mask = (stimulus > 0) & (restimulus == 0)
        emg_active = emg[mask, :]
        stimulus_active = stimulus[mask]

        # Dacă nu sunt destule date în acest fișier, trecem la următorul
        if len(emg_active) < window_size:
            continue

        # --- Creare ferestre (Windowing) ---
        # Folosim o buclă locală pentru viteză
        for start in range(0, len(emg_active) - window_size, step_size):
            end = start + window_size
            window_data = emg_active[start:end, :]
            
            # Eticheta ferestrei = cea mai frecventă valoare din stimulus
            # (Optimizare: luăm valoarea din mijloc sau majoritatea, aici luăm majoritatea)
            labels_in_window = stimulus_active[start:end]
            # O metodă rapidă de a găsi moda (elementul cel mai frecvent)
            label = np.bincount(labels_in_window).argmax()
            
            all_x_windows.append(window_data)
            all_y_labels.append(label)

    except Exception as e:
        print(f"!!! Eroare la citirea fișierului {filename}: {e}")

# 5. Convertire la NumPy Arrays (Formatul final pentru restul codului)
print("Concatenare finală a datelor... Așteaptă...")
x = np.array(all_x_windows)
y = np.array(all_y_labels)

print(f"\n[GATA] Dataset final creat.")
print(f" - Total ferestre (x): {x.shape}")
print(f" - Total etichete (y): {y.shape}")

# -------------------------------
# 6. Eliminarea automată a claselor rare
# -------------------------------
min_windows = 20
unique, counts = np.unique(y, return_counts=True)
classes_to_remove = unique[counts < min_windows]

print("\nClase găsite în fișier:", unique)
if len(classes_to_remove) > 0:
    print(f"Clase eliminate (sub {min_windows} ferestre): {classes_to_remove}")

mask_keep = ~np.isin(y, classes_to_remove)
x = x[mask_keep]
y = y[mask_keep]

valid_classes = np.unique(y)

# Remapare y la 0..N-1 pentru antrenare
label_to_index = {label: i for i, label in enumerate(valid_classes)}
index_to_label = {i: label for label, i in label_to_index.items()}
y_mapped = np.array([label_to_index[l] for l in y])

num_classes = len(valid_classes)
print(f"Clase valide rămase: {valid_classes}")
print(f"Număr total ferestre: {len(x)}")

# ==============================================================================
# SECTIUNE NOUA (6.b): Gruparea Claselor (Simplificare Strategică)
# ==============================================================================
print("\n--- APLICARE STRATEGIE DE GRUPARE (23 -> 7 Clase) ---")

# Definim harta: {Clasa_Veche_NinaPro: Clasa_Noua_Simplificata}
# Tinta: 0=Flex, 1=Ext, 2=Pro, 3=Sup, 4=Power, 5=Open, 6=Pinch

grouping_map = {
    # --- 1. Wrist Flexion (0) ---
    13: 0, # Wrist Flexion
    16: 0, # Ulnar Deviation (seamănă muscular cu flexia)
    37: 0, # Wrist Flexion (din setul E3)
    
    # --- 2. Wrist Extension (1) ---
    14: 1, # Wrist Extension
    15: 1, # Radial Deviation (seamănă muscular cu extensia)
    38: 1, # Wrist Extension (din setul E3)
    
    # --- 3. Pronation (2) ---
    17: 2,
    39: 2, # Pronation / Palm Ext (contextual)
    
    # --- 4. Supination (3) ---
    18: 3,
    40: 3, # Supination / Tip Pinch Open (contextual)
    
    # --- 5. Power Grip / Hand Close (4) ---
    19: 4, # Hand close classic
    21: 4, # Power grasp
    22: 4, # Lateral grasp
    29: 4, # Stick grasp
    30: 4, # Large Dia
    31: 4, # Small Dia
    32: 4, # Fixed Hook
    35: 4, # Sphere Grasp
    36: 4, # Plate Grasp
    37: 4, # Hook Grasp (duplicate protection)
    
    # --- 6. Hand Open (5) ---
    20: 5, # Hand open classic
    # Notă: Deschiderile sunt mai rare în E3, folosim 20 ca bază principală
    
    # --- 7. Precision Pinch (6) ---
    23: 6, # Tripod
    24: 6, # Index pointer
    25: 6, # Thumb pointer
    26: 6, # Pinch 2 fingers
    27: 6, # Pinch 3 fingers
    28: 6, # Pinch 4 fingers
    33: 6, # Palmar Pinch
    34: 6, # Tip Pinch
}

# Funcție care aplică gruparea
def apply_grouping(y_original, mapping):
    y_new = []
    indices_to_keep = []
    
    for i in range(len(y_original)):
        original_label = y_original[i]
        # Dacă eticheta există în harta noastră, o traducem
        if original_label in mapping:
            y_new.append(mapping[original_label])
            indices_to_keep.append(i)
        
    return np.array(y_new), np.array(indices_to_keep)

# Aplicăm transformarea
y_grouped, valid_indices = apply_grouping(y, grouping_map)
x_grouped = x[valid_indices] # Păstrăm doar datele mapate

# Suprascriem variabilele principale
x = x_grouped
y = y_grouped

# Actualizăm variabilele de sistem
valid_classes = np.unique(y)
num_classes = len(valid_classes)

# Definim noul Label Map pentru grafice
label_map = {
    0: "Wrist Flexion",
    1: "Wrist Extension",
    2: "Pronation",
    3: "Supination",
    4: "Power Grip (Close)",
    5: "Hand Open",
    6: "Precision Pinch"
}

print(f"[REZULTAT GRUPARE]")
print(f" - Noua formă x: {x.shape}")
print(f" - Clase rămase: {valid_classes}")
print(f" - Nume clase: {list(label_map.values())}")
print("----------------------------------------------------------\n")

# AICI CONTINUĂ CODUL TĂU CU: # 7. Split și Normalizare...

# -------------------------------
# 7. Split și Normalizare (CORECTAT)
# -------------------------------
# Pas Critic: Regenerăm y_mapped pe baza datelor grupate (y)
# Acum y are valori gen [1, 4, 6...]. Trebuie să le facem [0, 1, 2...] strict.

valid_classes_final = np.unique(y)
num_classes = len(valid_classes_final)

# Mapare strictă 0..N-1
final_label_map = {original: idx for idx, original in enumerate(valid_classes_final)}
y_mapped = np.array([final_label_map[lbl] for lbl in y])

print(f"\n[INFO] Etichete regenerate pentru antrenare.")
print(f" - Clase originale găsite: {valid_classes_final}")
print(f" - Mapped to 0..{num_classes-1}")

# Shuffle
x, y_mapped = shuffle(x, y_mapped, random_state=42)

# Split Stratificat
x_train, x_temp, y_train, y_temp = train_test_split(
    x, y_mapped, test_size=0.3, random_state=42, stratify=y_mapped)

x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# One-hot encoding (Acum va merge perfect)
y_train_cat = to_categorical(y_train, num_classes)
y_val_cat   = to_categorical(y_val, num_classes)
y_test_cat  = to_categorical(y_test, num_classes)

# Normalizare Z-score per fereastră
def normalize_windows(data):
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    return (data - mean) / (std + 1e-8)

x_train = normalize_windows(x_train)
x_val = normalize_windows(x_val)
x_test = normalize_windows(x_test)

# Actualizăm label_map-ul de afișare pentru a corespunde noilor indici 0, 1, 2...
# Preluăm numele din harta veche pe baza claselor care au "supraviețuit"
new_display_map = {}
old_map_names = label_map # label_map-ul definit la grupare
for original_cls, new_idx in final_label_map.items():
    new_display_map[new_idx] = old_map_names.get(original_cls, f"Class {original_cls}")

# Suprascriem label_map pentru ca graficele de la final să afișeze numele corecte
label_map = new_display_map

# -------------------------------
# 8. Model ResNet 1D (Advanced Architecture)
# -------------------------------
from tensorflow.keras.layers import Add, Activation, Input, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    
    # Calea principală (Convoluții)
    x = Conv1D(filters, kernel_size, padding='same', strides=stride, kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x) # Dropout moderat în interiorul blocului

    x = Conv1D(filters, kernel_size, padding='same', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)

    # Calea scurtăturii (Shortcut / Skip Connection)
    # Dacă dimensiunile nu se potrivesc (ex: am schimbat nr de filtre sau stride), ajustăm scurtătura
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = Conv1D(filters, 1, padding='same', strides=stride)(shortcut)
        shortcut = BatchNormalization()(shortcut)

    # AICI e magia: Adunăm rezultatul procesat cu cel original (x + shortcut)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

# --- Construcția Rețelei ---
input_shape = (x_train.shape[1], x_train.shape[2])
inputs = Input(shape=input_shape)

# Strat inițial - Procesare brută
x = Conv1D(64, kernel_size=7, padding='same', strides=1)(inputs)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# Adăugăm blocurile reziduale (adâncimea rețelei)
# Creștem progresiv numărul de filtre pentru a extrage trăsături tot mai complexe
x = residual_block(x, filters=64, kernel_size=5)
x = residual_block(x, filters=64, kernel_size=5)
x = MaxPooling1D(pool_size=2)(x) # Reducem dimensiunea (Downsampling)

x = residual_block(x, filters=128, kernel_size=3)
x = residual_block(x, filters=128, kernel_size=3)
x = MaxPooling1D(pool_size=2)(x)

x = residual_block(x, filters=256, kernel_size=3)
x = residual_block(x, filters=256, kernel_size=3)

# Clasificatorul Final
x = GlobalAveragePooling1D()(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(0.0001))(x)
x = Dropout(0.5)(x) # Dropout final puternic
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs, outputs)

# Compilare
# Folosim un learning rate puțin mai mic pentru stabilitatea ResNet
optimizer = Adam(learning_rate=0.0005) 

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Callbacks (Foarte importante pentru ResNet)
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
]

# -------------------------------
# 9. Antrenare
# -------------------------------
history = model.fit(
    x_train, y_train_cat,
    validation_data=(x_val, y_val_cat),
    epochs=60,            # Putem pune mai multe, EarlyStopping îl va opri
    batch_size=64,
    callbacks=callbacks
)

# -------------------------------
# 10. Evaluare și Raportare (Final & Corectat)
# -------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}")

# Predicții
y_pred_probs = model.predict(x_test)
y_pred_idx = np.argmax(y_pred_probs, axis=1)

# Construim lista de nume corecte pentru cele 7 clase
# Ordinea este critică: 0, 1, 2, 3, 4, 5, 6
target_names_7_classes = [
    "Wrist Flexion", 
    "Wrist Extension", 
    "Pronation", 
    "Supination", 
    "Power Grip (Close)", 
    "Hand Open", 
    "Precision Pinch"
]

# Matricea de Confuzie (Simplificată - folosim direct y_test numeric)
cm = confusion_matrix(y_test, y_pred_idx)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names_7_classes,
            yticklabels=target_names_7_classes)
plt.title(f'Confusion Matrix (Accuracy: {test_acc*100:.1f}%)')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.show()

# -------------------------------
# 10. (Final Fix) Raportare Robustă
# -------------------------------
# Definim explicit etichetele numerice posibile (0..6)
all_labels = [0, 1, 2, 3, 4, 5, 6]

print("\nClassification Report (FIXED):")
print(classification_report(
    y_test, 
    y_pred_idx, 
    labels=all_labels,  # <--- ASTA rezolvă problema
    target_names=target_names_7_classes,
    zero_division=0     # Evită erori dacă o clasă e goală
))

# -------------------------------
# 11. Test Viteză Inferență (Realist)
# -------------------------------
import tensorflow as tf
# Convertim la TFLite pentru test relevant de viteză
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Testăm pe un singur eșantion
input_data = np.array(x_test[0:1], dtype=np.float32)
start_time = time.time()
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
end_time = time.time()

print(f"\n[Viteză TFLite] Timp procesare o fereastră: {(end_time - start_time) * 1000:.2f} ms")