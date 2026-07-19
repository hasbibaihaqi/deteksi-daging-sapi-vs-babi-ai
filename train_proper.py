import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import glob, shutil

# ─────────────────────────────────────────────────────
# 1. BACA dan SORT DATASET dari struktur Segar / Tidak_Segar
# ─────────────────────────────────────────────────────
BASE_DATASET  = 'training/dataset_daging'
SEGAR_DIR     = os.path.join(BASE_DATASET, 'Segar')
TIDAK_DIR     = os.path.join(BASE_DATASET, 'Tidak_Segar')

# Kelas Babi = semua file di folder Babi
# Kelas Sapi = semua file di folder Sapi
babi_files = [f for f in glob.glob(os.path.join(BASE_DATASET, 'Babi', '*.jpg'))] + \
             [f for f in glob.glob(os.path.join(BASE_DATASET, 'Babi', '*.png'))]

sapi_files = [f for f in glob.glob(os.path.join(BASE_DATASET, 'Sapi', '*.jpg'))] + \
             [f for f in glob.glob(os.path.join(BASE_DATASET, 'Sapi', '*.png'))]

print(f"Gambar BABI: {len(babi_files)}")
print(f"Gambar SAPI: {len(sapi_files)}")

# ─────────────────────────────────────────────────────
# 2. BUAT FOLDER TRAINING BERSIH
# ─────────────────────────────────────────────────────
CLEAN_DIR = 'training/dataset_clean'

# Bersihkan folder lama jika ada
if os.path.exists(CLEAN_DIR):
    shutil.rmtree(CLEAN_DIR)

for label in ['Babi', 'Sapi']:
    for split in ['train', 'val']:
        os.makedirs(os.path.join(CLEAN_DIR, split, label), exist_ok=True)

def copy_split(file_list, label, ratio=0.8):
    # Acak file agar pembagian adil
    np.random.shuffle(file_list)
    train_files = file_list[:int(len(file_list) * ratio)]
    val_files   = file_list[int(len(file_list) * ratio):]
    for f in train_files:
        shutil.copy(f, os.path.join(CLEAN_DIR, 'train', label, os.path.basename(f)))
    for f in val_files:
        shutil.copy(f, os.path.join(CLEAN_DIR, 'val',   label, os.path.basename(f)))
    return len(train_files), len(val_files)

tr_b, val_b = copy_split(babi_files, 'Babi')
tr_s, val_s = copy_split(sapi_files, 'Sapi')
print(f"Train  -> Babi: {tr_b}, Sapi: {tr_s}")
print(f"Val    -> Babi: {val_b}, Sapi: {val_s}")

# Hitung class weight untuk menangani imbalance dataset
total = tr_b + tr_s
w_babi = total / (2.0 * tr_b)  if tr_b else 1.0
w_sapi = total / (2.0 * tr_s)  if tr_s else 1.0
class_weight  = {0: w_babi, 1: w_sapi}
print(f"Class weights -> Babi: {w_babi:.2f}, Sapi: {w_sapi:.2f}")

# ─────────────────────────────────────────────────────
# 3. DATA GENERATOR
# ─────────────────────────────────────────────────────
IMG_SIZE = 299   # Xception default
BATCH    = 16

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    os.path.join(CLEAN_DIR, 'train'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode='binary',
    classes=['Babi', 'Sapi'],  # Babi=0, Sapi=1
    shuffle=True,
)
val_gen = val_datagen.flow_from_directory(
    os.path.join(CLEAN_DIR, 'val'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode='binary',
    classes=['Babi', 'Sapi'],
    shuffle=False,
)

print("Class mapping:", train_gen.class_indices)

# ─────────────────────────────────────────────────────
# 4. BANGUN MODEL (Xception + custom head)
# ─────────────────────────────────────────────────────
base = Xception(weights='imagenet', include_top=False,
                   input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False          # Phase 1: hanya latih head

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(64, activation='relu')(x)
out = Dense(1, activation='sigmoid')(x)

model = Model(base.input, out)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='binary_crossentropy', metrics=['accuracy'])

cb = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
]

# ─────────────────────────────────────────────────────
# 5. PHASE 1: LATIH HEAD (10 epoch)
# ─────────────────────────────────────────────────────
print("\n--- Phase 1: Training head ---")
model.fit(train_gen, epochs=10, validation_data=val_gen, callbacks=cb,
          class_weight=class_weight)

# ─────────────────────────────────────────────────────
# 6. PHASE 2: FINE-TUNE (unfreeze 30 layer terakhir)
# ─────────────────────────────────────────────────────
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),   # LR lebih kecil
              loss='binary_crossentropy', metrics=['accuracy'])

print("\n--- Phase 2: Fine-tuning ---")
model.fit(train_gen, epochs=20, validation_data=val_gen, callbacks=cb,
          class_weight=class_weight)

# ─────────────────────────────────────────────────────
# 7. EVALUASI & SIMPAN
# ─────────────────────────────────────────────────────
loss, acc = model.evaluate(val_gen)
print(f"\nValidasi Accuracy: {acc*100:.2f}%")

os.makedirs('models', exist_ok=True)
model.save('models/model_sapi_babi.h5')
print("Model disimpan di models/model_sapi_babi.h5")

# Simpan class order agar app.py tahu indeks
with open('models/class_indices.json', 'w') as f:
    json.dump(train_gen.class_indices, f)
print("Class indices disimpan di models/class_indices.json")
