import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import layers, models

DATASET_DIR = "skin_dataset_cleaned"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

train_ds = image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# -------------------
# Data Augmentation
# -------------------
data_aug = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomBrightness(0.2),
])

# -------------------
# TRANSFER LEARNING MODEL
# -------------------
base = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)

base.trainable = False  # freeze backbone

model = models.Sequential([
    data_aug,
    layers.Rescaling(1./255),

    base,
    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),

    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------
# Callbacks
# -------------------
checkpoint = ModelCheckpoint("best_skin_model.keras", save_best_only=True, monitor="val_accuracy")
early_stop = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.3)

# -------------------
# Train
# -------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

model.save("skin_model_final.keras")
print("Model Saved!")


'''import os
import shutil

RAW_DIR = "archive (5)/files"   # your original
OUT_DIR = "skin_dataset_cleaned"

os.makedirs(OUT_DIR, exist_ok=True)

classes = ["acne", "bags", "redness"]

for cls in classes:
    src_path = os.path.join(RAW_DIR, cls)
    dst_path = os.path.join(OUT_DIR, cls)
    os.makedirs(dst_path, exist_ok=True)

    for person_folder in os.listdir(src_path):
        person_path = os.path.join(src_path, person_folder)

        if not os.path.isdir(person_path):
            continue

        for img in os.listdir(person_path):
            src_img = os.path.join(person_path, img)
            if not img.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            new_name = f"{person_folder}_{img}"
            dst_img = os.path.join(dst_path, new_name)

            shutil.copy(src_img, dst_img)

print("Dataset prepared successfully in skin_dataset_cleaned/")
'''