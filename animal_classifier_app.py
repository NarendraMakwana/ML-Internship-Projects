import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32
MODEL_PATH = "animal_classifier.h5"
DATASET_PATH = "dataset"

# Streamlit UI setup
st.set_page_config(page_title="Animal Classifier", layout="centered")
st.title("🐾 Animal Image Classifier App")

# Step 1: Train model (if not already trained)
if not os.path.exists(MODEL_PATH):
    st.info("Training model. Please wait...")

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    val_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(train_generator.num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy', mode='max')
    model.fit(train_generator, validation_data=val_generator, epochs=5, callbacks=[checkpoint])

    st.success("Model training complete!")
else:
    st.success("Pretrained model loaded.")

# Step 2: Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load class names from directory
class_names = sorted(os.listdir(DATASET_PATH))

# Step 3: Upload and predict
uploaded_file = st.file_uploader("📷 Upload an animal image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.markdown(f"### 🧠 Prediction: **{predicted_class}** ({confidence:.2f}%)")

    
