import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from PIL import Image
import io

# --- Configuration ---
DATA_DIR = r'archive'
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
RANDOM_SEED = 42
VALIDATION_SPLIT = 0.2

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# --- Data Generators ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=VALIDATION_SPLIT
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    seed=RANDOM_SEED,
    shuffle=True  # Ensure shuffling for training
)

validation_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    seed=RANDOM_SEED,
    shuffle=False # No need to shuffle validation data
)

class_labels = list(train_generator.class_indices.keys())
num_classes = len(class_labels)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    seed=RANDOM_SEED
)

# --- Model Definition ---
def create_cnn_model(input_shape, num_classes):
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

input_shape = IMAGE_SIZE + (3,)
model = create_cnn_model(input_shape, num_classes)

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- Streamlit Interface ---
st.title("Image Classification with CNN")

# Sidebar for model training and evaluation
with st.sidebar:
    st.header("Model Training and Evaluation")
    if st.button("Train Model"):
        with st.spinner("Training the model..."):
            history = model.fit(
                train_generator,
                steps_per_epoch=train_generator.samples // BATCH_SIZE,
                epochs=EPOCHS,
                validation_data=validation_generator,
                validation_steps=validation_generator.samples // BATCH_SIZE,
                verbose=1
            )
            st.success("Model training complete!")

            loss, accuracy = model.evaluate(test_generator, verbose=0)
            st.write(f"**Test Loss:** {loss:.4f}")
            st.write(f"**Test Accuracy:** {accuracy:.4f}")

            predictions = model.predict(test_generator, verbose=0)
            predicted_classes = np.argmax(predictions, axis=1)
            true_classes = test_generator.classes

            report = classification_report(true_classes, predicted_classes, target_names=class_labels, zero_division=0)
            st.subheader("Classification Report")
            st.text(report)

            cm = confusion_matrix(true_classes, predicted_classes)
            plt.figure(figsize=(8, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_labels, yticklabels=class_labels)
            st.subheader("Confusion Matrix")
            st.pyplot(plt)

            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Train Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            st.subheader("Training History")
            st.pyplot(plt)

# Image Upload and Prediction
st.header("Image Prediction")
uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])

def predict_single_image(img_file, model, target_size, class_labels):
    try:
        img = Image.open(img_file).resize(target_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_label = class_labels[predicted_class_index]
        confidence = predictions[0][predicted_class_index]
        return predicted_class_label, confidence
    except Exception as e:
        return f"Error processing image: {e}", None

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    if st.button("Predict"):
        if 'model' in locals():  # Ensure the model is trained
            with st.spinner("Predicting..."):
                predicted_label, confidence = predict_single_image(uploaded_file, model, IMAGE_SIZE, class_labels)
                if predicted_label and confidence is not None:
                    st.write(f"**Predicted Class:** {predicted_label}")
                    st.write(f"**Confidence:** {confidence:.4f}")
                elif predicted_label:
                    st.error(f"Prediction Error: {predicted_label}")
        else:
            st.warning("Please train the model first in the sidebar.")
