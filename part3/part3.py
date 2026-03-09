import os
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

import kagglehub
from pathlib import Path

DATA_ROOT = Path(kagglehub.dataset_download("harisudhan411/hand-navigation-landmarks")) / "HandNavigation"
print("Path to dataset files:", DATA_ROOT)

GESTURE_CLASSES = ["up", "down", "left", "right"]
IMAGE_SIZE = 64

os.makedirs("imgs", exist_ok=True)

def load_dataset():
    samples = []

    for split_name in ["Train", "Validation", "Test"]:
        for gesture in GESTURE_CLASSES:
            directory = Path(DATA_ROOT) / split_name / gesture
            if not directory.exists():
                continue

            for img_path in directory.glob("*.png"):
                image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue

                image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
                vector = image.flatten() / 255.0
                samples.append(np.concatenate([vector, [gesture]]))

    feature_count = IMAGE_SIZE * IMAGE_SIZE
    columns = [f"pixel_{i}" for i in range(feature_count)] + ["label"]

    df = pd.DataFrame(samples, columns=columns)
    return df


def prepare_data(df):
    X = df.drop("label", axis=1).values.astype("float32")
    labels = df["label"].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    X = X.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_val, y_train, y_val, encoder


def create_augmentation(X_train):
    generator = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False
    )
    generator.fit(X_train)
    return generator


def build_model(num_classes=4):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(64, 64, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])

    return model


def train_model(model, generator, X_train, y_train, X_val, y_val):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, monitor="val_accuracy"),
        ReduceLROnPlateau(factor=0.5, patience=5, monitor="val_loss", min_lr=1e-6),
        ModelCheckpoint("best_model.keras", save_best_only=True, monitor="val_accuracy")
    ]

    history = model.fit(
        generator.flow(X_train, y_train, batch_size=32),
        validation_data=(X_val, y_val),
        epochs=50,
        callbacks=callbacks
    )

    return history


def evaluate_model(model, X_val, y_val, encoder):
    loss, accuracy = model.evaluate(X_val, y_val)

    predictions = np.argmax(model.predict(X_val), axis=1)
    print(classification_report(y_val, predictions, target_names=encoder.classes_))

    matrix = confusion_matrix(y_val, predictions)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=encoder.classes_,
        yticklabels=encoder.classes_
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("imgs/confusion_matrix.png")
    plt.close()
    return accuracy


def plot_training(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["accuracy"], label="Train")
    ax1.plot(history.history["val_accuracy"], label="Val")
    ax1.set_title("Accuracy")
    ax1.legend()

    ax2.plot(history.history["loss"], label="Train")
    ax2.plot(history.history["val_loss"], label="Val")
    ax2.set_title("Loss")
    ax2.legend()

    plt.savefig("imgs/training_curves.png")
    plt.close()


def save_label_map(encoder):
    label_map = {int(i): str(cls) for i, cls in enumerate(encoder.classes_)}

    with open("label_map.json", "w") as f:
        json.dump(label_map, f)

    return label_map


def convert_to_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open("gesture_model.tflite", "wb") as f:
        f.write(tflite_model)

    return tflite_model


def verify_tflite(X_val, y_val, label_map):
    interpreter = tf.lite.Interpreter(model_path="gesture_model.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    correct = 0

    for true_label in range(4):
        indices = np.where(y_val == true_label)[0]
        inp = X_val[indices[0]].reshape(1, 64, 64, 1).astype(np.float32)

        interpreter.set_tensor(input_details[0]["index"], inp)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]["index"])[0]
        pred_idx = int(np.argmax(output))

        if pred_idx == true_label:
            correct += 1

    return correct


def export_files(label_map):
    with open("label_map.json", "w") as f:
        json.dump(label_map, f)


def main():
    df = load_dataset()

    X_train, X_val, y_train, y_val, encoder = prepare_data(df)

    datagen = create_augmentation(X_train)

    model = build_model()

    history = train_model(model, datagen, X_train, y_train, X_val, y_val)

    model = tf.keras.models.load_model("best_model.keras")

    evaluate_model(model, X_val, y_val, encoder)

    plot_training(history)

    label_map = save_label_map(encoder)

    convert_to_tflite(model)

    verify_tflite(X_val, y_val, label_map)

    export_files(label_map)


if __name__ == "__main__":
    main()