
# pip install anvil-uplink
import anvil.server
anvil.server.connect("server_xxxxxxxxxxxxx")

from google.colab import drive
drive.mount('/content/drive')

import os
import cv2
import numpy as np
from keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

def LoadRiceImages(data_dir, class_labels, target_size=(224, 224)):  # Set target size to (224, 224)
    num_classes = len(class_labels)
    target_classes = np.eye(num_classes)
    X = []
    T = []
    for i in range(num_classes):
        class_dir = os.path.join(data_dir, class_labels[i])
        files = os.listdir(class_dir)
        for f in files:
            ff = f.lower()
            if ff.endswith('.jpg') or ff.endswith('.jpeg') or ff.endswith('.png'):
                file_path = os.path.join(class_dir, f)
                img = cv2.imread(file_path, 1)
                img = cv2.resize(img, target_size)  # Resize to target size
                img = np.asarray(img) / 255.0
                img = img.astype('float32')
                X.append(img)
                T.append(target_classes[i])
    X = np.array(X)
    T = np.array(T)
    X = X.astype('float32')
    T = T.astype('float32')
    return X, T

def RiceDiseaseModel(num_classes, input_shape=(224, 224, 3)):  # Set default input shape to (224, 224, 3)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)  # Use custom input shape
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def TrainRiceDiseaseModel(epochs, data_dir, class_labels, weights_file='rice_disease_weights.h5', validation_split=0.2):
    X, T = LoadRiceImages(data_dir, class_labels)
    num_classes = len(class_labels)
    input_shape = X.shape[1:]  # Get input shape from loaded images
    model = RiceDiseaseModel(num_classes, input_shape=input_shape)  # Pass input shape to model

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        validation_split=validation_split
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

    train_gen = datagen.flow(X, T, subset='training', batch_size=32)
    val_gen = datagen.flow(X, T, subset='validation', batch_size=32)

    history = model.fit(train_gen,
                        epochs=epochs,
                        validation_data=val_gen,
                        callbacks=[early_stopping, reduce_lr])

    model.save(weights_file)
    return model, history




# Main Program
data_dir = "/content/drive/MyDrive/DatasetRiceleafdisease"
class_labels = ["Bacterial leaf blight", "Brown spot", "Leaf smut"]
epochs = 30
weights_file = "rice_disease_weights.h5"

model, history = TrainRiceDiseaseModel(epochs, data_dir, class_labels, weights_file)
model.summary()


# Plot loss and accuracy
#plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
#plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()

# Print accuracy

final_accuracy = history.history['val_accuracy'][-1]
print(f"Final validation accuracy: {final_accuracy * 100:.2f}%")

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from PIL import Image
import gdown

model_path = "rice_disease_weights.h5"
model = tf.keras.models.load_model(model_path)

# Class labels for the disease prediction
class_labels = {0: 'Bacterial leaf blight', 1: 'Brown spot', 2: 'Leaf smut'}

def prediksi(img):
    img = cv2.resize(img, (224, 224))  # Resize the image
    img_array = np.expand_dims(img, axis=0) / 255.0  # Normalize

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)

    # Check the predicted class
    if predicted_class in class_labels:
        return class_labels[predicted_class]
    else:
        return 'Unknown class'

import anvil.media

@anvil.server.callable
def prediksipenyakitpadi(file):

    with anvil.media.TempFile(file) as f:
        img = np.array(Image.open(f))
    hasil = "hasil prediksi penyakit : " + prediksi(img)
    return hasil

anvil.server.wait_forever()