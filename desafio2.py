import keras
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import pandas as pd
import numpy as np
import zipfile
import os
import matplotlib.pyplot as plt


# Define the path to the zip file and the extraction directory
zip_file_path = 'dataset.zip'
extract_dir = 'louisiana_images'

# Create the extraction directory if it doesn't exist
if not os.path.exists(extract_dir):
    os.makedirs(extract_dir)

# Unzip the file
print(f"Descompactando {zip_file_path} para {extract_dir}...")
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
print("Descompactação concluída.")


import tensorflow as tf
import os

image_size = (512, 360)
batch_size = 32

def preprocess_image_py(image_path_tensor, label):
    image_path = image_path_tensor.numpy().decode('utf-8')
    try:
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=3)
        if img.shape.rank == 0:  # Verifica se a imagem está vazia
            raise ValueError(f"imagem vazia")
        img = tf.image.resize(img, image_size)
        img = tf.cast(img, tf.float32) / 255.0  # Normaliza para o intervalo [0, 1]
        return img, label
    except Exception as e:
        print(f"Erro ao processar a imagem {image_path}: {e}")
        # Retorna uma imagem preta preenchida com zeros como substituta
        return tf.zeros((*image_size, 3), dtype=tf.float32), label

def preprocess_image(image_path, label):
    processed_image, processed_label = tf.py_function(
        preprocess_image_py,
        inp=[image_path, label],
        Tout=[tf.float32, label.dtype]
    )
    processed_image.set_shape([*image_size, 3])
    processed_label.set_shape([])
    return processed_image, processed_label

# --- Criar Dataset de Treino ---
print("Criando dataset de treino...")
train_image_filenames = train_df['Image ID'].values
train_labels = train_df['Flooded'].values

train_image_paths = [os.path.join(extract_dir, 'train', fname) for fname in train_image_filenames]

train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))

train_dataset = train_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(buffer_size=len(train_image_paths)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# --- Criar Dataset de Teste ---
print("Criando dataset de teste...")
test_image_filenames = test_df['Image ID'].values
test_labels = test_df['Flooded'].values

test_image_paths = [os.path.join(extract_dir, 'test', fname) for fname in test_image_filenames]

test_dataset = tf.data.Dataset.from_tensor_slices((test_image_paths, test_labels))
test_dataset = test_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE) # Não embaralha o conjunto de teste

print("Datasets de treino e teste criados com sucesso.")
print(f"Número de amostras de treino: {len(train_image_paths)}")
print(f"Número de amostras de teste: {len(test_image_paths)}")

# Opcional: Verificar um batch do dataset de treino
for images, labels in train_dataset.take(1):
    print(f"Formato do batch de imagens de treino: {images.shape}")
    print(f"Formato do batch de labels de treino: {labels.shape}")
    print(f"Tipo de dados do batch de imagens de treino: {images.dtype}")
    print(f"Tipo de dados do batch de labels de treino: {labels.dtype}")
    break


from keras.applications import Xception, EfficientNetB4
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import Adam

base_model = Xception(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=(image_size[0], image_size[1], 3),
    name="xception",
)

eff_b4 = EfficientNetB4(
    
) 


# TODO: Congelar as camadas do modelo base para não serem treinadas


# TODO: Adiciona camadas para classificação binária

# Cria o modelo final
model = Model(inputs=base_model.input, outputs=output_layer)

# Compila o modelo
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Exibe o resumo do modelo
model.summary()

print("Modelo de Transfer Learning criado e compilado com sucesso!")

