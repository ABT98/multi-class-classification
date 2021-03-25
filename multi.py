# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 09:13:04 2020

@author: abt21
"""
#Kütüphaneler
import tensorflow as tf
from keras import applications
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Flatten, Dropout
from keras import optimizers
import numpy as np
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator

#GPU Kullanmak için
tf.config.list_physical_devices('GPU')
tf.test.is_gpu_available()
tf.config.experimental.list_physical_devices(device_type=None)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

#Veri Ön İşleme(Train verileri için)
training_set = ImageDataGenerator(
        rescale=1./255,            
        shear_range=0.2,      
        zoom_range=0.2,    
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)  

#Görüntü boyutu
goruntu_genislik, goruntu_yukseklik = 299, 299
# Eğitim verisinden geçiş sayısı (tur)
epochs = 1
#Aynı anda işlenen görüntü sayısı
batch_size = 700
#Eğitim verisinin lokasyonu
egitim_seti_dizin = 'veriseti/egitim' 
#Doğrulama verisinin lokasyonu
dogrulama_seti_dizin = 'veriseti/dogrulama' 


from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
  rescale = 1./255
  )
training_set = train_datagen.flow_from_directory(
  egitim_seti_dizin,
  target_size = (150, 150),
  batch_size = batch_size,
  classes=["DMO","KN","NR","DR"])

validation_set = train_datagen.flow_from_directory(
  dogrulama_seti_dizin,
  target_size = (150, 150),
  batch_size = batch_size,
  classes=["DMO","KN","NR","DR"])

deneme_model= InceptionV3()
print(deneme_model.summary())
temel_model = applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(goruntu_genislik, goruntu_yukseklik, 3))
siniflandirici = Sequential()
siniflandirici.add(Conv2D(32, (3, 3), input_shape = (150, 150, 3), activation = 'relu'))
siniflandirici.add(MaxPooling2D(pool_size = (4, 4)))
siniflandirici.add(Conv2D(512, (3, 3), activation = 'relu'))
siniflandirici.add(MaxPooling2D(pool_size = (3, 3)))
siniflandirici.add(Conv2D(32, (3, 3), activation = 'relu'))
siniflandirici.add(MaxPooling2D(pool_size = (3, 3)))
siniflandirici.add(Dropout(0.2))
siniflandirici.add(Flatten())
siniflandirici.add(Dense(units = 128, activation = 'relu'))
siniflandirici.add(Dense(units = 4, activation = 'sigmoid'))
siniflandirici.summary()
print(siniflandirici.metrics_names)

siniflandirici.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])


gecmis=siniflandirici.fit_generator(
  training_set,
  steps_per_epoch = 83484/100,
  epochs = epochs,
  validation_data=validation_set,
  validation_steps=10)

siniflandirici.evaluate_generator(validation_set, steps = 10)

"""Grafik çizme"""

import matplotlib.pyplot as plt

print(gecmis.history.keys())

plt.figure()
plt.plot(gecmis.history['accuracy'], 'orange', label='Egitim Dogruluk')
plt.plot(gecmis.history['val_accuracy'], 'blue', label='Dogrulama Dogruluk')
plt.plot(gecmis.history['loss'], 'red', label='Egitim Kayıp')
plt.plot(gecmis.history['val_loss'], 'green', label='Dogrulama Kayıp')
plt.legend()
plt.show()
