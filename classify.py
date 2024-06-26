# -*- coding: utf-8 -*-
"""Untitled2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1LX1jTRG0zQknsFZSZ3sqbQt163wc9Wam
"""

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

train_labels[:5]

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history=model.fit(train_images, # данные для обучения
          train_labels, #
          epochs=10,    # кол-во эпох обучения
          validation_data=(test_images, test_labels)) # данные для проверки

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')

plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

import matplotlib.pyplot as plt1
plt1.plot(history.history['accuracy'])
plt1.plot(history.history['val_accuracy'])
plt.title('Model accuracy')

plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

from google.colab import drive
drive.mount("/content/drive/")

model.save('NeuroModels/onemorenumb.h5') #Сохраняем модель

# Как загрузить ранее сохраненную модель с весами с гугл-диска
from keras.models import load_model
#model2 = load_model('SavedModels/One_layer_model.h5')
model2 = load_model('NeuroModels/onemorenumb.h5')

import random
from PIL import Image, ImageDraw #Подключим необходимые библиотеки.
import matplotlib.pyplot as plt

digit_classes=['Цифра 0','Цифра 1','Цифра 2','Цифра 3','Цифра 4',
               'Цифра 5','Цифра 6','Цифра 7','Цифра 8','Цифра 9']

def resize_to_gray_pic(f_name):
  img = Image.open(f_name) #Открываем изображение.
  img = img.resize((28, 28), Image.ANTIALIAS) # приводм к размеру 28x28
  width = img.size[0] #Определяем ширину.
  height = img.size[1] #Определяем высоту.

  draw = ImageDraw.Draw(img) #Создаем инструмент для рисования.
  pix = img.load() #Выгружаем значения пикселей.

  for i in range(width):
    for j in range(height):
      r = pix[i, j][0]
      g = pix[i, j][1]
      b = pix[i, j][2]
      S = (r + g + b) // 3 # усредняем пикселы к серому
      draw.point((i, j), (S, S,S))
  return img

file_name='drive/MyDrive/Pics/45.jpg'

from keras import backend as keras_backend #импортируем настройки керас
import numpy as np
image=resize_to_gray_pic(file_name)

plt.imshow(image) # выводим кратинку в исходном виде
plt.show()

image = keras_backend.cast_to_floatx(image) #переводим в вещественные значения
image /= 255.0 #нормализуем значения
image=np.delete(image, 0, axis = 2) #удаляем повторяющуюся размерность
image=np.delete(image, 0, axis = 2) #удаляем еще повторяющуюся размерность
image = np.reshape(image, [1, 28, 28]) #приводим масив к вектору 784 эл
#ynew = model.predict_classes(i) - старый синтаксис
ynew_classes=np.argmax(model.predict(image), axis=-1)
ynew_prob = model.predict(image)

print(digit_classes[ynew_classes[0]], ", Вероятность - %.4f" % ynew_prob[0][ynew_classes[0]])
print(ynew_prob[0])