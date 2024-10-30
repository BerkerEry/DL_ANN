import tensorflow as tf


import os

from tensorflow.python.ops.metrics_impl import precision

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
print("oneDNN optimizasyonları kapatıldı.")
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import  load_model, Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical, plot_model

import matplotlib.pyplot as plt
import numpy as np

import warnings
from warnings import filterwarnings

from unicodedata import category

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
filterwarnings("ignore")

# Mnist veri setinin yüklenmesi

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Eğitim Seti Boyutu:", x_train.shape, y_train.shape)

print("Test Seti Boyutu:", x_test.shape, y_test.shape)

num_labels = len(np.unique(y_train)) # y_train sayısı 10

# Veri Setinden Örnekler Gösterilmesi

plt.figure(figsize=(10,10)) # örnekler 10 10 boyutlarında olsun
plt.imshow(x_train[2], cmap ="gray")
#plt.show()

plt.figure(figsize=(10,10))
for n in range(0,10):
    ax = plt.subplot(5,5,n+1)
    plt.imshow(x_train[n], cmap="gray")
    plt.axis("off")
#plt.show()

def visualize_img(data, x=20):
    plt.figure(figsize=(10,10))
    for n in range(x):
        ax = plt.subplot(5, 5, n+1)
        plt.imshow(data[n], cmap="gray")
        plt.axis("off")

visualize_img(x_train,25)
#plt.show()

# RGB (0-255
# r: 250 g:0 b:250 mor, daha açık bir mor

print(x_train[2].shape)
print(x_train[2])
print("----")
print(x_train[2][10,10]) # görüntüdeki 10 10 pikselindeki rengi söyler
print(x_train[2][14,10])
print("----")
print(x_train[2].mean())
print(x_train[2].sum())
print("----")
print(x_train[2][14:20, 10:20])
print(x_train[2][14:20, 10:20].mean())
print("----")
def pixel_visualize(img):
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap="gray")
    width, height = img.shape

    threshold = img.max() / 2.5

    for x in range(width):
        for y in range(height):
            ax.annotate(str(round(img[x][y], 2)), xy=(y,x), color="white" if img[x][y]<threshold else "black")

pixel_visualize(x_train[2]) # resmimizin her pixelin ortalamasını alır

print(y_train[0:5]) # encoding, hangi sayı kaçıncı sırada ise o 1 değerini alır

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train[0:5])
print("----")

image_size = x_train.shape[1] # reshaping

print(image_size)

print(f"x_train boyutu: {x_train.shape}")
print(f"x_test boyutu: {x_test.shape}")

x_train = x_train.reshape(x_train.shape[0],28,28,1) # 784 tane bilgiyi 1 tutuyor
x_test = x_test.reshape(x_test.shape[0],28,28,1)

print(f"x_train boyutu: {x_train.shape}")
print(f"x_test boyutu: {x_test.shape}")

print("----")

# standardization

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

print(x_train)
print(x_test)

print("--------------------")

# modelleme, sinir ağı mimarisini tanımlamak

model = tf.keras.Sequential([Flatten(input_shape=(28,28,1)), Dense(units=128, activation="relu", name="layer1"),
                             Dense(units=num_labels, activation="softmax", name="output_layer")]) # Dense = hidden layer

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=[tf.keras.metrics.Precision(),
                       tf.keras.metrics.Recall(),
                       "accuracy"])
#print(model.summary())

model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test)) # 128 her iterasyonda 128 gözlem birimine odaklan

# Evaluation, Model başarısını değerlendirme

history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))

#############################
# Accuracy ve Loss Grafikleri
#############################

#----------- Grafik 1 Accuracy --------------

plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], color="b",label="Training Accuracy")
plt.plot(history.history["val_accuracy"],color="r",label="Validation Accuracy")
plt.legend(loc="lower right")
plt.xlabel("Epoch",fontsize=16)
plt.ylabel("Accuracy",fontsize=16)
plt.ylim([min(plt.ylim()),1])
plt.title("Eğitim ve Test Başarım Grafiği", fontsize=16)

#----------- Grafik 1 Loss --------------

plt.subplot(1,2,2)
plt.plot(history.history["loss"], color="b",label="Training Loss")
plt.plot(history.history["val_loss"],color="r",label="Validation Loss")
plt.legend(loc="upper right")
plt.xlabel("Epoch",fontsize=16)
plt.ylabel("Loss",fontsize=16)
plt.ylim([0,max(plt.ylim())])
plt.title("Eğitim ve Test Kayıp Grafiği", fontsize=16)
plt.show()

loss, precision, recall, acc = model.evaluate(x_test, y_test, verbose=False)
print("\nTest Accuracy: %.1f%%"% (100.0*acc))
print("\nTest Loss: %.1f%%"% (100.0*loss))
print("\nTest Precision: %.1f%%"% (100.0*precision)) # tahmin ettiklerimizin başarısı
print("\nTest Recall: %.1f%%"% (100.0*recall)) # gerçek değerlerin % kaçı doğru

# modelin kaydedilmesi ve tahmin için kullanılması

model.save("mnist_model.h5")

import random

random = random.randint(0, x_test.shape[0])

print(random)

test_image = x_test[random] # random sayıyı görüntü olarak alırız

print(y_test[random])
plt.imshow(test_image.reshape(28,28),cmap="gray")
plt.show()

test_data = x_test[random].reshape(1,28,28,1) # random sayısı, DL çıktısı olarak alırız

probability = model.predict(test_data)
print(probability)

predicted_classes = np.argmax(probability)
print(predicted_classes)

print(f"Tahmin Edilen Sınıf: {predicted_classes} \n")
print(f"Tahmin Edilen Sınıfın Olasılık Değeri: {(np.max(probability, axis = -1))[0]} \n")
print(f"Diğer Sınıfların Olasılık Değerleri: \n {probability}")

