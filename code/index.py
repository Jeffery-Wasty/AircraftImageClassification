import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D
import os
​
main_dir = "../input/"
train_dir = "train/train_demo"
path = os.path.join(main_dir, train_dir)
​
X = []  # Training array
y = []  # Target array
​


def convert(category): return int(category == 'dog')


​
for p in os.listdir(path):
    category = p.split(".")[0]
    category = convert(category)
    img_array = cv2.imread(os.path.join(path, p), cv2.IMREAD_GRAYSCALE)
    new_img_array = cv2.resize(img_array, dsize=(80, 80))
    X.append(new_img_array)
    y.append(category)
​
plt.imshow(X[0], cmap="gray")
plt.show()
​
X = np.array(X).reshape(-1, 80, 80, 1)
y = np.array(y)
​
# Normalize data
X = X/255.0
​
model = Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=X.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Add another:
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
​
model.add(Flatten())
model.add(Dense(64, activation='relu'))
# Add a softmax layer with 10 output units:
model.add(Dense(1, activation='sigmoid'))
​
model.compile(optimizer="adam",
              loss='binary_crossentropy',
              metrics=['accuracy'])
​
model.summary()
​
input()
​
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
​
# Testing
test_dir = "test1/test1"
path = os.path.join(main_dir, test_dir)
image_to_test = input("Enter image number to test (0 to 12499): ")
​
while (image_to_test.isdigit()):
    X_test = []
    img_array = cv2.imread(os.path.join(path, os.listdir(
        path)[int(image_to_test)]), cv2.IMREAD_GRAYSCALE)
    new_img_array = cv2.resize(img_array, dsize=(80, 80))
​
   X_test.append(new_img_array)
    X_test = np.array(X_test).reshape(-1, 80, 80, 1)
    X_test = X_test/255
​
   prediction = model.predict(X_test)
​
   if (prediction[0][0] < 0.5):
        result = "Cat - confidence: {}%".format(
            (0.5 - prediction[0][0]) / 0.5 * 100)
    else:
        result = "Dog - confidence: {}%".format(
            (prediction[0][0] - 0.5) / 0.5 * 100)
​
   print(result)
    fig = plt.figure(result)
    plt.imshow(new_img_array, cmap="gray")
    plt.show()
    image_to_test = input("Enter image number to test (0 to 12499): ")
