# MNIST Handwritten Digit Classification using CNN

This project implements a Convolutional Neural Network (CNN) using Keras to classify handwritten digits from the MNIST dataset. The MNIST dataset is a classic benchmark dataset in the field of machine learning and computer vision, consisting of grayscale images of handwritten digits (0–9).

## 📌 Requirements

* Python 3.7+
* TensorFlow / Keras
* NumPy
* Matplotlib

You can install the dependencies using:

```bash
pip install numpy matplotlib tensorflow
```

## 📅 Dataset

The project uses the MNIST dataset, which is directly loaded using:

```python
from keras.datasets import mnist
```

## 🚀 Steps Performed

1. **Load Dataset**

```python
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

2. **Visualize Sample Digits**

```python
import matplotlib.pyplot as plt
plt.imshow(x_train[0])
plt.show()
```

3. **Preprocess the Data**

   * Normalize pixel values to the range \[0, 1]
   * Expand dimensions for CNN input
   * One-hot encode labels

```python
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
```

4. **Build CNN Model**

```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))
```

5. **Compile and Train the Model**

```python
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, validation_split=0.3)
```

6. **Evaluate and Predict**

```python
predictions = model.predict(x_test)
y_pred = np.argmax(predictions, axis=1)
```

7. **Visualize Predictions**

```python
plt.title(f"Predicted: {y_pred[0]}")
plt.imshow(x_test[0].squeeze(), cmap='gray')
plt.show()
```

## 📈 Results

* Final validation accuracy: \~98.9%
* Model performs very well on handwritten digit recognition tasks.

## 📌 Notes

* You can experiment by adjusting the number of layers, filters, or dropout rate.
* Try training on fewer epochs or with different optimizers to see changes in performance.
