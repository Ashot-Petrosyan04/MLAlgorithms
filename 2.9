import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32')  / 255.0

x_train_flat = x_train.reshape(-1, 28*28)
x_test_flat  = x_test.reshape(-1, 28*28)

x_train_cnn = x_train.reshape(-1, 28, 28, 1)
x_test_cnn  = x_test.reshape(-1, 28, 28, 1)

y_train_ohe = to_categorical(y_train, 10)
y_test_ohe  = to_categorical(y_test, 10)

def build_dnn():
    model = models.Sequential([
        layers.Input(shape=(28*28,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

dnn = build_dnn()
print("Training DNN...")
dnn.fit(x_train_flat, y_train_ohe,
        validation_split=0.1,
        epochs=10, batch_size=128, verbose=2)

dnn_eval = dnn.evaluate(x_test_flat, y_test_ohe, verbose=0)
print(f"DNN Test accuracy: {dnn_eval[1]*100:.2f}%\n")

def build_cnn():
    model = models.Sequential([
        layers.Input(shape=(28,28,1)),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

cnn = build_cnn()
print("Training CNN...")
cnn.fit(x_train_cnn, y_train_ohe,
        validation_split=0.1,
        epochs=10, batch_size=128, verbose=2)

cnn_eval = cnn.evaluate(x_test_cnn, y_test_ohe, verbose=0)
print(f"CNN Test accuracy: {cnn_eval[1]*100:.2f}%\n")


print("Summary:")
print(f"  DNN accuracy: {dnn_eval[1]*100:.2f}%")
print(f"  CNN accuracy: {cnn_eval[1]*100:.2f}%")
