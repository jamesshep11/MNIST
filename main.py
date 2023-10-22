import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load model if it exists
try:
    model = tf.keras.models.load_model("MNIST.model")
except:
    # Load training data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # normalise data
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    # Define model
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train and save model
    model.fit(x_train, y_train, epochs=3)
    model.save('MNIST2.model')

    # Evaluate model
    loss, accuracy = model.evaluate(x_test, y_test)

    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")


# Use model to evaluate your own images
image_num = 1

# Loop through image files
while os.path.isfile(f"digits/digit{image_num}.png"):
    try:
        # load image file and format to usable array vals
        img = cv2.imread(f"digits/digit{image_num}.png")[:,:,0]
        img = np.invert(np.array([img]))
        
        # process image through model
        prediction = model.predict(img)
        
        # Show results
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.title(label=f"This digit is probably a {np.argmax(prediction)}")
        plt.show()
    except:
        print("Error")
    finally:
        image_num += 1