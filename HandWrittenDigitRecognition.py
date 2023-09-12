import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Loading the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizing the images to scale pixel values between 0 and 1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Defining the neural network model
model = tf.keras.models.Sequential([
    # Flatten layer to transform 2D image data into 1D
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    
    # Two dense layers with 128 units and ReLU activation
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    
    # Output layer with 10 units (for 10 digits) and softmax activation
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compiling the model with Adam optimizer and sparse categorical crossentropy loss
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training the model on the training dataset
model.fit(x_train, y_train, epochs=10, batch_size=128)

# Saving the trained model
model.save('handwritten.model')

# Uncomment the following line to load the trained model and be sure to comment the lines above as well
#model = tf.keras.models.load_model('handwritten.model')

# Evaluating the model on the test dataset
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

# Predicting digits from local images in 'digits' directory
image_number = 1
while os.path.isfile(f'digits/digit{image_number}.png'):
    try:
        # Reading the image and extracting the grayscale channel
        img = cv2.imread(f'digits/digit{image_number}.png', cv2.IMREAD_GRAYSCALE)
        
        # Resizing the image to match the input size of the model (28x28)
        img = cv2.resize(img, (28, 28))
        
        # Inverting the image colors
        img = np.invert(img)
        
        # Reshaping the image to fit the model's input shape
        img = img.reshape(1, 28, 28)
        
        # Predicting the digit
        prediction = model.predict(img)
        predicted_digit = np.argmax(prediction)
        
        print(f"This digit is probably a: {predicted_digit}")
        
        # Displaying the image
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()

    except Exception as e:
        print(f"Error reading image {image_number}: {e}")

    finally:
        image_number += 1
