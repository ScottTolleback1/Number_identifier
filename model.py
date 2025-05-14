import os
import numpy as np
import cv2
import struct
import tensorflow as tf
import random

def read_uint32(file):
    return struct.unpack('>I', file.read(4))[0]

def read_mnist_labels(filename):
    with open(filename, 'rb') as file:
        magic = read_uint32(file)
        if magic != 0x00000801:
            print(f"Invalid magic number in label file: {hex(magic)}")
            return []
        num_labels = read_uint32(file)
        labels = np.frombuffer(file.read(num_labels), dtype=np.uint8)
    return labels

def read_mnist_images(filename):
    with open(filename, 'rb') as file:
        magic = read_uint32(file)
        num_images = read_uint32(file)
        num_rows = read_uint32(file)
        num_cols = read_uint32(file)
        images = np.frombuffer(file.read(num_images * num_rows * num_cols), dtype=np.uint8)
        images = images.reshape((num_images, num_rows, num_cols))
    return images

def create_deeper_cnn_model(input_shape=(28, 28, 1)):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def load_model_or_train(model_path, train_images, train_labels, test_images, test_labels):
    if os.path.exists(model_path):
        print("Loading model from file...")
        model = tf.keras.models.load_model(model_path)
    else:
        print("Creating and training new model...")
        model = create_deeper_cnn_model()
        model.fit(train_images, train_labels, epochs=5, batch_size=32, validation_data=(test_images, test_labels))
        model.save(model_path)
        print("Model saved to disk.")
    return model

def preprocess_image(png_path):
    img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image could not be loaded.")
    if img.shape != (28, 28):
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img = 255 - img
    return img.astype("float32") / 255.0

def predict_and_display(img, model):
    input_img = img.reshape(1, 28, 28, 1)
    prediction = model.predict(input_img)
    predicted_label = np.argmax(prediction)
    print(f"Predicted label: {predicted_label}")

    enlarged = cv2.resize(img, None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
    cv2.putText(enlarged, f"Predicted: {predicted_label}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2)
    cv2.imshow("Normalized Digit", enlarged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def evaluate_model(model, test_images, test_labels):
    loss, accuracy = model.evaluate(test_images, test_labels)
    print(f"Validation accuracy: {accuracy:.4f}")

def display_misclassified(model, test_images, test_labels):
    predictions = model.predict(test_images)
    for i, (image, label) in enumerate(zip(test_images, test_labels)):
        predicted = np.argmax(predictions[i])
        if predicted != label:
            print(f"Image {i}: Correct label: {label}, Predicted label: {predicted}")
            enlarged = cv2.resize(image, None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
            cv2.putText(enlarged, f"Correct: {label}, Predicted: {predicted}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2)
            cv2.imshow(f"Misclassified {i}", enlarged)
            cv2.waitKey(0)
            cv2.destroyWindow(f"Misclassified {i}")

def display_random_samples(model, test_images, test_labels):
    predictions = model.predict(test_images)
    indices = random.sample(range(len(test_images)), 10)
    for i in indices:
        image = test_images[i]
        label = test_labels[i]
        predicted = np.argmax(predictions[i])
        print(f"Image {i}: Correct label: {label}, Predicted label: {predicted}")
        enlarged = cv2.resize(image, None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
        cv2.putText(enlarged, f"Correct: {label}, Predicted: {predicted}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2)
        cv2.imshow(f"Sample {i}", enlarged)
        cv2.waitKey(0)
        cv2.destroyWindow(f"Sample {i}")

def main():
    image_file = 'mnist/train-images.idx3-ubyte'
    label_file = 'mnist/train-labels.idx1-ubyte'
    test_image_file = 'mnist/t10k-images.idx3-ubyte'
    test_label_file = 'mnist/t10k-labels.idx1-ubyte'
    model_path = 'mnist_model.h5'

    train_images = read_mnist_images(image_file)
    train_labels = read_mnist_labels(label_file)
    test_images = read_mnist_images(test_image_file)
    test_labels = read_mnist_labels(test_label_file)

    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0
    train_images = train_images.reshape((-1, 28, 28, 1))
    test_images = test_images.reshape((-1, 28, 28, 1))

    model = load_model_or_train(model_path, train_images, train_labels, test_images, test_labels)

    evaluate_model(model, test_images, test_labels)

    user_image_path = 'digit.png'
    try:
        img = preprocess_image(user_image_path)
        predict_and_display(img, model)
    except FileNotFoundError as e:
        print(e)

    # display_misclassified(model, test_images, test_labels)
    # display_random_samples(model, test_images, test_labels)

if __name__ == "__main__":
    main()
