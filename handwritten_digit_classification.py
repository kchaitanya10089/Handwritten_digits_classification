# Required Imports
import cv2
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model

# CONST
MODEL_PATH = './classification_model/mnist_cnn_classifier.h5'


# Function to load deep learning model from file
def load_classifier():
    try:
        classifier = load_model(MODEL_PATH)
        return classifier
    except Exception as identifier:
        print('[ERROR] ', identifier)


# Function to show output
def draw_output(image_display, pred):
    try:
        BLACK = [0, 0, 0]
        expanded_image = cv2.copyMakeBorder(
            image_display, 0, 0, 0, image_display.shape[0], cv2.BORDER_CONSTANT, value=BLACK)

        expanded_image = cv2.cvtColor(expanded_image, cv2.COLOR_GRAY2BGR)

        cv2.putText(expanded_image, str(pred), (152, 70),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 4, (0, 255, 0), 2)

        cv2.imshow('prediction', expanded_image)
    except Exception as identifier:
        print('[ERROR] ', identifier)


# Main function of program
def main():
    try:
        # Loading model in to memory
        classifier = load_classifier()
        if classifier is not None:

            # Display model summary
            classifier.summary()

            # Display model layer shape
            for layer in classifier.layers:
                print(layer.input_shape)

            # Generate ranodon images from test data and perdict the output of that image
            for i in range(0, 10):

                # Load dataset into the memory
                (x_train, y_train), (x_test, y_test) = mnist.load_data()

                # Generating image for prediction from test image dataset
                rand_index = np.random.randint(0, len(x_test))
                input_image = x_test[rand_index]

                # Changing shape of image as per model
                resized_image = input_image.reshape(1, 28, 28, 1)

                # Predicting results
                pred = classifier.predict_classes(
                    resized_image, 1, verbose=0)[0]
                print(pred)
                # Generate output image
                display_image = cv2.resize(
                    input_image, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

                # Displying output of prediction on image
                draw_output(display_image, pred)
                cv2.waitKey(0)

            cv2.destroyAllWindows()

    except Exception as identifier:
        print('[ERROR] ', identifier)


# Entry point of program
if __name__ == "__main__":
    main()
