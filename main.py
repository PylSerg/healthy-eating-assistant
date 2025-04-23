from tkinter import Tk, messagebox
from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("model/keras_Model.h5", compile=False)

# Load the labels
class_names = open("model/labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

# Create variable to store the class name and confidence score
# of the predicted class
finally_class_name = ""
finally_confidence_score = 0

# Function to predict the class of the image
def predict():
    global finally_class_name, finally_confidence_score
    
    for i in range(10):
        # Grab the webcamera's image.
        ret, image = camera.read()

        # Resize the raw image into (224-height,224-width) pixels
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

        # Show the image in a window
        cv2.imshow("Webcam Image", image)

        # Make the image a numpy array and reshape it to the models input shape.
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

        # Normalize the image array
        image = (image / 127.5) - 1

        # Predicts the model
        prediction = model.predict(image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Write prediction and confidence score
        current_class_name = class_name[2:]
        current_confidence_score = str(np.round(confidence_score * 100))[:-2]
        
        # Write the current class name and confidence score on the image if the confidence score is greater than 95%
        if int(current_confidence_score) > 95:
            finally_class_name = current_class_name
            finally_confidence_score = current_confidence_score

        # Listen to the keyboard for presses.
        keyboard_input = cv2.waitKey(1)

        # 27 is the ASCII for the esc key on your keyboard.
        if keyboard_input == 27:
            break

# Retry prediction if predict() function fails to predict the class name     
while True:
    predict()
    
    if finally_class_name != "":
        break   
    
print(f"\n\nClass: {finally_class_name} \nConfidence Score: {finally_confidence_score}%\n\n")

camera.release()
cv2.destroyAllWindows()

def main():
    try:
        root = Tk()
        root.title("Health Eating Assistant")
        root.mainloop()
    except Exception as e:
        print(f"\n\nError: {str(e)}\n\n")
        messagebox.showerror("ERROR", f"Error: {str(e)}")

if __name__ == "__main__":
    main()