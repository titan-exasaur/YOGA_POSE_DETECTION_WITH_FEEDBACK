import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model

# Function to check if the pose is within frame
def inFrame(lst):
    if lst[28].visibility > 0.6 and lst[27].visibility > 0.6 and lst[15].visibility > 0.6 and lst[16].visibility > 0.6:
        return True
    return False

# Load the trained model and class labels
model = load_model("trained_model.h5")
label_names = np.load("class_labels.npy")

# Initialize the pose detection module
holistic = mp.solutions.pose
holis = holistic.Pose()
drawing = mp.solutions.drawing_utils

# Open the video capture device (webcam)
cap = cv2.VideoCapture(0)

# Main loop to capture video frames and detect poses
while True:
    lst = []

    # Capture a frame from the video feed
    _, frm = cap.read()

    # Flip the frame horizontally
    frm = cv2.flip(frm, 1)

    # Process the frame to detect the pose landmarks
    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    # Apply blur to the frame
    frm = cv2.blur(frm, (4, 4))

    # If pose landmarks are detected and the pose is within frame, make a prediction
    if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
        for i in res.pose_landmarks.landmark:
            lst.append(i.x - res.pose_landmarks.landmark[0].x)
            lst.append(i.y - res.pose_landmarks.landmark[0].y)

        # Convert the pose landmarks to a numpy array and reshape it
        lst = np.array(lst).reshape(1, -1)

        # Make a prediction using the trained model
        p = model.predict(lst)

        # Get the predicted label
        pred = label_names[np.argmax(p)]

        # If the prediction confidence is high enough, display the predicted label
        if p[0][np.argmax(p)] > 0.75:
            cv2.putText(frm, pred, (180, 180), cv2.FONT_ITALIC, 1.3, (0, 255, 0), 2)
        else:
            cv2.putText(frm, "Asana is either wrong or not trained", (100, 180), cv2.FONT_ITALIC, 1.8, (0, 0, 255), 3)
    else:
        cv2.putText(frm, "WARNING : FULL BODY NOT VISIBLE", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    # Draw the pose landmarks on the frame
    drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS,
                           connection_drawing_spec=drawing.DrawingSpec(color=(255, 255, 255), thickness=6),
                           landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3, thickness=3))

    # Display the frame in a window
    cv2.imshow("window", cv2.resize(frm, (720, 480)))

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        cap.release()
        break
