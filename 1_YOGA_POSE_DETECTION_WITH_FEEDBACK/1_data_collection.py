import mediapipe as mp
import numpy as np
import cv2


# Function to check if the pose is within frame
def check_pose(lst):
    if lst[28].visibility > 0.6 and lst[27].visibility > 0.6 and lst[15].visibility > 0.6 and lst[16].visibility > 0.6:
        return True
    return False


# Open the video capture device (webcam)
cap = cv2.VideoCapture(0)

# Get the name of the Asana or Feedback from the user
name = input("Enter the name of the Asana or Feedback : ")

# Initialize the pose detection module
holistic = mp.solutions.pose
holis = holistic.Pose()
drawing = mp.solutions.drawing_utils

# Initialize an empty list to store the pose landmarks
X = []

# Initialize the variable to keep track of the number of data points collected
data_size = 0

# Main loop to capture video frames and detect poses
while True:
    pose_landmarks = []

    # Capture a frame from the video feed
    _, frame = cap.read()

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Process the frame to detect the pose landmarks
    results = holis.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # If pose landmarks are detected and the pose is within frame, add the pose landmarks to the list
    if results.pose_landmarks and check_pose(results.pose_landmarks.landmark):
        for i in results.pose_landmarks.landmark:
            pose_landmarks.append(i.x - results.pose_landmarks.landmark[0].x)
            pose_landmarks.append(i.y - results.pose_landmarks.landmark[0].y)

        # Add the pose landmarks to the list of X
        X.append(pose_landmarks)

        # Increment the data size
        data_size = data_size + 1

    # If the pose is not within frame, display a warning message
    else:
        cv2.putText(frame, "WARNING : Full Body Not Visible!!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Draw the pose landmarks on the frame
    drawing.draw_landmarks(frame, results.pose_landmarks, holistic.POSE_CONNECTIONS)

    # Display the number of data points collected on the frame
    cv2.putText(frame, str(data_size), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame in a window
    cv2.imshow("window", frame)

    # Exit the loop if the 'Esc' key is pressed or the number of data points collected exceeds 180
    if cv2.waitKey(1) == 27 or data_size > 180:
        cv2.destroyAllWindows()
        cap.release()
        break

# Save the collected pose landmarks to a numpy file
np.save(f"{name}.npy", np.array(X))

# Print the shape of the saved numpy array
print(np.array(X).shape)
