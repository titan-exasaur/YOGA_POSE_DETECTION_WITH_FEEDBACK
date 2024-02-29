import mediapipe as mp
import numpy as np
import cv2

def check_pose(lst):
    if lst[28].visibility > 0.6 and lst[27].visibility > 0.6 and lst[15].visibility > 0.6 and lst[16].visibility > 0.6:
        return True
    return False

cap = cv2.VideoCapture(0)

name = input("Enter the name of the Asana or Feedback : ")

holistic = mp.solutions.pose
holis = holistic.Pose()
drawing = mp.solutions.drawing_utils

X = []
data_size = 0

while True:
    pose_landmarks = []

    _, frame = cap.read()

    frame = cv2.flip(frame, 1)

    results = holis.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks and check_pose(results.pose_landmarks.landmark):
        for i in results.pose_landmarks.landmark:
            pose_landmarks.append(i.x - results.pose_landmarks.landmark[0].x)
            pose_landmarks.append(i.y - results.pose_landmarks.landmark[0].y)

        X.append(pose_landmarks)
        data_size = data_size + 1

    else:
        cv2.putText(frame, "WARNING : Full Body Not Visible!!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    drawing.draw_landmarks(frame, results.pose_landmarks, holistic.POSE_CONNECTIONS)

    cv2.putText(frame, str(data_size), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("window", frame)

    if cv2.waitKey(1) == 27 or data_size > 180:
        cv2.destroyAllWindows()
        cap.release()
        break

np.save(f"{name}.npy", np.array(X))
print(np.array(X).shape)

