This project is a real-time pose detection and recognition system using the MediaPipe library and a pre-trained Keras neural network. The system is divided into three main files:

data_collection.py: This file is used to collect training data for the pose recognition model. It captures video frames from a webcam, detects the pose using the MediaPipe library, and saves the pose landmarks to a numpy file for later use.

model_training.py: This file trains a neural network model using the collected pose landmarks. It uses the Keras library to build and train a simple feedforward neural network with two hidden layers. The model is trained to recognize different yoga poses based on the pose landmarks.

inference.py: This file is the main application that uses the trained model for real-time inference. It captures video frames from a webcam, detects the pose using the MediaPipe library, and feeds the pose landmarks into the trained model to recognize the pose. The recognized pose is displayed on the video frame in real-time.

The project can be used to learn and recognize different yoga poses in real-time, which can be helpful for yoga practitioners to improve their form and technique.
