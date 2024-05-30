import os
import cv2 as cv
import torch
import numpy as np
import mediapipe as mp

from tqdm import tqdm
from time import sleep

from glob import glob
from torch.utils.data import Dataset


class getData(Dataset):
    def __init__(self, folder='C:/Users/LENOVO/Downloads/archive/DATASET/TRAIN'):
        mp_pose = mp.solutions.pose
        self.folder = folder
        self.dataset = []

        classlist = os.listdir(folder)
        print(classlist)
        classint = np.eye(6)

        for i in tqdm(range(len(classlist)), mininterval=3, desc="Dataset Loading!"):
            for j in glob(folder + '/'+classlist[i] + '/*'):
                keypoints = []
                with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                    image = cv.imread(j)
                    image = cv.resize(image, (500, 500))
                    output_image, landmarks = self.detectPose(image, pose, display=False)
                    if landmarks!=[]:
                        keypoints.append(landmarks[0])
                        keypoints.extend(landmarks[11:17])
                        keypoints.extend(landmarks[23:31])
                if keypoints!=[]:
                    keypoints = self.normalize(keypoints)
                    self.dataset.append([keypoints, classint[i]])
            sleep(.1)


    def __getitem__(self, item):
        feature, label = self.dataset[item]
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.dataset)

    def normalize(self, x):
        keypoints = []
        for i in range(len(x)):
            for j in range(2):
                keypoints.append(x[i][j]/500)

        return keypoints

    def detectPose(self, image, pose, display=True):

        # Create a copy of the input image.
        output_image = image.copy()

        # Convert the image from BGR into RGB format.
        imageRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # Perform the Pose Detection.
        results = pose.process(imageRGB)

        # Retrieve the height and width of the input image.
        height, width, _ = image.shape

        # Initialize a list to store the detected landmarks.
        landmarks = []

        # Check if any landmarks are detected.
        if results.pose_landmarks:

            # Iterate over the detected landmarks.
            for landmark in results.pose_landmarks.landmark:
                # Append the landmark into the list.
                landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))


        return output_image, landmarks

if __name__=="__main__":
    data = getData()