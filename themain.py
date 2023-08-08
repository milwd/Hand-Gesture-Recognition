#!/usr/bin/env python

import csv
import copy

import cv2 as cv
import numpy as np
import mediapipe as mp


import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.neural_network import MLPClassifier
import pickle

from collections import deque

confMatrix = [] #TP , FN, FP , TN


class CvFpsCalc(object):
    def __init__(self, buffer_len=1):
        self._start_tick = cv.getTickCount()
        self._freq = 1000.0 / cv.getTickFrequency()
        self._difftimes = deque(maxlen=buffer_len)

    def get(self):
        current_tick = cv.getTickCount()
        different_time = (current_tick - self._start_tick) * self._freq
        self._start_tick = current_tick

        self._difftimes.append(different_time)

        fps = 1000.0 / (sum(self._difftimes) / len(self._difftimes))
        fps_rounded = round(fps, 2)

        return fps_rounded

class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.d1 = nn.Linear(input_size, 60)
        self.d2 = nn.Linear(60, 40)
        self.d3 = nn.Linear(40, output_size)

    def forward(self, x):
        x = self.d1(x)
        x = F.relu(x)
        x = self.d2(x)
        x = F.relu(x)
        x = self.d3(x)
        return x


def get_bounding_rect(shape, landmarks):
    image_width, image_height = shape[1], shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def prep_landmark_list(shape, landmarks):
    image_width, image_height = shape[1], shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmarks(image, hand_landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    points = []
    for i in range(21):
        # cv.circle(image,(int((1-hand_landmarks.landmark[i].x) * image_width),int(hand_landmarks.landmark[i].y * image_height)),1,(0,0,0),3)
        points.append([int((1-hand_landmarks.landmark[i].x) * image_width),int(hand_landmarks.landmark[i].y * image_height)])
    points = np.array(points)

    minX = np.min(points[:,0])
    maxX = np.max(points[:,0])
    minY = np.min(points[:,1])
    maxY = np.max(points[:,1])

    # cv.rectangle(image,(minX,minY),(maxX,maxY),(0,0,0),2)
    points = np.float16(points)
    points[:,0] = (points[:,0]-minX)/(maxX-minX)
    points[:,1] = (points[:,1]-minY)/(maxY-minY)
    points = points.reshape(-1)
    points = list(points)

    return points


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        thumbs = [2, 3, 4]
        for k in range(len(thumbs)-1):
            cv.line(image, tuple(landmark_point[thumbs[k]]), tuple(landmark_point[thumbs[k+1]]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[thumbs[k]]), tuple(landmark_point[thumbs[k+1]]), (255, 255, 255), 2)

        # Index finger
        ifingers = [5, 6, 7, 8]
        for k in range(len(ifingers)-1):
            cv.line(image, tuple(landmark_point[ifingers[k]]), tuple(landmark_point[ifingers[k+1]]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[ifingers[k]]), tuple(landmark_point[ifingers[k+1]]), (255, 255, 255), 2)

        # Middle finger
        mfingers = [9, 10, 11, 12]
        for k in range(len(mfingers)-1):
            cv.line(image, tuple(landmark_point[mfingers[k]]), tuple(landmark_point[mfingers[k+1]]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[mfingers[k]]), tuple(landmark_point[mfingers[k+1]]), (255, 255, 255), 2)

        # Ring finger
        rfingers = [13, 14, 15, 16]
        for k in range(len(rfingers)-1):        
            cv.line(image, tuple(landmark_point[rfingers[k]]), tuple(landmark_point[rfingers[k+1]]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[rfingers[k]]), tuple(landmark_point[rfingers[k+1]]), (255, 255, 255), 2)

        # Little finger
        lfingers = [17, 18, 19, 20]
        for k in range(len(lfingers)-1):
            cv.line(image, tuple(landmark_point[lfingers[k]]), tuple(landmark_point[lfingers[k+1]]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[lfingers[k]]), tuple(landmark_point[lfingers[k+1]]), (255, 255, 255), 2)

        # Palm
        palms = [0, 1, 2, 5, 9, 13, 17, 0]
        for k in range(len(palms)-1):
            cv.line(image, tuple(landmark_point[palms[k]]), tuple(landmark_point[palms[k+1]]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[palms[k]]), tuple(landmark_point[palms[k+1]]), (255, 255, 255), 2)

    for index, landmark in enumerate(landmark_point):
        if index > 0 and index % 4 == 0:
            h = 8
        else:
            h = 5
        cv.circle(image, (landmark[0], landmark[1]), h, (255, 255, 255), -1)
        cv.circle(image, (landmark[0], landmark[1]), h, (0, 0, 0), 1)

    return image


def main(use_pytorch=True, 
        NUM_CLASSES=5, 
        use_static_image_mode=True, 
        min_detection_confidence=0.7, 
        min_tracking_confidence=0.5, 
        cap_device=0, 
        cap_width=960, 
        cap_height=540):

    # camera preparation 
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # mediapipe model load 
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # classifier model load
    if not use_pytorch:                                         # scikit-learn
        clf = pickle.load(open("scikit-classifier.pik", 'rb')) 
    else:                                                       # pytorch
        clf = Net(21*2, NUM_CLASSES)
        clf.load_state_dict(torch.load('pytorch-classifier.pt'))

    # labels load
    with open('thelabels.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]
    
    # fps measurement 
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    while True:
        fps = cvFpsCalc.get()

        # camera capture 
        ret, image = cap.read()
        if not ret:
            print("\nCAP NOT OPENED")
            break

        image = cv.flip(image, 1)  # mirror display
        debug_image = copy.deepcopy(image)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                
                pre_processed_landmark_list = pre_process_landmarks(image, hand_landmarks)
                
                # Hand sign classification
                if not use_pytorch:  # scikit-learn
                    hand_sign_id = np.argmax(clf.predict_proba([pre_processed_landmark_list]), axis=1)[0]
                else:                # pytorch
                    hand_sign_id = np.argmax(clf(torch.Tensor(pre_processed_landmark_list)).detach().numpy())
                
                # Drawing part: get positions
                brect         = get_bounding_rect(debug_image.shape, hand_landmarks)
                landmark_list = prep_landmark_list(debug_image.shape, hand_landmarks)
                
                # Drawing part: draw
                cv.rectangle(debug_image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1) 
                debug_image   = draw_landmarks(debug_image, landmark_list)

                cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)
                info_text = handedness.classification[0].label[0:]
                if keypoint_classifier_labels[hand_sign_id] != "":
                    info_text = info_text + ':' + keypoint_classifier_labels[hand_sign_id]
                cv.putText(debug_image, info_text, (brect[0] + 5, brect[1] - 4), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv.LINE_AA)

        # Display :)
        cv.imshow('Hand Gesture Recognition', debug_image)
        key = cv.waitKey(10)
        global confMatrix
        if key == 49 or key == 50 or key == 51 or key == 52 or key == 53:
            if hand_sign_id == key -49:
                confMatrix.append(1)
            else:
                confMatrix.append(0)
        if key == 27:  # ESC
            break

    cap.release()
    cv.destroyAllWindows()
    confMatrix = np.array(confMatrix)

    print("acc = ", np.mean(confMatrix))


if __name__ == '__main__':
    
    width=960 
    height=540 
        
    main(use_pytorch=True, 
        NUM_CLASSES=5, 
        use_static_image_mode=True, 
        min_detection_confidence=0.7, 
        min_tracking_confidence=0.5, 
        cap_device = 0, 
        cap_width  = width, 
        cap_height = height
    )

