import cv2
import numpy as np
import mediapipe as mp
import csv
import time

min_detection_confidence = 0.5
min_tracking_confidence = 0.5


# def key(ch):
#     if(cv2.waitKey() == ord(ch)):
#         return True
#     else:
#         return False

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=min_detection_confidence,
    min_tracking_confidence=min_tracking_confidence)



# --------------------------------------------------- inja cap read o ina check kon



cap = cv2.VideoCapture(0)
while True:

    ret, image = cap.read()
    if not ret:
        break
    # image = cv2.imread("download.jfif")
    mediaImage = image.copy()

    mediaImage = cv2.flip(mediaImage, 1)
    mediaImage = cv2.cvtColor(mediaImage, cv2.COLOR_BGR2RGB)
    # Convert the BGR image to RGB before processing.
    results = hands.process(mediaImage)

    # Print handedness and draw hand landmarks on the image.
    # print('Handedness:', results.multi_handedness)
    # if not results.multi_hand_landmarks:
    #     # print("no result")
    #     continue
    image_height, image_width, _ = image.shape

    points = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # print((int((1-hand_landmarks.landmark[0].x) * image_width),int(hand_landmarks.landmark[0].y * image_height)))
            for i in range(21):
                cv2.circle(image,(int((1-hand_landmarks.landmark[i].x) * image_width),int(hand_landmarks.landmark[i].y * image_height)),1,(0,0,0),3)
                points.append([int((1-hand_landmarks.landmark[i].x) * image_width),int(hand_landmarks.landmark[i].y * image_height)])

        points = np.array(points)
        # print(points)

        minX = np.min(points[:,0])
        maxX = np.max(points[:,0])
        minY = np.min(points[:,1])
        maxY = np.max(points[:,1])

        cv2.rectangle(image,(minX,minY),(maxX,maxY),(0,0,0),2)
        points = np.float16(points)
        points[:,0] = (points[:,0]-minX)/(maxX-minX)
        points[:,1] = (points[:,1]-minY)/(maxY-minY)
        points = points.reshape(-1)
        points = list(points)
        # print(points)

        cv2.imshow("hand",image)

        handClass = cv2.waitKey(1)

        if handClass == 49 or handClass == 50 or handClass == 51 or handClass == 52 or handClass == 53:
            print(handClass-48)
            points.insert(0,handClass-49)
            with open('dataMAMAMIA.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(list(points))
    else:
        cv2.imshow("hand",image)
        cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
