import os
import mediapipe as mp
import cv2
import time
import numpy as np


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# classes_list = ['a','l']
classes_list = [chr(i) for i in range(97, 97+26)]
print(len(classes_list), classes_list)
dataset_size = 50

cap = cv2.VideoCapture(0)
for j in  classes_list:
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.imshow('frame', frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                # Get bounding box coordinates
                x_min, y_min = int(min(landmark.x for landmark in landmarks.landmark) * frame.shape[1]), \
                            int(min(landmark.y for landmark in landmarks.landmark) * frame.shape[0])
                x_max, y_max = int(max(landmark.x for landmark in landmarks.landmark) * frame.shape[1]), \
                            int(max(landmark.y for landmark in landmarks.landmark) * frame.shape[0])
                y_min = int(y_min*0.8)
                y_max = int(y_max*1.2)
                x_min = int(x_min*0.8)
                x_max = int(x_max*1.2)
                hand_image = frame[y_min:y_max, x_min:x_max]
            if hand_image.shape[0] > 0 and hand_image.shape[1] > 0:
                # Display the extracted hand image
                # hand_image_gray = cv2.cvtColor(hand_image, cv2.COLOR_BGR2GRAY)
                hand_image_gray = hand_image
                hand_image_gray = cv2.resize(hand_image_gray, (200, 200))
                cv2.waitKey(100)
                # gray = cv2.cvtColor(hand_image_gray, cv2.COLOR_BGR2GRAY)
                # _, thresholded = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

                # lower = np.array([180, 180, 180])
                # upper = np.array([255, 255, 255])
                # thresholded = cv2.inRange(hand_image_gray, lower, upper)
                # # apply morphology
                # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
                # morph = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
                # mask = morph
                # mask = 255-mask
                # hand = cv2.bitwise_and(hand_image_gray, hand_image_gray, mask=mask)

                edge_detection_kernel = np.array([[-1,-1,-1],
                                                  [-1,8,-1],
                                                  [-1,-1,-1]])
                
                edges = cv2.filter2D(hand_image_gray, -1, edge_detection_kernel)


                sharpen_detection_kernel = np.array([[0,-1,0],
                                                  [-1,5,-1],
                                                  [0,-1,0]])
                
                shapened = cv2.filter2D(hand_image_gray, -1, sharpen_detection_kernel)

                cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), hand_image_gray)
                # cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}_edges.jpg'.format(counter)), edges)
                cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}_sharp.jpg'.format(counter)), shapened)
                # cv2.imwrite(os.path.join(DATA_DIR, str(j), f"{counter}_hand.jpg"), hand)
                # cv2.imwrite(os.path.join(DATA_DIR, str(j), f"{counter}_mask.jpg"), mask)

                counter += 1

cap.release()
cv2.destroyAllWindows()
