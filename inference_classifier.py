import torch
import cv2
import mediapipe as mp
import SignLangToAlphabets
import numpy as np
import time

input_size = 42
hidden_size = 35
num_classes = 3

model = SignLangToAlphabets.ConvNet()
model.load_state_dict(torch.load('cnn.pth'))
model.eval()


cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# labels_dict = {0: 'B', 1: 'A', 2: 'L'}
# labels_dict = {0:"a",1:"b",2:"c",3:"d",4:"e",5:"f",6:"g",7:"h",8:"i",9:"j",10:"k",11:"l",12:"m",13:"n",14:"o",15:"p",16:"q",17:"r",18:"s",19:"t",20:"u",21:"unkowen",22:"v",23:"w",24:"x",25:"y",26:"z"}
labels_dict = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g", 7: "h", 8: "i", 9: "j", 10: "k", 11: "l", 12: "m", 13: "n",
               14: "o", 15: "p", 16: "q", 17: "r", 18: "s", 19: "t", 20: "u", 21: "unkowen", 22: "v", 23: "w", 24: "x", 25: "y", 26: "z"}


# while True:
i = 0
start_time = time.time()
while cap.isOpened():
    elapsed_time = time.time() - start_time
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        # for hand_landmarks in results.multi_hand_landmarks:
        #     mp_drawing.draw_landmarks(
        #         frame,  # image to draw
        #         hand_landmarks,  # model output
        #         mp_hands.HAND_CONNECTIONS,  # hand connections
        #         mp_drawing_styles.get_default_hand_landmarks_style(),
        #         mp_drawing_styles.get_default_hand_connections_style())

        # for hand_landmarks in results.multi_hand_landmarks:
        #     for i in range(len(hand_landmarks.landmark)):
        #         x = hand_landmarks.landmark[i].x
        #         y = hand_landmarks.landmark[i].y

        #         x_.append(x)
        #         y_.append(y)

        #     for i in range(len(hand_landmarks.landmark)):
        #         x = hand_landmarks.landmark[i].x
        #         y = hand_landmarks.landmark[i].y
        #         data_aux.append(x - min(x_))
        #         data_aux.append(y - min(y_))

        # x1 = int(min(x_) * W) - 10
        # y1 = int(min(y_) * H) - 10

        # x2 = int(max(x_) * W) - 10
        # y2 = int(max(y_) * H) - 10
        for landmarks in results.multi_hand_landmarks:
            # Get bounding box coordinates
            x_min, y_min = int(min(landmark.x for landmark in landmarks.landmark) * frame.shape[1]), \
                int(min(landmark.y for landmark in landmarks.landmark)
                    * frame.shape[0])
            x_max, y_max = int(max(landmark.x for landmark in landmarks.landmark) * frame.shape[1]), \
                int(max(landmark.y for landmark in landmarks.landmark)
                    * frame.shape[0])
            y_min = int(y_min*0.8)
            y_max = int(y_max*1.2)
            x_min = int(x_min*0.8)
            x_max = int(x_max*1.2)
            hand_image = frame[y_min:y_max, x_min:x_max]
        if elapsed_time >= 1 and hand_image.shape[0] > 0 and hand_image.shape[1] > 0:
            start_time = time.time()
            # Display the extracted hand image
            # hand_image_gray = hand_image
            hand_image_gray = cv2.cvtColor(hand_image, cv2.COLOR_BGR2GRAY)
            hand_image_gray = cv2.resize(hand_image_gray, (200, 200))
            hand_image_gray = hand_image_gray/256
            hand_image_gray = hand_image_gray.astype(np.float32)
            hand_image_gray = np.stack([hand_image_gray] * 3)
            hand_image_gray = hand_image_gray[np.newaxis, :, :, :]
            # hand_image_gray = np.array([hand_image_gray] * 3)
            # hand_image_gray = np.transpose(hand_image_gray, (1, 2, 0))
            # print(type(hand_image_gray))
            # print(hand_image_gray.shape)
            # use model and predict

            outputs = model(torch.tensor(hand_image_gray))
            _, predicted = torch.max(outputs, 1)

            print(f"{i}: {labels_dict[predicted.item()]}")
            i += 1

            # print(type(hand_image_gray))
            cv2.putText(hand_image_gray, "S", (5, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            # cv2.imshow('Hand Image', hand_image_gray)

            # outputs = model(torch.tensor(data_aux))labels_dict[predicted.item()]

            # prediction = torch.argmax(outputs).item()

            # predicted_character = labels_dict[prediction]

    cv2.imshow('frame', frame)
    # cv2.waitKey(1)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
