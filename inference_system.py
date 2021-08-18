from person_detection import person_and_phone_detection
from eye_gaze_test import eye_tracker
import cv2

PersonAndPhone = person_and_phone_detection()
EyeGaze = eye_tracker()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret == False:
        break
    flag = PersonAndPhone.infer(frame)
    if flag == 0:
        print('Mobile Phone detected')
    elif flag == 1:
        print('No person detected')
    elif flag == 2:
        print('More than one person detected')
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break