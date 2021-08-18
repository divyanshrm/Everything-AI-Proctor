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
    if 0 in flag:
        print('Mobile Phone detected')
    if 1 in flag:
        print('No person detected')
    if 2 in flag:
        print('More than one person detected')
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break