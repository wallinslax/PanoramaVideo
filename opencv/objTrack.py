import cv2
import numpy as np

cap = cv2.VideoCapture("video/SAL.mp4")
out = cv2.createBackgroundSubtractorMOG2()

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
output = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480), isColor=False)

while(cap.isOpened()):
    ret, frame = cap.read()

    if ret==True:
        frame = cv2.flip(frame,180)

        outmask = out.apply(frame)
        output.write(outmask)

        cv2.imshow('original', frame)
        cv2.imshow('Motion Tracker', outmask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break


output.release()
cap.release()
cv2.destroyAllWindows()