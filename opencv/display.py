import cv2
import numpy as np
import os.path
import sys

videoname = sys.argv[1]
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, '../video/'+videoname+'.mp4')
frames = []

def mp4toRGB():
    cap = cv2.VideoCapture(filename)
    while (cap.isOpened()):
        # Get a video frame
        hasFrame, frame = cap.read()

        if hasFrame == True:
            ## Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height = frame.shape[0]
            width = frame.shape[1]
            frame = np.reshape(frame.ravel(),(height, width,3))
            frames.append(frame)

        else:
            break

    cap.release()

def playVideo():
    for frame in frames:
        # to display with cv2 we need to convert to BGR first
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('rgb_frames',frame)
        cv2.waitKey(25)

if __name__ == '__main__':
    mp4toRGB()

    # frames is a 4d array [frame][height][width][r,g,b]
    # print(frames)
    # print(len(frames)) #frame count
    # print(len(frames[0])) #height
    # print(len(frames[0][0])) #width
    

    playVideo()