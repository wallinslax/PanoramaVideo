from __future__ import print_function
import cv2 as cv
import argparse
# https://www.youtube.com/watch?v=sSJ--fZLkZA&ab_channel=FalconInfomatic
# https://github.com/opencv/opencv/tree/4.x/samples/python/tutorial_code/video/background_subtraction
# https://stackoverflow.com/questions/66876520/how-to-extract-foreground-form-a-moving-camera-by-using-opencv
# https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/iet-cvi.2017.0187
parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='video/SAL.mp4')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()

## [create]
#create Background Subtractor objects
if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()
## [create]
backSub = cv.createBackgroundSubtractorKNN()

## [capture]
capture = cv.VideoCapture("video/SAL.mp4")
if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)
## [capture]

while True:
    ret, frame = capture.read()
    if frame is None:
        break
    ## [apply]
    #update the background model
    fgMask = backSub.apply(frame)
    # bgMask = 255 - fgMask
    ## [apply]

    ## [display_frame_number]
    #get the frame number and write it on the current frame
    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    ## [display_frame_number]

    ## [show]
    #show the current frame and the fg masks
    nRow, nCol = fgMask.shape
    # for r in range(nRow):
    #     for c in range(nCol):
    #         if fgMask[r][c] == 0:
    #             frame[r][c] = [0,0,0]
    fg = cv.bitwise_or(frame, frame, mask=fgMask)

    # cv.imshow('Frame', frame)
    # cv.imshow('FG Mask', fgMask)
    cv.imshow('fg', fg)
    ## [show]   

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break