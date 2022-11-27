import cv2 as cv
import numpy as np
from PIL import Image
import sys, os.path, argparse
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from collections import defaultdict
from ioVideo import mp4toRGB, loadRGB, playVideo
from motionVector import getMotionVectors
from fgextract import getForegroundMask, getForeAndBack, getForeground_Naive

def stichParorama(inImgs,videoName):
    # https://www.geeksforgeeks.org/opencv-panorama-stitching/
    imgs = []
    for i in range(1,len(inImgs),30):
        imgs.append(cv.cvtColor(inImgs[i], cv.COLOR_RGB2BGR))
    # imgs.append(cv.cvtColor(inImgs[-1], cv.COLOR_RGB2BGR))
    stitchy=cv.Stitcher.create()
    (dummy,output)=stitchy.stitch(imgs) 
    if dummy == cv.STITCHER_OK:
        print('Your Panorama is ready!!!')
        fileName = 'result/panorama_'+ videoName + '.jpg'
        cv.imwrite(fileName, output)
        # cv.imshow('final result',output)
        # cv.waitKey(0)
    else:
        print("stitching ain't successful")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filepath", type=str, default="./video/SAL.mp4",help="specify video file name")
    parser.add_argument("-d", "--filedir", type=str, default="C:\\video_rgb\\SAL_490_270_437",help="specify rgb directory")
    args = parser.parse_args()

    # 1. Read Video
    inImgs, videoName = mp4toRGB(args.filepath)
    # inImgs, videoName = loadRGB(args.filedir)

    # 2. Get Motion Vector
    macroSize = 16
    interval_MV = 2
    nFrame, height, width, _ = np.shape(inImgs) 
    nProcess = 10
    inImgs = inImgs[0:nProcess]
    motionVectors = getMotionVectors(inImgs, macroSize, nProcess, videoName,interval_MV=interval_MV)
    inImgs = inImgs[interval_MV:] # only keep frames with motion vector

    # 3. Get Foreground and Background [middle piont 1]
    
    # fMasks = getForegroundMask(motionVectors, height, width,2, macroSize)
    # fgs, bgs = getForeAndBack(inImgs, fMasks)
    fgs = getForeground_Naive(inImgs,motionVectors, macroSize)
    playVideo(fgs,wait=3000)

    # 4. Stick Background to Parorama [middle piont 2]
    stichParorama(inImgs, videoName)

    # 5. Application Outputs 1:  Panorama Video

    # 6. Application Outputs 2:  Panorama Video with specified path

    # 7. Application Outputs 3:  Panorama Video by removing one of foreground object

    