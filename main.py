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
from parorama import stichParorama, genApp1
        
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
    interval_MV = 1
    nFrame, height, width, _ = np.shape(inImgs) 
    nProcess = 437

    inImgs_sub = inImgs[0:nProcess]
    motionVectors = getMotionVectors(inImgs_sub, macroSize, videoName,interval_MV=interval_MV)
    inImgs_sub = inImgs_sub[interval_MV:] # only keep frames with motion vector

    # 3. Get Foreground and Background [middle piont 1]
    # fgs, bgs = getForeAndBack(inImgs_sub, motionVectors)
    fgs = getForeground_Naive(inImgs_sub, motionVectors, macroSize) # super fast
    # playVideo(fgs,wait=3000)

    # 4. Stick Background to Parorama [middle piont 2]
    pararamaImg = stichParorama(inImgs, videoName)

    # 5. Application Outputs 1:  Panorama Video
    videoFrames_App1 = genApp1(pararamaImg, fgs, videoName)
    # playVideo(videoFrames_App1,wait=3000)
    
    # 6. Application Outputs 2:  Panorama Video with specified path

    # 7. Application Outputs 3:  Panorama Video by removing one of foreground object

    