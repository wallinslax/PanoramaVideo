import cv2 as cv
import numpy as np
from PIL import Image
import sys, os.path, argparse
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from collections import defaultdict
from ioVideo import mp4toRGB, loadRGB, playVideo,saveVideo
from motionVector import getMotionVectors
from fgextract import getFgBg_withYOLO, genApp3
from parorama import stichParorama

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filepath", type=str, default="./video/SAL.mp4",help="specify video file name")
    args = parser.parse_args()

    # 1. Read Video
    inImgs, videoName = mp4toRGB(args.filepath)

    # 2. Get Motion Vector
    macroSize = 16
    interval_MV = 1
    nFrame, height, width, _ = np.shape(inImgs) 
    motionVectors = getMotionVectors(inImgs, macroSize, videoName,interval_MV=interval_MV)

    # 3. Get Foreground and Background [middle piont 1]
    fg1s, fg2s, fgs, bgs, fg1Trims = getFgBg_withYOLO(inImgs, videoName)
    playVideo(fgs,wait=30)
    saveVideo(bgs,filePath='result/' + videoName + '_bgs.mp4')
    saveVideo(fgs,filePath='result/' + videoName + '_fgs.mp4')

    # 4. Stick Background to Parorama [middle piont 2]
    nSplit = 15
    pararamaImg = stichParorama(inImgs[::nSplit], filePath = 'result/' + videoName + '_panorama_'+ str(nSplit) +'.jpg')
    # cv.imshow('PararamaImg',cv.cvtColor(pararamaImg, cv.COLOR_RGB2BGR))
    # cv.waitKey(0)

    # 5. Application Outputs 1:  Motion Trail

    # 6. Application Outputs 2:  Panorama Video with specified path

    # 7. Application Outputs 3:  Original Video without Foreground
    print("Create Application Outputs 3. It will take a hour or more.")
    bgFilleds = genApp3(inImgs, motionVectors, videoName)
    saveVideo(bgFilleds,filePath='result/' + videoName + '_bgFilleds.mp4')

    