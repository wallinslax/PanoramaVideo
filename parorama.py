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
    stitchy=cv.Stitcher.create()
    (dummy,output)=stitchy.stitch(inImgs[::10]) 
    if dummy == cv.STITCHER_OK:
        print('Your Panorama is ready!!!')
        fileName = 'result/panorama_'+ videoName + '.jpg'
        cv.imwrite(fileName, cv.cvtColor(output, cv.COLOR_RGB2BGR))
        # cv.imshow('final result',cv.cvtColor(output, cv.COLOR_RGB2BGR))
        # cv.waitKey(0)
    else:
        print("stitching ain't successful")
    return output

def genApp1(pararamaImg, fgs, videoName):
    videoFrames = []
    print("making App1...")
    for fg in fgs:
        tmp = np.copy(pararamaImg)
        hieght, width, _ = np.shape(fg)
        # https://stackoverflow.com/questions/34436137/how-to-replace-all-zeros-in-numpy-matrix-with-corresponding-values-from-another
        d = (fg!=[0,0,0])
        tmp[0:hieght,0:width,:][d] = fg[d]
        # for r in range(hieght):
        #     for c in range(width):
        #         if fg[r][c][0] != 0 or fg[r][c][1] != 0 or fg[r][c][2] != 0:
        #             tmp[r][c] = fg[r][c]
        videoFrames.append(tmp)
    # write MP4 to disk
    pHeight, pWidth, _ = np.shape(pararamaImg)
    out = cv.VideoWriter('result/app1_' + videoName + '.mp4', cv.VideoWriter_fourcc(*'MP4V'), 30.0, frameSize=(pWidth,pHeight))
    for frame in videoFrames:
        out.write(cv.cvtColor(frame, cv.COLOR_RGB2BGR))
    out.release()
    return videoFrames

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

    # 4. Stick Background to Parorama [middle piont 2]
    pararamaImg = stichParorama(inImgs, videoName)

    # 5. Application Outputs 1:  Panorama Video
    videoFrames_App1 = genApp1(pararamaImg, fgs, videoName)
    # playVideo(videoFrames_App1,wait=3000)
    