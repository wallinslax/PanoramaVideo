import random
import cv2 as cv
import numpy as np
from PIL import Image
import sys, os.path, argparse, datetime
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from collections import defaultdict
from ioVideo import mp4toRGB, loadRGB, playVideo,saveVideo
from motionVector import getMotionVectors
from fgextract import getForeAndBack_mode6, getFgBg_withYOLO
# from panorama.motion_trail_stitcher import compute_homography,stitch

def stichParorama(inImgs,filePath):
    # https://www.geeksforgeeks.org/opencv-panorama-stitching/
    stitchy=cv.Stitcher.create()
    print('Sticking Panorama...')
    (success, output1) = stitchy.stitch(inImgs) 
    if success == cv.STITCHER_OK:
        print('Your Panorama is ready!!!')
        cv.imwrite(filePath, cv.cvtColor(output1, cv.COLOR_RGB2BGR))
    else:
        print("stitching ain't successful")
    return output1

def genApp2(pnrmImgs, orgHieght, orgWidth, videoName):
    print("making App2...")
    
    pHieght, pWidth, _ = np.shape(pnrmImgs[0])
    delta = pWidth//len(pnrmImgs)
    novelFrames = []
    focusHieght, focusWidth = orgHieght//2, orgWidth//2

    for idx,pnrmImg in enumerate(pnrmImgs): 
        xShift = idx*delta
        if (orgWidth+xShift)>=pWidth:
            xShift = pWidth - orgWidth
        yShift = random.randint(30, 50)

        novelFrames.append(pnrmImg[(0 + yShift):(focusHieght + yShift), (0 + xShift):(focusWidth + xShift)])
    fileName = 'result/'+ videoName + '_App2.mp4'
    saveVideo(novelFrames, filePath = fileName)
    return novelFrames

    # https://stackoverflow.com/questions/34436137/how-to-replace-all-zeros-in-numpy-matrix-with-corresponding-values-from-another
    # d = (fg!=[0,0,0])
    # tmp[0:hieght, (0+shift):(width+shift),:][d] = fg[d]

    # for r in range(hieght):
    #     for c in range(width):
    #         if fg[r][c][0] != 0 or fg[r][c][1] != 0 or fg[r][c][2] != 0:
    #             tmp[r][c] = fg[r][c]
    # novelFrames.append(tmp[:, (0+shift):(width+shift),:])
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filepath", type=str, default="./video/test1.mp4",help="specify video file name")
    parser.add_argument("-d", "--filedir", type=str, default="C:\\video_rgb\\SAL_490_270_437",help="specify rgb directory")
    args = parser.parse_args()
    ## 1. Read Video
    frames, videoName = mp4toRGB(args.filepath)
    # playVideo(frames, 3000)

    ## 2. Get Motion Vector
    macroSize = 16
    interval_MV = 1
    nFrame, height, width, _ = np.shape(frames) 
    # motionVectors = getMotionVectors(frames, macroSize, videoName,interval_MV=interval_MV)

    ## 3. Get Foreground and Background [middle piont 1]
    nSplit = 10
    frames = frames[::nSplit]
    fg1s, fgs, bgs = getForeAndBack_mode6(frames, videoName)
    # fg1s, fg2s, fgs, bgs, fg1Trims = getFgBg_withYOLO(frames, videoName)
    # playVideo(frames, 3000)
    
    ## 4. Stick Background to Parorama [middle piont 2]
    # video 2: 0:200 200-260 260-383
    # pbgs = np.concatenate((bgs[0:200][::10],bgs[200:260][::2],bgs[260:][::15]))
    pararamaImg = stichParorama(frames, filePath = 'cache/' + videoName + '_panorama_'+ str(nSplit) +'.jpg')
    # cv.imshow('PararamaImg',cv.cvtColor(pararamaImg, cv.COLOR_RGB2BGR))
    # cv.waitKey(0)

    ## 5. Application Outputs 1: Panorama with foreground motion trail


    ## 6. Application Outputs 2: Panorama Video with specified path
    pFrames, videoName = mp4toRGB("video/motion_video_SAL.mp4")
    # playVideo(pFrames, 30)
    nFrame, height, width, _ = np.shape(frames) 
    genApp2(pFrames, height, width, videoName)
    