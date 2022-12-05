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
from fgextract import getForeAndBack_mode6, getFgBg_withYOLO

def stichParorama(inImgs,videoName):
    # https://www.geeksforgeeks.org/opencv-panorama-stitching/
    stitchy=cv.Stitcher.create()
    # cv.imshow('last Img',cv.cvtColor(inImgs[-10], cv.COLOR_RGB2BGR))
    # cv.waitKey(0)
    print('Sticking Panorama...')
    (dummy,output1)=stitchy.stitch(inImgs) 
    # fileName = 'result/' + videoName + '_panorama1.jpg'
    # cv.imwrite(fileName, cv.cvtColor(output1, cv.COLOR_RGB2BGR))

    # (dummy,output2)=stitchy.stitch(inImgs[-150:][::10]) 
    # fileName = 'result/'+ videoName + '_panorama2.jpg'
    # cv.imwrite(fileName, cv.cvtColor(output2, cv.COLOR_RGB2BGR))
    # (dummy,output)=stitchy.stitch([output2,output1]) 

    if dummy == cv.STITCHER_OK:
        print('Your Panorama is ready!!!')
        fileName = 'result/'+ videoName + '_panorama.jpg'
        cv.imwrite(fileName, cv.cvtColor(output1, cv.COLOR_RGB2BGR))
        # cv.imshow('final result',cv.cvtColor(output, cv.COLOR_RGB2BGR))
        # cv.waitKey(0)
    else:
        print("stitching ain't successful")
    return output1

def stichParoramaSquence(inImgs,videoName):
    print('Sticking Panorama Squencially...')
    stitchy=cv.Stitcher.create()
    inImgs = inImgs[::15]
    curImg = inImgs[0]
    for idx in range(1, len(inImgs)):

        (dummy,curImg) = stitchy.stitch([curImg,inImgs[idx]]) 

        if dummy == cv.STITCHER_OK:
            print('Your Panorama is ready!!!')
            cv.imshow('final result',cv.cvtColor(curImg, cv.COLOR_RGB2BGR))
            cv.waitKey(0)
        else:
            print("stitching ain't successful")
            break
    fileName = 'result/'+ videoName + '_panorama.jpg'
    cv.imwrite(fileName, cv.cvtColor(curImg, cv.COLOR_RGB2BGR))
    return curImg

def genApp1(pararamaImg, fgs, videoName):
    print("making App1...")
    cv.imshow('pararamaImg',cv.cvtColor(pararamaImg, cv.COLOR_RGB2BGR))
    cv.waitKey(0)

    # fgs = fgs[::40]

    stitchy = cv.Stitcher.create()
    for fg in fgs:
        cv.imshow('fg',cv.cvtColor(fg, cv.COLOR_RGB2BGR))
        cv.waitKey(0)
        (dummy,output) = stitchy.stitch([pararamaImg,fg]) 
        if dummy == cv.STITCHER_OK:
            print('Your App1 is ready!!!')
            fileName = 'result/'+ videoName + '_App1.jpg'
            cv.imwrite(fileName, cv.cvtColor(output, cv.COLOR_RGB2BGR))
            cv.imshow('final result',cv.cvtColor(output, cv.COLOR_RGB2BGR))
            cv.waitKey(0)
            break
        else:
            print("stitching ain't successful")

def genApp2(pararamaImg, fgs, videoName):
    print("making App2...")
    pHieght, pWidth, _ = np.shape(pararamaImg)
    hieght, width, _ = np.shape(fgs[0])
    delta = pWidth//hieght
    novelFrames = []
    for idx,fg in enumerate(fgs):
        tmp = np.copy(pararamaImg)
        # https://stackoverflow.com/questions/34436137/how-to-replace-all-zeros-in-numpy-matrix-with-corresponding-values-from-another
        d = (fg!=[0,0,0])
        shift = idx*delta
        tmp[0:hieght, (0+shift):(width+shift),:][d] = fg[d]
        # for r in range(hieght):
        #     for c in range(width):
        #         if fg[r][c][0] != 0 or fg[r][c][1] != 0 or fg[r][c][2] != 0:
        #             tmp[r][c] = fg[r][c]
        # novelFrames.append(tmp[:, (0+shift):(width+shift),:])
        novelFrames.append(tmp[:, (-1-shift-width):(-1-shift),:])
    fileName = 'result/App2_'+ videoName + '.mp4'
    saveVideo(novelFrames, filePath = fileName)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filepath", type=str, default="./video/test3.mp4",help="specify video file name")
    parser.add_argument("-d", "--filedir", type=str, default="C:\\video_rgb\\SAL_490_270_437",help="specify rgb directory")
    args = parser.parse_args()
    # 1. Read Video
    frames, videoName = mp4toRGB(args.filepath)
    # playVideo(frames, 3000)

    # 2. Get Motion Vector
    macroSize = 16
    interval_MV = 1
    nFrame, height, width, _ = np.shape(frames) 
    # motionVectors = getMotionVectors(frames, macroSize, videoName,interval_MV=interval_MV)

    # 3. Get Foreground and Background [middle piont 1]
    # fg1s, fgs, bgs = getForeAndBack_mode6(frames, videoName)
    fg1s, fg2s, fgs, bgs, fg1Trims = getFgBg_withYOLO(frames, videoName)

    # playVideo(fg1Trims, 3000)
    for idx, fg1Trim in enumerate(fg1Trims):
        fileName = 'cache/fg1Trims/'+ videoName + '_fg1_'+ str(idx) +'.jpg'
        if fg1Trim == []: continue
        cv.imwrite(fileName, cv.cvtColor(fg1Trim, cv.COLOR_RGB2BGR))
    # stichParoramaSquence(bgs,videoName)
    # genApp1(bgs[100], bgs, videoName)
    
    # saveVideo(fg1s,filePath='cache/' + videoName + '_fg1s.mp4')
    # saveVideo(fgs,filePath='cache/' + videoName + '_fgs.mp4')
    # saveVideo(bgs,filePath='cache/' + videoName + '_bgs.mp4')
    
    # 4. Stick Background to Parorama [middle piont 2]
    # video 2: 0:200 200-260 260-383
    # pbgs = np.concatenate((bgs[0:200][::10],bgs[200:260][::2],bgs[260:][::15]))
    pbgs = bgs[0:200][::15]
    pararamaImg1 = stichParorama(pbgs, videoName)
    pbgs = bgs[200:][::15]
    pararamaImg2 = stichParorama(pbgs, videoName)

    # 5. Application Outputs 1: Panorama with foreground motion trail
    genApp1(pararamaImg, fg1Trims, videoName)

    # 6. Application Outputs 2: Panorama Video with specified path
    # genApp2(pararamaImg, fgs, videoName)
    