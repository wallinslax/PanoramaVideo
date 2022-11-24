import cv2 as cv
import numpy as np
from PIL import Image
import sys, os.path, argparse
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from collections import defaultdict
from ioVideo import mp4toRGB, loadRGB

gMacroSize = 8
gHeight, gWidth = 0, 0
def getMotionVectors(inImgs, macroSize, nProcess, videoName):
    global gMacroSize, gHeight, gWidth
    gMacroSize = macroSize
    nFrame, gHeight, gWidth, _ = np.shape(inImgs) 
    motionVectors = []

    # File handle
    motionVectorsFileName = "cache/motionVectors_"+ videoName+"_"+ str(nProcess) +".npy"
    # motionVectorsFileName = "cache/motionVectors_SAL_437small.npy" ########
    if os.path.exists(motionVectorsFileName):
        with open(motionVectorsFileName, 'rb') as f:
            motionVectors = np.load(f)
        return motionVectors

    for fIdx in tqdm(range(1,nProcess)):
        curFrame, prvFrame = inImgs[fIdx], inImgs[fIdx - 1]
        motionVectorsPerFrame = getMotionVectorsPerFrame(curFrame, prvFrame)
        motionVectors.append(motionVectorsPerFrame)

    # File handle
    with open(motionVectorsFileName, 'wb') as f:
        np.save(f,motionVectors)

    return motionVectors

def getMotionVectorsPerFrame(curFrame, prvFrame):
    nRow, nCol = gHeight//gMacroSize, gWidth//gMacroSize
    k = gMacroSize
    motionVectorsPerFrame = np.empty((nRow,nCol,2))
    motionVectorMADs = np.ones((nRow,nCol))* float('inf')
    for r in tqdm(range(nRow),leave=False):
        for c in range(nCol):
            motionVectorsPerFrame[r][c] = [0,0]
            error = MAD(curFrame,prvFrame, 0, 0, r, c)
            for vec_x in range(-k, k):
                for vec_y in range(-k, k):
                    error = MAD(curFrame,prvFrame, vec_x, vec_y, r, c)
                    if error < motionVectorMADs[r][c]:
                        motionVectorMADs[r][c] = error
                        motionVectorsPerFrame[r][c] = [vec_x, vec_y]

    # print(motionVectorsPerFrame)
    return motionVectorsPerFrame

def MAD(curFrame, prvFrame, vec_x, vec_y, r, c):
    base_x, base_y = c * gMacroSize, r * gMacroSize # current macroblock start point
    # early retrun when illegal previous macroblock
    if base_x + vec_x < 0 or base_x + vec_x + gMacroSize >= gWidth\
    or base_y + vec_y < 0 or base_y + vec_y + gMacroSize >= gHeight:
        return float('inf')
    curMB_RGB = curFrame[base_y:(base_y + gMacroSize), base_x:(base_x + gMacroSize)]
    prvMB_RGB = prvFrame[(base_y + vec_y):(base_y + vec_y + gMacroSize), (base_x + vec_x):(base_x + vec_x + gMacroSize)]
    curMB_Y = 0.299 * curMB_RGB[:,:,0] + 0.587 * curMB_RGB[:,:,1] + 0.114 * curMB_RGB[:,:,2]
    prvMB_Y = 0.299 * prvMB_RGB[:,:,0] + 0.587 * prvMB_RGB[:,:,1] + 0.114 * prvMB_RGB[:,:,2]
    subError = abs(np.subtract(curMB_Y,prvMB_Y)).sum()
    return subError

    subError = 0
    for x in range(gMacroSize):
        for y in range(gMacroSize):
            
            if base_x + x >= width or base_y + y >= height:
                print(base_x + x)
            cur_c = curFrame[base_y + y][base_x + x]
            prv_c = prvFrame[base_y + y + vec_y][base_x + x + vec_x]
            cur_y = 0.299 * cur_c[0] + 0.587 * cur_c[1] + 0.114 * cur_c[2]
            prv_y = 0.299 * prv_c[0] + 0.587 * prv_c[1] + 0.114 * prv_c[2]
            subError += abs(prv_y - cur_y)
    return subError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filepath", type=str, default="./video/SAL.mp4",help="specify video file name")
    parser.add_argument("-d", "--filedir", type=str, default="C:\\video_rgb\\SAL_490_270_437",help="specify rgb directory")
    args = parser.parse_args()
    # Global variable
    macroSize = 16
    # process
    inImgs = mp4toRGB(args.filepath)
    inImgs = loadRGB(args.filedir)
    motionVectors = getMotionVectors(inImgs)