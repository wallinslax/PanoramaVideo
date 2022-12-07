import cv2 as cv
import numpy as np
from PIL import Image
import sys, os.path, argparse
from os import listdir
from os.path import isfile, join
import torch
from tqdm import tqdm
from collections import defaultdict
from ioVideo import mp4toRGB, loadRGB

def getMotionVectors(inImgs, macroSize, videoName, interval_MV=1):
    nFrame, height, width, _ = np.shape(inImgs) 
    motionVectors = []
    nProcess = len(inImgs)
    # File handle
    motionVectorsFileName = "cache/motionVectors_"+ videoName+"_"+ str(nProcess) +"_"+ str(interval_MV) +".npy"
    # motionVectorsFileName = "cache/motionVectors_SAL_437small.npy" ########
    if os.path.exists(motionVectorsFileName):
        with open(motionVectorsFileName, 'rb') as f:
            motionVectors = np.load(f)
        return motionVectors

    for fIdx in tqdm(range(interval_MV,nProcess)):
        curFrame, prvFrame = inImgs[fIdx], inImgs[fIdx - interval_MV]
        motionVectorsPerFrame = getMotionVectorsPerFrame(curFrame, prvFrame, macroSize)
        motionVectors.append(motionVectorsPerFrame)

    # File handle
    with open(motionVectorsFileName, 'wb') as f:
        np.save(f,motionVectors)

    return motionVectors

def getMotionVectorsPerFrame(curFrame, prvFrame, macroSize):
    height, width, _ = np.shape(curFrame) 
    nRow, nCol = height//macroSize, width//macroSize
    searchRange = macroSize

    motionVectorsPerFrame = np.empty((nRow,nCol,2))
    motionVectorMADs = np.ones((nRow,nCol))* float('inf')
    for r in tqdm(range(nRow),leave=False):
        for c in range(nCol):
            motionVectorMADs[r][c] = MAD(curFrame,prvFrame, 0, 0, r, c, macroSize)
            motionVectorsPerFrame[r][c] = [0,0]
            
            for vec_x in range(-searchRange, searchRange):
                for vec_y in range(-searchRange, searchRange):
                    error = MAD(curFrame,prvFrame, vec_x, vec_y, r, c, macroSize)
                    if error < motionVectorMADs[r][c]:
                        motionVectorMADs[r][c] = error
                        motionVectorsPerFrame[r][c] = [vec_x, vec_y]

    # print(motionVectorsPerFrame)
    # motionDict = defaultdict(int)
    # for r in range(nRow):
    #     for c in range(nCol):
    #         motionDict[tuple(motionVectorsPerFrame[r][c])] += 1
    # bgMotion_x, bgMotion_y = sorted(motionDict.items(), key=lambda x:x[1])[-1][0]
    # print('all macroblock=',sum(motionDict.values()))
    # print (bgMotion_x, bgMotion_y)
    return motionVectorsPerFrame

def MAD(curFrame, prvFrame, vec_x, vec_y, r, c, macroSize):
    height, width, _ = np.shape(curFrame) 
    base_x, base_y = c * macroSize, r * macroSize # current macroblock start point
    # early retrun when illegal previous macroblock
    if base_x + vec_x < 0 or base_x + vec_x + macroSize >= width\
    or base_y + vec_y < 0 or base_y + vec_y + macroSize >= height:
        return float('inf')
    curMB_BGR = curFrame[base_y:(base_y + macroSize), base_x:(base_x + macroSize)]
    prvMB_BGR = prvFrame[(base_y + vec_y):(base_y + vec_y + macroSize), (base_x + vec_x):(base_x + vec_x + macroSize)]

    # (bgMotion_x, bgMotion_y)= (8.0, 15.0): 310; (15.0, 8.0): 414; (11.0, 15.0): 273; (13.0, 15.0): 284
    curMB_Y = 0.299 * curMB_BGR[:,:,2] + 0.587 * curMB_BGR[:,:,1] + 0.114 * curMB_BGR[:,:,0]
    prvMB_Y = 0.299 * prvMB_BGR[:,:,2] + 0.587 * prvMB_BGR[:,:,1] + 0.114 * prvMB_BGR[:,:,0]
    subError1 = abs(np.subtract(curMB_Y,prvMB_Y)).sum()

    # (bgMotion_x, bgMotion_y)= (11.0, 15.0) 244, (8.0, 15.0) 244, (13,15) 232
    # curMB_Y = 0.299 * curMB_BGR[:,:,0] + 0.587 * curMB_BGR[:,:,1] + 0.114 * curMB_BGR[:,:,2]
    # prvMB_Y = 0.299 * prvMB_BGR[:,:,0] + 0.587 * prvMB_BGR[:,:,1] + 0.114 * prvMB_BGR[:,:,2]
    # subError2 = abs(np.subtract(curMB_Y,prvMB_Y)).sum()

    # (bgMotion_x, bgMotion_y)= (13.0 8.0) centralize MV
    # curMB_Grey = cv.cvtColor(curMB_BGR,cv.COLOR_BGR2GRAY)
    # prvMB_Grey = cv.cvtColor(prvMB_BGR,cv.COLOR_BGR2GRAY)
    # curMB_Grey = np.array(curMB_Grey).astype(np.int16)
    # prvMB_Grey = np.array(prvMB_Grey).astype(np.int16)
    # subError3 = abs(np.subtract(curMB_Grey,prvMB_Grey)).sum()

    # NO centralize MV
    # curMB_HSV = cv.cvtColor(curMB_BGR,cv.COLOR_BGR2HSV)
    # prvMB_HSV = cv.cvtColor(prvMB_BGR,cv.COLOR_BGR2HSV)
    # subError4 = abs(np.subtract(curMB_HSV[:,:,2],prvMB_HSV[:,:,2])).sum()

    # subError1 = round(subError1, 3)
    # subError3 = round(subError3, 3)
    # if  subError1 != subError3:
    #     print("mismatch")

    return subError1

    subErrorEvl = 0
    for x in range(macroSize):
        for y in range(macroSize):
            
            if base_x + x >= width or base_y + y >= height:
                print(base_x + x)
            cur_c = curFrame[base_y + y][base_x + x]
            prv_c = prvFrame[base_y + y + vec_y][base_x + x + vec_x]
            cur_y = 0.299 * cur_c[0] + 0.587 * cur_c[1] + 0.114 * cur_c[2]
            prv_y = 0.299 * prv_c[0] + 0.587 * prv_c[1] + 0.114 * prv_c[2]
            subErrorEvl += abs(prv_y - cur_y)
    subErrorEvl = round(subErrorEvl, 3)
    subErrorEvl = round(subError2, 3)
    if  subErrorEvl != subError2:
        print("mismatch")

if __name__ == '__main__':
    '''
    # MV evaluation ----------------------------------------------------
    rfImg = cv.imread("mvTest/reference.jpg", cv.IMREAD_COLOR)
    rfImg_11dx_15dy = cv.imread("mvTest/reference_11dx_15dy.jpg", cv.IMREAD_COLOR)
    rfImg_minus6dx_minus4dy = cv.imread("mvTest/reference_minus6dx_minus4dy.jpg", cv.IMREAD_COLOR)
    height, width, _ = np.shape(rfImg)
    macroSize = 16
    nRow, nCol = height//macroSize, width//macroSize
    
    # for r in range(nRow):
    #     for c in range(nCol):
    #         base_x, base_y = c * macroSize, r * macroSize
    #         rfImg = cv.rectangle(rfImg, (base_x,base_y), (base_x + macroSize, base_y + macroSize), (255, 0, 0), 1)
    # cv.imshow("rfImg",  rfImg)
    # cv.waitKey(0)
    
    # print(rfImg[-1][0]) # BGR
    # curFrame_Grey = cv.cvtColor(rfImg,cv.COLOR_BGR2GRAY)
    # curFrame_YUV = cv.cvtColor(rfImg,cv.COLOR_BGR2YUV)
    # cv.imshow("curFrame_Grey",  curFrame_Grey)
    # cv.imshow("curFrame_YUV",  curFrame_YUV[:,:,0])
    # cv.waitKey(0)

    motionVectorsPerFrame = getMotionVectorsPerFrame(curFrame = rfImg, prvFrame = rfImg, macroSize = macroSize)

    # MV evaluation ----------------------------------------------------
    '''

    # 1. Read Video
    inImgs, videoName = mp4toRGB("video/SAL.mp4")
    # inImgs, videoName = loadRGB(args.filedir)

    # 2. Get Motion Vector
    macroSize = 16
    interval_MV = 1
    nFrame, height, width, _ = np.shape(inImgs) 


    # motionVectorsPerFrame = getMotionVectorsPerFrame(curFrame = inImgs[11], prvFrame = inImgs[10], macroSize = macroSize)
    motionVectors = getMotionVectors(inImgs, 16, videoName,interval_MV=1)
    # Fill background-------------
    fIdx = 21
    k = 20
    
    # calculate background motion vector
    nRow, nCol = height//macroSize, width//macroSize
    motionVectorsPerFrame = motionVectors[fIdx-1]
    motionDict = defaultdict(int)
    for r in range(nRow):
        for c in range(nCol):
            motionDict[tuple(motionVectorsPerFrame[r][c])] += 1
    bgMotion_x, bgMotion_y = sorted(motionDict.items(), key=lambda x:x[1])[-1][0]
    bgMotion_x, bgMotion_y = int(bgMotion_x), int(bgMotion_y)
    print (bgMotion_x, bgMotion_y)

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    results = model(inImgs[fIdx])
    labels, cord_thres = results.xyxy[0][:, -1].numpy(), results.xyxy[0][:, :-1].numpy()
    bg = np.copy(inImgs[fIdx])
    fillbg = np.zeros_like(inImgs[fIdx])
    for label, cord_thre in zip(labels,cord_thres):
        xA, yA, xB, yB, confidence = cord_thre
        xA, yA, xB, yB = int(xA), int(yA), int(xB), int(yB)
        if label == 0 and confidence >0.7: # 0: person
            bg[yA:yB,xA:xB] = 0
            vec_x, vec_y = bgMotion_x*k, bgMotion_y*k
            fillbg[yA:yB,xA:xB] = inImgs[fIdx- k][(yA+vec_y):(yB+vec_y),(xA+vec_x):(xB+vec_x)]
    cv.imshow('bg',bg)
    cv.imshow('fillbg',fillbg)
    cv.waitKey(0)
    

    