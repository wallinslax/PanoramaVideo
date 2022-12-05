import os
import cv2
import numpy as np
import sys
import math
from scipy import stats
from tqdm import tqdm
from sklearn.cluster import KMeans
from ioVideo import mp4toRGB, playVideo, saveVideo
from collections import defaultdict
import torch
from sort import *

def getForegroundMask_withYOLOandSort(frames):
    framesCount, height, width, _ = np.shape(frames) 
    zeroMask = np.zeros((framesCount,height,width)).astype('uint8')
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    mot_tracker = Sort()
    fMaskDict = dict()
    trackerCounts = dict()
    for fIdx, frame in tqdm(enumerate(frames)):
        results = model(frame)
        # https://stackoverflow.com/questions/68008886/how-to-get-bounding-box-coordinates-from-yolov5-inference-with-a-custom-model
        boxes = results.pandas().xyxy[0]  # img1 predictions (pandas)
        # https://stackoverflow.com/questions/67244258/how-to-get-class-and-bounding-box-coordinates-from-yolov5-predictions
        labels, cord_thres = results.xyxy[0][:, -1].numpy(), results.xyxy[0][:, :-1].numpy()
        # results.show()
        
        dets = []
        for idx, label in enumerate(labels):
            if label == 0: # 0: person
                det = cord_thres[idx]
                dets.append(det)
        # https://github.com/abewley/sort
        trackers = mot_tracker.update(np.asarray(dets))
        # print(trackers)
        for tracker in trackers:
            xA, yA, xB, yB, trackNo = tracker
            xA, yA, xB, yB = int(xA), int(yA), int(xB), int(yB)
            if fMaskDict.get(trackNo) is None:
                fMaskDict[trackNo] = zeroMask.copy()
                trackerCounts[trackNo] = 0
            fMaskDict[trackNo][fIdx,yA:yB,xA:xB] = 255
            trackerCounts[trackNo] += 1

    # print(fMaskDict)
    trackerCountArr = list(trackerCounts.items())
    trackerCountArr.sort(key=lambda x:-x[1])
    if len(trackerCountArr)>1:
        trackerNo1, trackerNo2 = trackerCountArr[0][0], trackerCountArr[1][0]
    else:
        trackerNo1, trackerNo2 = trackerCountArr[0][0], trackerCountArr[0][0]
    return fMaskDict[trackerNo1], fMaskDict[trackerNo2]         

def getForeAndBack_mode6(frames,videoName):
    # File handle
    fMasks_FileName = "cache/fMasks_"+ videoName +"_YOLO.npy"
    if os.path.exists(fMasks_FileName):
        with open(fMasks_FileName, 'rb') as f:
            f1Masks,f2Masks = np.load(f)
    else:
        f1Masks,f2Masks = getForegroundMask_withYOLOandSort(frames)
        with open(fMasks_FileName, 'wb') as f:
            np.save(f,[f1Masks,f2Masks])
    
    black = np.zeros_like(frames[0])
    black = black+255
    fg1s, fgs, bgs = [], [], []
  
    framesCount, height, width, _ = np.shape(frames) 
    for n in tqdm(range(framesCount)):
        copy = frames[n].copy()
        fg1 = cv2.bitwise_and(copy, copy, mask = f1Masks[n])
        fgMask = cv2.bitwise_or(f1Masks[n],f2Masks[n])
        fg = cv2.bitwise_and(copy, copy, mask = fgMask)
        bg = cv2.bitwise_not(black, copy, mask = fgMask)
        fg1s.append(fg1)
        fgs.append(fg)
        bgs.append(bg)
    return fg1s,fgs,bgs

def getFgBg_withYOLO(frames, videoName):
    # File handle
    # fMasks_FileName = "cache/fMasks_"+ videoName +"_YOLO.npy"
    # if os.path.exists(fMasks_FileName):
    #     with open(fMasks_FileName, 'rb') as f:
    #         fg1s, fg2s, fgs, bgs, fg1Trims = np.load(f, allow_pickle=True)
    #         return fg1s, fg2s, fgs, bgs, fg1Trims

    _, height, width, _ = np.shape(frames)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    fMasks, f1Masks, f2Masks, bgs, fgs, fg1s, fg2s = [], [], [], [], [], [], []
    fg1Trims = []
    for frame in tqdm(frames):
        results = model(frame)
        # https://stackoverflow.com/questions/68008886/how-to-get-bounding-box-coordinates-from-yolov5-inference-with-a-custom-model
        boxes = results.pandas().xyxy[0]  # img1 predictions (pandas)
        # https://stackoverflow.com/questions/67244258/how-to-get-class-and-bounding-box-coordinates-from-yolov5-predictions
        labels, cord_thres = results.xyxy[0][:, -1].numpy(), results.xyxy[0][:, :-1].numpy()
        # results.show()

        # Generate foreground mask
        fg1 = np.zeros_like(frame)
        fg1Trim = []
        fg2 = np.zeros_like(frame)
        fg = np.zeros_like(frame)
        bg = np.copy(frame)
        f1Mask = np.zeros((height, width)).astype('uint8')
        f2Mask = np.zeros((height, width)).astype('uint8')
        fMask = np.zeros((height, width)).astype('uint8')
        
        for label, cord_thre in zip(labels,cord_thres):
            xA, yA, xB, yB, confidence = cord_thre
            xA, yA, xB, yB = int(xA), int(yA), int(xB), int(yB)
            if (xB-xA)<=30: continue

            if label == 0 and confidence >0.7: # 0: person
                fg1[yA:yB,xA:xB] = frame[yA:yB,xA:xB]
                rg = 200
                fg1Trim = np.copy(frame[(yA-rg):(yB+rg),(xA-rg):(xB+rg)])
                fg[yA:yB,xA:xB] = frame[yA:yB,xA:xB]
                f1Mask[yA:yB,xA:xB] = 255
                fMask[yA:yB,xA:xB] = 255
                bg[yA:yB,xA:xB] = 0
                
            if label == 36 and confidence >0.8: # 36: skateboard
                fg2[yA:yB,xA:xB] = frame[yA:yB,xA:xB]
                fg[yA:yB,xA:xB] = frame[yA:yB,xA:xB]
                f2Mask[yA:yB,xA:xB] = 255
                fMask[yA:yB,xA:xB] = 255
                bg[yA:yB,xA:xB] = 0

            # cv2.imshow('result',cv2.cvtColor(fg1, cv2.COLOR_RGB2BGR))
            # cv2.waitKey(0)
        
        fMasks.append(fMask)
        f1Masks.append(f1Mask)
        f2Masks.append(f2Mask)
        bgs.append(bg) 
        fg1s.append(fg1)
        fg1Trims.append(fg1Trim)
        fg2s.append(fg2)
        fgs.append(fg)

    # playVideo(fg1s, 30)
    # playVideo(fg2s, 30)
    # playVideo(fgs, 30)
    # with open(fMasks_FileName, 'wb') as f:
    #     np.save(f,[fg1s, fg2s, fgs, bgs, fg1Trims])
    return fg1s, fg2s, fgs, bgs, fg1Trims

if __name__ == '__main__':
    frames, videoName = mp4toRGB(filepath="./video/video2.mp4")
    fg1s, fgs, bgs = getForeAndBack_mode6(frames, videoName)
    playVideo(fg1s, 30)
    playVideo(fgs, 30)
    playVideo(bgs, 30) 
    # saveVideo(fg1s, "./cache/video2_fg1s.mp4")
    # saveVideo(fgs, "./cache/video2_fgs.mp4")
    # saveVideo(bgs, "./cache/video2_bgs.mp4")


    
    