import os
import cv2
import numpy as np
import sys
from scipy import stats
from tqdm import tqdm
from sklearn.cluster import KMeans
from ioVideo import mp4toRGB, playVideo
from collections import defaultdict

# simplified loadRGB
def loadRGB(filedir):
    frames = []
    # File handle
    # videoName = filedir.split("\\")[-1].split("_")[0]
    videoName = 'SAL'
    framesNPYName = "cache/frames_"+ videoName +".npy"
    if os.path.exists(framesNPYName):
        with open(framesNPYName, 'rb') as f:
            frames = np.load(f)
        return frames
# simplified getMotionVectors
def getMotionVectors(inImgs):
    motionVectors = []
    # File handle
    motionVectorsFileName = "cache/motionVectors_SAL.npy"
    if os.path.exists(motionVectorsFileName):
        with open(motionVectorsFileName, 'rb') as f:
            motionVectors = np.load(f)
        return motionVectors

# mode:
# 1= Mode with mv
# 2= K-Mean with mv
# 3= opticalflow
def  getForegroundMask(frames, motionVectors, mode, macroSize=16):

    _, height, width, _ = np.shape(frames)
    if mode == 1:
        fMasks = getForegroundMask_withMV(motionVectors, height, width, 1, macroSize)
    elif mode == 2:
        fMasks = getForegroundMask_withMV(motionVectors, height, width, 2, macroSize)
    elif mode == 3:
        fMasks = getForegroundMask_withOF(frames, height, width)

    return fMasks


def getForegroundMask_withOF(frames, height, width):
    fMasks = []
    fMasks.append(np.zeros((height, width, 1)))
    # Reading the first frame
    frame1 = frames[0]
    # Convert to gray scale
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    # Create mask
    hsv_mask = np.zeros_like(frame1)
    # Make image saturation to a maximum value
    hsv_mask[..., 1] = 255
    for fc in tqdm(range(1, len(frames))):
        frame2 = frames[fc]
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Optical flow is now calculated
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # Compute magnite and angle of 2D vector
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Set image hue value according to the angle of optical flow
        hsv_mask[..., 0] = ang * 180 / np.pi / 2
        # Set value as per the normalized magnitude of optical flow
        hsv_mask[..., 2] = np.minimum(mag*4,255)
        prvs = next
        hsv = hsv_mask.reshape(-1,3)
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(hsv)
        labels = kmeans.labels_
        labels_mode = stats.mode(labels, keepdims=False)[0]
        if labels_mode == 1:
            labels = labels+1

        fMasks.append(labels.reshape((height, width, 1)))
    return fMasks

# getForegroundMask_withMV, need to be merged with main
def getForegroundMask_withMV(motionVectors, height, width, mode=1, macroSize=16):
    # mode:
    # 1= Mode
    # 2= K-Mean

    fMasks = []
    fMasks.append(np.zeros((height, width, 1)))

    mvCount, mvHeight, mvWidth, _ = np.shape(motionVectors)
    
    if mode == 1:
        for motionVector in motionVectors:
            fMask = np.zeros((height, width, 1))
            colMode = stats.mode(motionVector)
            overallMode = stats.mode(colMode[0], axis=1)[0][0][0]
            for vCol in range(mvHeight):
                for vRow in range(mvWidth):
                    if motionVector[vCol][vRow][0] != overallMode[0] and motionVector[vCol][vRow][1] != overallMode[1]:
                        for i in range(macroSize):
                            for j in range(macroSize): 
                                fMask[vCol*macroSize+i][vRow*macroSize+j][0] = 1
            fMasks.append(fMask)

    if mode == 2:
        for motionVector in motionVectors:
            fMask = np.zeros((height, width, 1))
            mvs = motionVector.reshape(-1,2)
            kmeans = KMeans(n_clusters=2)
            kmeans.fit(mvs)
            labels = kmeans.labels_
            labels_mode = stats.mode(labels, keepdims=False)[0]
            if labels_mode == 1:
                labels = labels+1
            labels.reshape((height//macroSize, width//macroSize, 1))
            for vCol in range(mvHeight):
                for vRow in range(mvWidth):
                    if labels[vCol][vRow][0] == 1:
                        for i in range(macroSize):
                            for j in range(macroSize): 
                                fMask[vCol*macroSize+i][vRow*macroSize+j][0] = 1
            fMasks.append(fMask)

    return fMasks
    
# getForeAndBack, need to be merged with main
def getForeAndBack(frames, motionVectors):
    fMasks = getForegroundMask(frames, motionVectors,3)
    fgs = []
    bgs = []    
    framesCount, height, width, _ = np.shape(frames) 
    for n in tqdm(range(framesCount)):
        fg = np.zeros((height, width, 3)).astype('uint8')
        bg = np.zeros((height, width, 3)).astype('uint8')
        
        for h in range(height):
            for w in range(width):
                if fMasks[n][h][w][0] == 1:
                    fg[h][w][0:3] = frames[n][h][w][0:3]
                else:
                    bg[h][w][0:3] = frames[n][h][w][0:3]
        fgs.append(fg)
        bgs.append(bg)

    return fgs,bgs

def getForeground_Naive(inImgs,motionVectors,macroSize):
    nFrame, height, width, _ = np.shape(inImgs) 
    nRow, nCol = height//macroSize, width//macroSize
    foreImgs = []
    for fIdx in range(1, len(motionVectors)+1):
        curFrame = inImgs[fIdx][:]
        motionVectorsPerFrame = motionVectors[fIdx-1]
        motionDict = defaultdict(int)
        for r in range(nRow):
            for c in range(nCol):
                motionDict[tuple(motionVectorsPerFrame[r][c])] += 1

        bgMotion_x, bgMotion_y = sorted(motionDict.items(), key=lambda x:x[1])[-1][0]

        for r in range(nRow):
            for c in range(nCol):
                vec_x, vec_y = motionVectorsPerFrame[r][c]
                delta = 2
                if (bgMotion_x-delta) <= vec_x <= (bgMotion_x+delta) and (bgMotion_y-delta) <= vec_y <= (bgMotion_y+delta):
                    base_x, base_y = c * macroSize, r * macroSize # current macroblock start point
                    curFrame[base_y:(base_y + macroSize), base_x:(base_x + macroSize)] = 0 # black out the macroblock
        foreImgs.append(curFrame)

    return foreImgs

if __name__ == '__main__':
    motionVectors = getMotionVectors(None)
    frames = loadRGB(None)
    frames = frames[:120]
    # frames = mp4toRGB("./video/SAL.mp4")
    framesCount, height, width, _ = np.shape(frames)
    fgs, bgs = getForeAndBack(frames, motionVectors)
    playVideo(fgs,300)