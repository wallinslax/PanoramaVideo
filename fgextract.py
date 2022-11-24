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
    motionVectorsFileName = "cache/motionVectors_SAL_437small.npy"
    if os.path.exists(motionVectorsFileName):
        with open(motionVectorsFileName, 'rb') as f:
            motionVectors = np.load(f)
        return motionVectors

# getForegroundMask, need to be merged with main
def getForegroundMask(motionVectors, height, width, mode=1, macroSize=16):
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
            labels = kmeans.labels_.reshape((height//macroSize, width//macroSize, 1))
            for vCol in range(mvHeight):
                for vRow in range(mvWidth):
                    if labels[vCol][vRow][0] == 1:
                        for i in range(macroSize):
                            for j in range(macroSize): 
                                fMask[vCol*macroSize+i][vRow*macroSize+j][0] = 1
            fMasks.append(fMask)

    return fMasks
    
# getForeAndBack, need to be merged with main
def getForeAndBack(frames, fMasks):
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
    # frames = mp4toRGB("./video/SAL.mp4")
    framesCount, height, width, _ = np.shape(frames) 
    fMasks = getForegroundMask(motionVectors, height, width,2, 16)
    fgs, bgs = getForeAndBack(frames, fMasks)
    playVideo(fgs)