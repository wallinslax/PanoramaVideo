import os
import cv2
import numpy as np
import sys
from scipy import stats
from tqdm import tqdm
from sklearn.cluster import KMeans

macroSize = 16


# simplified loadRGB
def loadRGB(filedir):
    frames = []
    # File handle
    # videoName = filedir.split("\\")[-1].split("_")[0]
    videoName = 'SAL'
    framesNPYName = "opencv/frames_"+ videoName +".npy"
    if os.path.exists(framesNPYName):
        with open(framesNPYName, 'rb') as f:
            frames = np.load(f)
        return frames
# simplified getMotionVectors
def getMotionVectors(inImgs):
    motionVectors = []
    # File handle
    motionVectorsFileName = "opencv/motionVectors.npy"
    if os.path.exists(motionVectorsFileName):
        with open(motionVectorsFileName, 'rb') as f:
            motionVectors = np.load(f)
        return motionVectors

# getForegroundMask, need to be merged with main
def getForegroundMask(motionVectors, height, width, mode=1):
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


def playVideo(frames):
    for frame in frames:
        # to display with cv2 we need to convert to BGR first
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('rgb_frames',frame)
        cv2.waitKey(25)

if __name__ == '__main__':
    motionVectors = getMotionVectors(None)
    frames = loadRGB(None)
    framesCount, height, width, _ = np.shape(frames) 
    fMasks = getForegroundMask(motionVectors, height, width,2)
    fgs, bgs = getForeAndBack(frames, fMasks)
    playVideo(fgs)