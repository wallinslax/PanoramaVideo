import os
import cv2
import numpy as np
import sys
import math
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
def getMotionVectors(motionVectorsFileName = "cache/motionVectors_SAL_437small.npy"):
    motionVectors = []
    # File handle 
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
    elif mode == 4:
        fMasks = getForegroundMask_withHOG(frames, height, width)

    return fMasks

def getForegroundMask_withHOG(frames, height, width):
    fMasks = []
    # initialize the HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    for frame in tqdm(frames):
        fMask = np.zeros((height, width)).astype('uint8')
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        boxes, weights = hog.detectMultiScale(frame, winStride=(8,8) )
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        for (xA, yA, xB, yB) in boxes:
            fMask[yA:yB,xA:xB] = 255
        fMasks.append(fMask)
    return fMasks


def getForegroundMask_withOF(frames, height, width):
    fMasks = []
    fMasks.append(np.zeros((height, width)).astype('uint8'))
    # Reading the first frame
    frame1 = frames[0]
    # Convert to gray scale
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    # Create mask
    hsv_mask = np.zeros_like(frame1)
    # Make image saturation to a maximum value
    hsv_mask[..., 1] = 255
    for fc in tqdm(range(1, len(frames))):
        fMask = np.zeros((height, width)).astype('uint8')
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
        fMask[labels.reshape((height, width)) == 1] = 255

        fMasks.append(fMask)
    return fMasks

def getAngMag(dx, dy):
    angle = 0.0
    if  dx == 0:
        angle = math.pi / 2.0
        if  dy == 0 :
            angle = 0.0
        elif dy < 0 :
            angle = 3.0 * math.pi / 2.0
    elif dx > 0 and dy > 0:
        angle = math.atan(dx / dy)
    elif  dx > 0 and  dy < 0 :
        angle = math.pi / 2 + math.atan(-dy / dx)
    elif  dx < 0 and dy < 0 :
        angle = math.pi + math.atan(dx / dy)
    elif  dx < 0 and dy > 0 :
        angle = 3.0 * math.pi / 2.0 + math.atan(dy / -dx)
    return (angle * 180 / math.pi),math.sqrt(dx**2 + dy **2)

# getForegroundMask_withMV, need to be merged with main
def getForegroundMask_withMV(motionVectors, height, width, mode=1, macroSize=16):
    # mode:
    # 1= Mode
    # 2= K-Mean

    fMasks = []
    fMasks.append(np.zeros((height, width)).astype('uint8'))

    mvCount, mvHeight, mvWidth, _ = np.shape(motionVectors)
    
    if mode == 1:
        for motionVector in tqdm(motionVectors):
            fMask = np.zeros((height, width, 1)).astype('uint8')
            colMode = stats.mode(motionVector, keepdims=True)
            overallMode = stats.mode(colMode[0], axis=1, keepdims=True)[0][0][0]
            xMode, yMode = overallMode
            angMode, magMode = getAngMag(xMode, yMode)
            magThreshold = 5
            angThreshold = 120

            for vCol in range(mvHeight):
                for vRow in range(mvWidth):
                    x, y = motionVector[vCol][vRow]
                    ang, mag = getAngMag(x, y)
                    if (abs(ang-angMode) > angThreshold):
                        for i in range(macroSize):
                            for j in range(macroSize): 
                                fMask[vCol*macroSize+i][vRow*macroSize+j] = 255
            fMasks.append(fMask)

    if mode == 2:
        for motionVector in tqdm(motionVectors):
            fMask = np.zeros((height, width)).astype('uint8')
            mvs = motionVector.reshape(-1,2)
            kmeans = KMeans(n_clusters=2)
            kmeans.fit(mvs)
            labels = kmeans.labels_
            labels_mode = stats.mode(labels, keepdims=False)[0]
            if labels_mode == 1:
                labels = labels+1
            labels = labels.reshape((height//macroSize, width//macroSize, 1))
            for vCol in range(mvHeight):
                for vRow in range(mvWidth):
                    if labels[vCol][vRow][0] == 1:
                        for i in range(macroSize):
                            for j in range(macroSize): 
                                fMask[vCol*macroSize+i][vRow*macroSize+j] = 255
            fMasks.append(fMask)

    return fMasks
    
# getForeAndBack, need to be merged with main
def getForeAndBack(frames, motionVectors, mode=3):
    fMasks = getForegroundMask(frames, motionVectors,mode)
    black = np.zeros_like(frames[0])
    black = black+255
    fgs = []
    bgs = []    
    framesCount, height, width, _ = np.shape(frames) 
    for n in tqdm(range(framesCount)):
        copy = frames[n].copy()
        fg = cv2.bitwise_and(copy, copy, mask = fMasks[n])
        bg = cv2.bitwise_not(black, copy, mask = fMasks[n])
        # fg = np.zeros((height, width, 3)).astype('uint8')
        # bg = np.zeros((height, width, 3)).astype('uint8')
        
        # for h in range(height):
        #     for w in range(width):
        #         if fMasks[n][h][w][0] == 1:
        #             fg[h][w][0:3] = frames[n][h][w][0:3]
        #         else:
        #             bg[h][w][0:3] = frames[n][h][w][0:3]
        fgs.append(fg)
        bgs.append(bg)

    return fgs,bgs

def getForeground_Naive(inImgs,motionVectors,macroSize=16):
    nFrame, height, width, _ = np.shape(inImgs) 
    nRow, nCol = height//macroSize, width//macroSize
    foreImgs = []
    for fIdx in range(nFrame):
        curFrame = np.copy(inImgs[fIdx])
        motionVectorsPerFrame = motionVectors[fIdx]
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

def visualizeMotionVector(motionVectors):
    mvCount, mvHeight, mvWidth, _ = np.shape(motionVectors)
    hsv = np.zeros((mvHeight, mvWidth,3))
    hsv[..., 1] = 255
    res = []
    for motionVector in tqdm(motionVectors):
        angs = np.zeros((mvHeight, mvWidth))
        mags = np.zeros((mvHeight, mvWidth))
        for vCol in range(mvHeight):
            for vRow in range(mvWidth):
                x, y = motionVector[vCol][vRow]
                ang, mag = getAngMag(x, y)            
                angs[vCol][vRow] = ang
                mags[vCol][vRow] = mag
        hsv[..., 0] = angs/2
        hsv[..., 2] = cv2.normalize(mags, None, 0, 255, cv2.NORM_MINMAX)

        result = hsv.copy()

        # define range of red and green color in HSV
        lower_red = np.array([0,100,20])
        upper_red = np.array([10,255,255])
        lower_red_2 = np.array([160,100,20])
        upper_red_2 = np.array([179,255,255])
        lower_green = np.array([35,100,20])
        upper_green = np.array([65,255,255])

        mask_1 = cv2.inRange(hsv, lower_red, upper_red)
        mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
        mask_3 = cv2.inRange(hsv, lower_green, upper_green)
        fullmask = mask_1 + mask_2 + mask_3
        result = cv2.bitwise_not(result, result, mask=fullmask)

        rgb = cv2.cvtColor(result.astype('uint8'), cv2.COLOR_HSV2RGB)
        res.append(rgb)
    
    playVideo(res, 30)


if __name__ == '__main__':
    # frames = loadRGB(None)
    frames, videoName = mp4toRGB(filepath="./video/SAL.mp4")
    motionVectors = getMotionVectors(motionVectorsFileName = "cache/motionVectors_SAL_437.npy")
    frames = frames[:]
    motionVectors = motionVectors[:]
    # visualizeMotionVector(motionVectors)
    fgs, bgs = getForeAndBack(frames, motionVectors, 4)
    playVideo(fgs, 30)
    playVideo(bgs, 30)