import cv2 as cv
import numpy as np
from PIL import Image
import sys, os.path, argparse
from os import listdir
from os.path import isfile, join
from tqdm import tqdm

macroSize = 16

def mp4toRGB(filename: str):
    ## [capture]
    capture = cv.VideoCapture(filename)
    if not capture.isOpened():
        print('Unable to open: ' + filename)
        exit(0)
    frames = []
    while True:
        ## [Get a video frame]
        ret, frame = capture.read()
        if frame is None:
            break
        
        ## [display_frame_number]
        cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
        cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
                cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
        
        ## [show]
        # cv.imshow('Frame', frame)
        # keyboard = cv.waitKey(30)
        # if keyboard == 'q' or keyboard == 27:
        #     break

        ## [Convert BGR to RGB]
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = np.clip(frame,0,255).astype(np.uint8)
        frames.append(frame)

        # height = frame.shape[0]
        # width = frame.shape[1]
        # frames.append(np.reshape(frame.ravel(),(height, width,3)))
        #print(frame.ravel())
        #print(len(frame.ravel()))
        #print(frame[0][1])

    capture.release()
    return frames

def loadRGB(filedir):
    tmp = filedir.split("_")
    width, height, nFrame = int(tmp[-3]), int(tmp[-2]), int(tmp[-1])
    rgbNames = [f for f in listdir(args.filedir) if isfile(join(filedir, f))]
    # rgbNames = rgbNames[0:30] # smaller dataset
    frames = []
    for rgbName in tqdm(rgbNames):
        frame = np.zeros((height,width,3))
        with open(join(filedir, rgbName), "rb") as f:
            for y in range(height):
                for x in range(width):
                    r = int.from_bytes(f.read(1), "big")
                    g = int.from_bytes(f.read(1), "big")
                    b = int.from_bytes(f.read(1), "big")
                    frame[y][x] = [r,g,b]
        # print(frame)
        frame = np.clip(frame,0,255).astype(np.uint8)
        frames.append(frame)
    return frames

def saveFramesRGB(filename: str,frames):
    videoname = filename.split('/')[-1]
    dirname = os.path.dirname(__file__)
    if not os.path.exists(os.path.join(dirname,'../rgb/'+ videoname)):
        os.makedirs(os.path.join(dirname,'../rgb/'+ videoname))

    for fIdx, frame in enumerate(frames):
        frame.tofile(os.path.join(dirname, '../rgb/'+videoname+"/"+videoname+"_"+str(fIdx).zfill(5)+".rgb"))

def numpy2pil(np_array: np.ndarray) -> Image:
    """
    Convert an HxWx3 numpy array into an RGB Image
    """
    assert_msg = 'Input shall be a HxWx3 ndarray'
    assert isinstance(np_array, np.ndarray), assert_msg
    assert len(np_array.shape) == 3, assert_msg
    assert np_array.shape[2] == 3, assert_msg

    img = Image.fromarray(np_array, 'RGB')
    return img

def getMotionVectors(inImgs):
    motionVectors = []
    for fIdx in range(1,len(inImgs)):
        curFrame, prvFrame = inImgs[fIdx], inImgs[fIdx - 1]
        motionVectorsPerFrame = getMotionVectorsPerFrame(curFrame, prvFrame)
        motionVectors.append(motionVectorsPerFrame)
    return motionVectors

def getMotionVectorsPerFrame(curFrame, prvFrame):
    height,width = curFrame.shape[0], curFrame.shape[1]
    nRow, nCol = height//macroSize, width//macroSize
    k = macroSize
    motionVectorsPerFrame = np.empty((nRow,nCol,2))
    motionVectorMADs = np.ones((nRow,nCol))* float('inf')
    for r in tqdm(range(nRow)):
        for c in tqdm(range(nCol),leave=False):
            motionVectorsPerFrame[r][c] = [0,0]
            for vec_x in range(-k, k):
                for vec_y in range(-k, k):
                    error = MAD(curFrame,prvFrame, vec_x, vec_y, r, c)
                    if error < motionVectorMADs[r][c]:
                        motionVectorMADs[r][c] = error
                        motionVectorsPerFrame[r][c] = [vec_x, vec_y]

    print(motionVectorsPerFrame)
    return motionVectorsPerFrame
    
def MAD(curFrame, prvFrame, vec_x, vec_y, r, c):
    height,width = curFrame.shape[0], curFrame.shape[1]
    base_x, base_y = c * macroSize, r * macroSize
    subError = 0
    for x in range(macroSize):
        for y in range(macroSize):
            if base_x + x + vec_x < 0 or base_x + x + vec_x >= width\
            or base_y + y + vec_y < 0 or base_y + y + vec_y >= height:
                return float('inf')
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
    
    # inImgs = mp4toRGB(args.filepath)
    inImgs2 = loadRGB(args.filedir)
    # motionVectors = getMotionVectors(inImgs2)

    # Debug View-----------------------------------
    # sampleImg = numpy2pil(inImgs2[-1])
    # sampleImg.show()
    for inImg in inImgs2:
        ## [show]
        inImg = cv.cvtColor(inImg, cv.COLOR_RGB2BGR)
        cv.imshow('Frame', inImg) # color wired HELP!!
        keyboard = cv.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break