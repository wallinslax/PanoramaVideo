import cv2 as cv
import numpy as np
from PIL import Image
import sys, os.path, argparse
from os import listdir
from os.path import isfile, join
from tqdm import tqdm

def mp4toRGB(filepath: str):
    videoName = filepath.split("/")[-1].split(".")[0]
    ## [capture]
    capture = cv.VideoCapture(filepath)
    if not capture.isOpened():
        print('Unable to open: ' + filepath)
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

        height = frame.shape[0]
        width = frame.shape[1]
        # frames.append(np.reshape(frame.ravel(),(height, width,3)))
        #print(frame.ravel())
        #print(len(frame.ravel()))
        #print(frame[0][1])

    capture.release()
    nFrame = len(frames)
    return frames, videoName

def loadRGB(filedir):
    global width, height, nFrame, videoName
    videoName = filedir.split("\\")[-1].split("_")[0]
    tmp = filedir.split("_")
    width, height, nFrame = int(tmp[-3]), int(tmp[-2]), int(tmp[-1])

    frames = []
    # File handle
    framesNPYName = "cache/frames_"+ videoName +".npy"
    if os.path.exists(framesNPYName):
        with open(framesNPYName, 'rb') as f:
            frames = np.load(f)
        return frames,videoName
    
    rgbNames = [f for f in listdir(filedir) if isfile(join(filedir, f))]
    # rgbNames = rgbNames[0:30] # smaller dataset
    
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

    # File handle
    with open(framesNPYName, 'wb') as f:
        np.save(f,frames)
    
    return frames, videoName

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

def playVideo(frames, wait=30):
    for frame in frames:
        # to display with cv2 we need to convert to BGR first
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        frame = cv.resize(frame, (960, 540))  
        cv.imshow('rgb_frames',frame)
        keyboard = cv.waitKey(wait)
        if keyboard == 'q' or keyboard == 27:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filepath", type=str, default="./video/SAL.mp4",help="specify video file name")
    parser.add_argument("-d", "--filedir", type=str, default="C:\\video_rgb\\SAL_490_270_437",help="specify rgb directory")
    args = parser.parse_args()

    # inImgs = mp4toRGB(args.filepath)
    inImgs = loadRGB(args.filedir)
    # Debug View-----------------------------------
    # sampleImg = numpy2pil(inImgs2[-1])
    # sampleImg.show()
    