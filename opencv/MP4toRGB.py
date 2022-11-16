import cv2
import numpy as np
from PIL import Image
import os.path
import sys

videoname = sys.argv[1]
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, '../video/'+videoname+'.mp4')
frames = []

def mp4toRGGB():
    cap = cv2.VideoCapture(filename)
    count = 0
    while (cap.isOpened()):
        # Get a video frame
        hasFrame, frame = cap.read()

        
        if hasFrame == True:
            ## Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height = frame.shape[0]
            width = frame.shape[1]
            frame = np.clip(frame,0,255).astype(np.uint8)
            frame.tofile(os.path.join(dirname, '../rgb/'+videoname+"/"+videoname+"_"+str(count).zfill(5)+".rgb"))
            count += 1

            # frames.append(np.reshape(frame.ravel(),(height, width,3)))
            #print(frame.ravel())
            #print(len(frame.ravel()))
            #print(frame[0][1])

        else:
            break

    cap.release()


# def numpy2pil(np_array: np.ndarray) -> Image:
#     """
#     Convert an HxWx3 numpy array into an RGB Image
#     """

#     assert_msg = 'Input shall be a HxWx3 ndarray'
#     assert isinstance(np_array, np.ndarray), assert_msg
#     assert len(np_array.shape) == 3, assert_msg
#     assert np_array.shape[2] == 3, assert_msg

#     img = Image.fromarray(np_array, 'RGB')
#     return img


if __name__ == '__main__':
    os.mkdir(os.path.join(dirname, '../rgb/'+videoname))
    mp4toRGGB()
    
    # print(frames[-1])
    # img = numpy2pil(frames[-1])
    # img.show()