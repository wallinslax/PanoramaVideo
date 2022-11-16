import cv2
import numpy as np
from PIL import Image
import os.path

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, '../video/SAL.mp4')
frames = []

def mp4toRGGB():
    cap = cv2.VideoCapture(filename)
    while (cap.isOpened()):
        # Get a video frame
        hasFrame, frame = cap.read()

        
        if hasFrame == True:
            ## Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height = frame.shape[0]
            width = frame.shape[1]
            frames.append(np.reshape(frame.ravel(),(height, width, 3)))
            #print(frame.ravel())
            #print(len(frame.ravel()))
            #print(frame[0][1])

        else:
            break

    cap.release()


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


if __name__ == '__main__':
    mp4toRGGB()
    img = numpy2pil(frames[-1])
    img.show()