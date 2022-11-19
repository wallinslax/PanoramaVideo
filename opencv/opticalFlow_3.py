# Importing libraries
import cv2
import numpy as np
import os.path

frames = []
# File handle
# videoName = filedir.split("\\")[-1].split("_")[0]
videoName = 'Stairs'
framesNPYName = "opencv/frames_"+ videoName +".npy"
if os.path.exists(framesNPYName):
    with open(framesNPYName, 'rb') as f:
        frames = np.load(f)

# Reading the first frame
frame1 = frames[0]
# Convert to gray scale
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
# Create mask
hsv_mask = np.zeros_like(frame1)
# Make image saturation to a maximum value
hsv_mask[..., 1] = 0
# hsv_mask[..., 0] = 255
# Till you scan the video
for fc in range(1, len(frames)):

# Capture another frame and convert to gray scale
    frame2 = frames[fc]
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Optical flow is now calculated
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # Compute magnite and angle of 2D vector
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # Set image hue value according to the angle of optical flow
    hsv_mask[..., 0] = ang * 180 / np.pi / 2
    h = hsv_mask[..., 0]
    mean = np.mean(hsv_mask[..., 0]).astype('uint8')
    mean_arr = np.full_like(hsv_mask[..., 0], mean)
    result_arr = abs(hsv_mask[..., 0] - mean_arr)
    # Set value as per the normalized magnitude of optical flow
    hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # # filtered out background part, might be specific for SAL.mp4 only 
    hsv_mask[(result_arr < 210)] = 0
    # hsv_mask[..., 1] = np.select([result_arr <100],[0],hsv_mask[..., 1])
    # hsv_mask[..., 2] = np.select([result_arr <100],[0],hsv_mask[..., 2])
    # hsv_mask[(hsv_mask[..., 2] < 3)] = 0
    # Convert to rgb
    rgb_representation = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)

    cv2.imshow('frame2', rgb_representation)
    kk = cv2.waitKey(20) & 0xff
    # Press 'e' to exit the video
    if kk == ord('e'):
        break
    # Press 's' to save the video
    elif kk == ord('s'):
        cv2.imwrite('Optical_image.png', frame2)
        cv2.imwrite('HSV_converted_image.png', rgb_representation)
    prvs = next

cv2.destroyAllWindows()
