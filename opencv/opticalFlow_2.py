# Importing libraries
import cv2
import numpy as np
import os.path

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, '../video/SAL.mp4')
# Capturing the video file 0 for videocam else you can provide the url
capture = cv2.VideoCapture(filename)

# Reading the first frame
_, frame1 = capture.read()
# Convert to gray scale
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
# Create mask
hsv_mask = np.zeros_like(frame1)
# Make image saturation to a maximum value
hsv_mask[..., 1] = 0

# Till you scan the video
while(1):
	
	# Capture another frame and convert to gray scale
	_, frame2 = capture.read()
	next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

	# Optical flow is now calculated
	flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
	# Compute magnite and angle of 2D vector
	mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
	# Set image hue value according to the angle of optical flow
	hsv_mask[..., 0] = ang * 180 / np.pi / 2
	# Set value as per the normalized magnitude of optical flow
	hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
	# filtered out background part, might be specific for SAL.mp4 only 
	hsv_mask[(hsv_mask[..., 0] > 75) & (hsv_mask[..., 0] < 105)] = 0
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

capture.release()
cv2.destroyAllWindows()
