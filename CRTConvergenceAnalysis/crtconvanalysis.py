import sys
import cv2
import numpy as np
import time
import math
from matplotlib import pyplot as plt

def largestContourCenter(iimg, low, high):
	return mlargestContourCenter(iimg, low, high, (), ())

def mlargestContourCenter(iimg, low, high, low2, high2):
	kernel = np.ones((5,5), np.uint8)

	# Convert BGR to HSV
	try:
		ihsv = cv2.cvtColor(iimg, cv2.COLOR_BGR2HSV)
	except:
		return (-1, -1, -1)
	# Threshold the HSV image to get only blue colors
	maskimg = cv2.inRange(ihsv, low, high)

	# Check for an extra hsv range
	if len(low2) > 0 and len(high2) > 0:
		maskimg2 = cv2.inRange(ihsv, low2, high2)
		# Join mask
		maskimg += maskimg2

	# Identify/Calculate center of point

	# Close holes within possible contours
	maskimg = cv2.morphologyEx(maskimg, cv2.MORPH_CLOSE, kernel)

	# Identify contours
	cntrs, hierarchy = cv2.findContours(maskimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

	#print(cntrs)

	#cv2.imshow("test", maskimg)
	#cv2.waitKey(0)

	if len(cntrs) > 0:
		# Search for largest contour
		totalArea = 0
		maxArea = 0
		cntr = cntrs[0]
		count = 0
		for n in cntrs:
			area = cv2.contourArea(n)
			totalArea += area
			if area > maxArea:
				maxArea = area
				#print(index)
				cntr = n

		#print("New Contour:")
		#print(maxArea)
		#print(cv2.contourArea(cntr))

		# Check for very small contours (divide 0 error)
		M = cv2.moments(cntr)
		if M["m00"] == 0:
			print("Fail")
			return (-1,-1,-1)
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
		return (cX,cY, totalArea)

		# Identify blue contours
	else:
		#print("No contours found")
		return (-1,-1, -1)



# Read File name
file = sys.argv[1]

scale = sys.argv[2]

# Opencv frame
#img = cv2.imread(file)
#cap = cv2.VideoCapture(2)
cap = cv2.VideoCapture(file)

ret, img = cap.read()

finaltest = img.copy()

scale_percent = int(scale) # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# Resize image
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

finalresized = img.copy()

cv2.imshow("Resized Image", img)

# Convert to Grayscale
mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Treshold to remove dark areas
mask = cv2.inRange(mask, 230, 255)

# Dilate the image a bit to fill any holes
kernel = np.ones((3,3), np.uint8)
mask = cv2.dilate(mask, kernel, iterations=2)
mask = cv2.erode(mask, kernel, iterations=1)
mask = cv2.dilate(mask, kernel, iterations=2)
mask = cv2.erode(mask, kernel, iterations=1)

#cv2.imshow("Threshold Image", mask)


# Find all contours
contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# Draw contours to screen
cv2.drawContours(img, contours, -1, (0,255,0), 3)

cv2.imshow("Initial Processed Image", img)

# Calculate center points for all contours
gridpoints = []
for n in contours:
	M = cv2.moments(n)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])
	gridpoints.append((cX, cY))


# Define range of blue color in HSV
lower_white = np.array([0,0,100])
upper_white = np.array([255,60,255])
lower_blue = np.array([100,50,50])
upper_blue = np.array([135,255,255])
lower_red = np.array([165,50,50])
upper_red = np.array([185,255,255])
lower_red2 = np.array([0,50,50])
upper_red2 = np.array([15,255,255])

# Extracting individual points
for p in gridpoints:
	xt = p[0]
	yt = p[1]
	#xt = gridpoints[0][0]
	#yt = gridpoints[0][1]


	# Draw Box around target point
	cv2.rectangle(finalresized, (xt-10,yt-10), (xt+10, yt+10), (0,255,0), 1)


	# Extract individual point into new frame object
	revscale = int(100/int(scale))
	newimg = finaltest[(yt * revscale - 50):(yt * revscale + 50), (xt * revscale - 50):(xt * revscale + 50)]

	# Find the largest white contour and calculate center
	wx, wy, wta = largestContourCenter(newimg, lower_white, upper_white)
	# Plot if not invalid
	if wx != -1:
		cv2.circle(newimg,(wx,wy), 5, (0,255,0), -1)

		# Find the largest blue contour and calculate center
		bx, by, bta = largestContourCenter(newimg, lower_blue, upper_blue)
		# Plot if not invalid
		if bx != -1:
			cv2.arrowedLine(newimg, (wx,wy), (bx, by), (255,0,0), 2, tipLength = 0.5)
			cv2.circle(newimg,(bx,by), 5, (255,255,0), -1)

		# Find the largest red contour and calculate center (with 2 different HSV color ranges
		rx, ry, rta = mlargestContourCenter(newimg, lower_red, upper_red, lower_red2, upper_red2)
		# Plot if not invalid
		if rx != -1:
			cv2.arrowedLine(newimg, (wx,wy), (rx, ry), (0,0,255), 2, tipLength = 0.5)
			cv2.circle(newimg,(rx,ry), 5, (0,255,255), -1)

	try:
		cv2.imshow("Resized Image", finalresized)
		cv2.imshow("Point", mask)
		cv2.imshow("Single", newimg)
	except:
		print("frame damaged")

	# Erase target box
	cv2.rectangle(finalresized, (xt-10,yt-10), (xt+10, yt+10), (0,0,0), 1)


	#cv2.imshow("Single", newimg)
	cv2.waitKey(0)


print(finaltest.shape[:2])
#print(newimg.shape[:2])






