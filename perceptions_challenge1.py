import cv2
import numpy as np

# reading
image = cv2.imread('red_cones.png')
h, w = image.shape[:2]

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# filtering out the cones using orange
mask = cv2.inRange(hsv, (0, 170, 180), (25, 255, 255))
orange = cv2.bitwise_and(image, image, mask=mask)

# grayscale and reshape
gray = cv2.cvtColor(orange, cv2.COLOR_BGR2GRAY)
pixels = gray.reshape((-1, 1))

# fixing parameter issue
pixels = np.float32(pixels)

# k-means clustering
num_clusters = 3
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS
compactness, labels, centers = cv2.kmeans(pixels, num_clusters, None, criteria, 10, flags)

with_centroids = centers[labels.flatten()].reshape(gray.shape)
# I dilated the kernel so that the small details aren't picked up on, overcomplicating it
# 20x20 was a satisfiable amount
kernel = np.ones((20,20),np.uint8)
dilated_contours = cv2.dilate(with_centroids, kernel, iterations=1)

# Finding contours of orange items (cones), and printing it for testing purposes
# prints the number contours, which adds up with the cones
contours, hierarchy = cv2.findContours(dilated_contours.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("Number of contours found:", len(contours))

# sorting contours from left to right so that left and right can be displayed
contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])

# split into left and right cone "columns" using contours array
midpoint = int(len(contours) / 2)
left_contours = contours[:midpoint]
right_contours = contours[midpoint:]

# get the best fit line for left side using fitLine()
points = np.vstack([c[:, 0] for c in left_contours])
vx, vy, cx, cy = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
lefty = int((-cx*vy/vx) + cy)
righty = int(((w-cx)*vy/vx)+cy)
#print red lines on left side
cv2.line(image,(w-1,righty),(0,lefty),(0,0,255),4)

# get the best fit line for right side using fitLine()
points = np.vstack([c[:, 0] for c in right_contours])
vx, vy, cx, cy = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
lefty = int((-cx*vy/vx) + cy)
righty = int(((w-cx)*vy/vx)+cy)
#print red lines on right side
cv2.line(image,(w-1,righty),(0,lefty),(0,0,255),4)

#save to answer.png and display for testing
cv2.imwrite('answer.png',image)
cv2.imshow('image with lines',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
