import matplotlib as mpl
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

#fixing parameter issue
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

#the way I did this was connecting each contour from its center,
# and so the line is staggered. I was also attempting to use the best fit line
# using the cv2.fitLine() function, but I was having issues with that at that moment

# Finding contours of orange items (cones), and printing it for testing
# prints 16 contours, which adds up with the cones + exit signs
contours, hierarchy = cv2.findContours(dilated_contours.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("Number of contours found:", len(contours))


# Sorting contours from left to right
contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])

#split into left and right cone "columns"
midpoint = int(len(contours) / 2)
left_contours = contours[:midpoint]
right_contours = contours[midpoint:]

# lines connecting contours on left side
for i in range(1, len(left_contours)):
    x1 = tuple(left_contours[i-1][left_contours[i-1][:,:,1].argmin()][0]) # Bottom point of previous contour
    x2 = tuple(left_contours[i][left_contours[i][:,:,1].argmin()][0]) # Top point of current contour
    cv2.line(image, x1, x2, (0, 0, 255), 4)

 # lines connecting contours on right side
for i in range(1, len(right_contours)):
    x1 = tuple(right_contours[i-1][right_contours[i-1][:,:,1].argmin()][0]) # Bottom point of previous contour
    x2 = tuple(right_contours[i][right_contours[i][:,:,1].argmin()][0]) # Top point of current contour
    cv2.line(image, x1, x2, (0, 0, 255), 4)   


#save to answer.png and display for testing
cv2.imwrite('answer.png',image)

cv2.imshow('image with lines',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

