# https://github.com/naokishibuya/car-finding-lane-lines
import cv2
import numpy as np

img = cv2.imread('DJI_0065.jpg', 0)

kernel = np.ones((3, 3), np.uint8)

erosion = cv2.erode(img, kernel, iterations=1)

cv2.namedWindow('erosion', cv2.WINDOW_NORMAL)
cv2.resizeWindow('erosion', 600,600)
cv2.imshow('erosion', erosion)

k = cv2.waitKey()

kernel2 = np.ones((3, 3), np.uint8)

dilation = cv2.dilate(erosion, kernel2, iterations=1)

cv2.namedWindow('dilation', cv2.WINDOW_NORMAL)
cv2.resizeWindow('dilation', 600,600)
# cv2.imshow('erosion', img)
cv2.imshow('dilation', dilation)

# cv2.imshow("teste", img)
k = cv2.waitKey()




cv2.destroyAllWindows()