import cv2
import numpy as np

img = cv2.imread('DJI_0065.jpg', 0)
kernel = np.ones((3, 3), np.uint8)
print(kernel)
erosion = cv2.erode(img, kernel, iterations=1)

kernel2 = np.ones((2, 2), np.uint8)

dilation = cv2.dilate(erosion, kernel2, iterations=2)

cv2.namedWindow('dilation', cv2.WINDOW_NORMAL)
cv2.resizeWindow('dilation', 600,600)
# cv2.imshow('erosion', img)
cv2.imshow('dilation', dilation)

# cv2.imshow("teste", img)
k = cv2.waitKey(0)

cv2.destroyAllWindows()