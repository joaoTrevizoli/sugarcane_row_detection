# https://github.com/naokishibuya/car-finding-lane-lines
import cv2
import numpy as np


def morpholocial_transformation(path, shape, iterations, img_format=0):
    if not isinstance(shape, tuple):
        raise TypeError("shape must be tuple")
    img = cv2.imread(path, img_format)
    erosion = cv2.erode(img, kernel=kernel, iterations=iterations)
    return erosion


def show(img, name):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 800, 800)
    cv2.imshow(name, img)
    cv2.waitKey()

img = cv2.imread('DJI_0065.jpg', 0)

kernel = np.ones((3, 3), np.uint8)
#
# erosion = cv2.erode(img, kernel, iterations=1)

# cv2.namedWindow('erosion', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('erosion', 600,600)
# cv2.imshow('erosion', erosion)
#
# k = cv2.waitKey()
#
# kernel2 = np.ones((3, 3), np.uint8)
#
# dilation = cv2.dilate(erosion, kernel2, iterations=1)
#
# cv2.namedWindow('dilation', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('dilation', 600,600)
# # cv2.imshow('erosion', img)
# cv2.imshow('dilation', dilation)
#
# # cv2.imshow("teste", img)
# k = cv2.waitKey()




if __name__ == '__main__':
    erode_callback = cv2.erode

    erosion = morpholocial_transformation('DJI_0065.jpg', (3, 3), 1)
    show(erosion, 'erosion')



    # cv2.namedWindow('erosion', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('erosion', 600, 600)
    # cv2.imshow('erosion', erosion)
    #
    # k = cv2.waitKey()
    cv2.destroyAllWindows()

