# https://github.com/naokishibuya/car-finding-lane-lines
import cv2
import numpy as np

# image is expected be in RGB color space


def select_rgb_green(image):

    lower = np.uint8([65, 190,   0])
    upper = np.uint8([178, 255, 175])
    green_mask = cv2.inRange(image, lower, upper)
    return green_mask


def morpholocial_transformation(img, shape, iterations, fct, img_format=0):
    if not isinstance(shape, tuple):
        raise TypeError("shape must be tuple")
    if isinstance(img, str):
        img = cv2.imread(img, img_format)
    kernel = np.ones(shape, np.uint8)
    erosion = fct(img, kernel=kernel, iterations=iterations)
    return erosion


def show(img, name):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 800, 800)
    cv2.imshow(name, img)
    cv2.waitKey()





if __name__ == '__main__':
    image = select_rgb_green(cv2.imread('caca_02.jpg'))
    show(image, 'green filter')
    erode_callback = cv2.erode

    # erosion = morpholocial_transformation(image, (2, 2), 1, erode_callback)
    # show(erosion, 'erosion')

    dilation = morpholocial_transformation(image, (3, 3), 1, cv2.dilate)
    show(dilation, 'dilation')

    cv2.destroyAllWindows()

