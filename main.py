# https://github.com/naokishibuya/car-finding-lane-lines
import cv2
import numpy as np

# image is expected be in RGB color space


def select_rgb_green(image):

    lower = np.uint8([65, 190,   0])
    upper = np.uint8([178, 255, 175])
    mask = cv2.inRange(image, lower, upper)
    return mask


def morphological_transformation(img, kernel_shape, iterations, fct, img_format=cv2.IMREAD_COLOR):
    if not isinstance(kernel_shape, tuple) and not isinstance(kernel_shape, int):
        raise TypeError("shape must be int or tuple")

    if isinstance(img, str):
        img = cv2.imread(img, img_format)

    if kernel_shape is int:
        kernel_shape = (kernel_shape, kernel_shape)

    kernel = np.ones(kernel_shape, np.uint8)

    erosion = fct(img, kernel=kernel, iterations=iterations)
    return erosion


def to_gray_scale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def gaussian_smooth(img, kernel_shape=25):
    return cv2.GaussianBlur(img, (kernel_shape, kernel_shape), 0)



def canny(img, low_thress=0, high_thress=10):
    return cv2.Canny(img, low_thress, high_thress)


def hough_lines(img):
    """
    `image` should be the output of a Canny transform.

    Returns hough lines (not the image with lines)
    """

    return cv2.HoughLinesP(img, rho=1, theta=np.pi / 180, threshold=5, minLineLength=15, maxLineGap=12)


def show(img, name):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 1200, 1200)
    cv2.imshow(name, img)
    cv2.waitKey()


def find_close(contour1, countour2):
    row1, row2 = contour1.shape[0], countour2.shape[0]
    for i in range(row1):
        for j in range(row2):
            dist = np.linalg.norm(contour1[i] - countour2[j])
            if abs(dist) < 50:
                return True
            elif i == row1 - 1 and j == row2 - 1:
                return False


# generalized hough lines

from collections import defaultdict
from scipy.ndimage.filters import sobel
import matplotlib.pyplot as plt
from scipy.misc import imread


def gradient_orientation(image):
    '''
    Calculate the gradient orientation for edge point in the image
    '''

    dx = sobel(image, axis=0, mode='constant')
    dy = sobel(image, axis=1, mode='constant')
    gradient = np.arctan2(dy, dx) * 180 / np.pi

    return gradient


def build_r_table(image, origin):
    '''
    Build the R-table from the given shape image and a reference point
    '''

    gradient = gradient_orientation(image)

    r_table = defaultdict(list)
    for (i, j), value in np.ndenumerate(edges):
        if value:
            r_table[gradient[i, j]].append((origin[0] - i, origin[1] - j))

    return r_table


def accumulate_gradients(r_table, img):
    '''
    Perform a General Hough Transform with the given image and R-table
    '''
    edges = img

    gradient = gradient_orientation(edges)

    accumulator = np.zeros(img.shape)
    for (i, j), value in np.ndenumerate(edges):
        if value:
            for r in r_table[gradient[i, j]]:
                accum_i, accum_j = i + r[0], j + r[1]
                if accum_i < accumulator.shape[0] and accum_j < accumulator.shape[1]:
                    accumulator[accum_i, accum_j] += 1

    return accumulator


def general_hough_closure(reference_image, image):
    '''
    Generator function to create a closure with the reference image and origin
    at the center of the reference image

    Returns a function f, which takes a query image and returns the accumulator
    '''
    referencePoint = (reference_image.shape[0] / 2, reference_image.shape[1] / 2)
    r_table = build_r_table(reference_image, referencePoint)

    # def f(query_image):
    #     return accumulate_gradients(r_table, query_image)

    return(accumulate_gradients(r_table, image))

def n_max(a, n):
    '''
    Return the N max elements and indices in a
    '''
    indices = a.ravel().argsort()[-n:]
    indices = (np.unravel_index(i, a.shape) for i in indices)
    return [(a[i], i) for i in indices]



if __name__ == '__main__':
    from image_processing import *

    sugarCaneImages = SugarCanePreProcessing('caca_01.jpg', "cana.jpg", True)
    pre_processed_image = sugarCaneImages()
    final = SugarCaneLineFinder(pre_processed_image, "pos_cana.jpg", True)
    final('caca_01.jpg')

    # sugarCaneImages.save(sugarCaneImages.gaussian_smooth)

    # sugarCaneImages.show("plain")

    image = cv2.imread('caca_01.jpg')

    # show(image, 'plain image')

    green_mask = select_rgb_green(image)


    # show(green_mask, 'green filter')

    cv2.imwrite("green_filter.jpg", green_mask)

    erode_callback = cv2.erode

    erosion = morphological_transformation(green_mask, 1, 1, erode_callback)
    # show(erosion, 'erosion')

    cv2.imwrite("erosion.jpg", erosion)

    dilation = morphological_transformation(erosion, 3, 1, cv2.dilate)
    # show(dilation, 'dilation')

    cv2.imwrite("dilation.jpg", dilation)


    smoothed_image = gaussian_smooth(dilation)
    # show(smoothed_image, "Gaussian Smooth")

    cv2.imwrite("smoothed_image.jpg", smoothed_image)


    edges = canny(smoothed_image)
    # show(edges, 'edge detection')

    cv2.imwrite("edges.jpg", edges)

    im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0,255,0), 3)

    print(len(contours))

    # show(image, "countours")
    cv2.imwrite("countours.jpg", image)


    # length = len(contours)
    # status = np.zeros((length, 1))
    # 
    # for i, cnt1 in enumerate(contours):
    #     x = 1
    #     if i != length-1:
    #         for j, cnt2 in enumerate(contours[i+1:]):
    #             x = x + 1
    #             dist = find_close(cnt1, cnt2)
    #             if dist is True:
    #                 val = min(status[i], status[x])
    #                 status[x] = status[i] = val
    #             else:
    #                 try:
    #                     if status[x] == status[i]:
    #                         status[x] = i+1
    #                 except Exception as e:
    #                     print(e)
    # unified = []
    # maximum = int(status.max()) + 1
    # for i in range(maximum):
    #     pos = np.where(status == i)[0]
    #     if pos.size != 0:
    #         cont = np.vstack(contours[i] for i in pos)
    #         hull = cv2.convexHull(cont)
    #         unified.append(hull)

    # cv2.drawContours(image, unified, -1, (0,255,0), 2)
    # show(image, "countours")
    cv2.imwrite("countours.jpg", image)





    # lines = hough_lines(edges)
    # for i in lines:
    #     x1, y1, x2, y2 = i[0]
    #
    #     cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # print(lines)
    # teste = []
    # for i in lines:
    #     x1, y1, x2, y2 = i[0]
    #     teste.append([x1, y1])
    #     teste.append([x2, y2])

    # print(teste)
    # teste = (np.sort(teste, axis=1))
    # teste = np.int32([teste])
    # cv2.polylines(image, np.array(teste), 3, (0, 255, 0))

    # show(image, "lines")


    c = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # show(cv2.drawContours(image, c[1], 1, (0, 255, 0), 3), 'teste')

    # for x1, y1, x2, y2 in lines:
    #     cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0) ,2)
    #     show(image, "lines")
    # print(gradient_orientation(edges))
    # teste = (general_hough_closure(edges))
    # fct = general_hough_closure(edges, image)

    # test_general_hough(general_hough_closure, image)

    cv2.destroyAllWindows()

