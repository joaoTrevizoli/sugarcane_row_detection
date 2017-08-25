# https://github.com/naokishibuya/car-finding-lane-lines
import cv2
import numpy as np


if __name__ == '__main__':
    from sugarcane_line_finder import *

    sugarCaneImages = SugarCanePreProcessing('cana_01.jpg', "cana.jpg", True)
    pre_processed_image = sugarCaneImages()
    final = SugarCaneLineFinder(pre_processed_image, "pos_cana.jpg", True)
    final('cana_01.jpg')

    # sugarCaneImages.save(sugarCaneImages.gaussian_smooth)

    # sugarCaneImages.show("plain")
    #
    # image = cv2.imread('caca_01.jpg')
    #
    # # show(image, 'plain image')
    #
    # green_mask = select_rgb_green(image)
    #
    #
    # # show(green_mask, 'green filter')
    #
    # cv2.imwrite("green_filter.jpg", green_mask)
    #
    # erode_callback = cv2.erode
    #
    # erosion = morphological_transformation(green_mask, 1, 1, erode_callback)
    # # show(erosion, 'erosion')
    #
    # cv2.imwrite("erosion.jpg", erosion)
    #
    # dilation = morphological_transformation(erosion, 3, 1, cv2.dilate)
    # # show(dilation, 'dilation')
    #
    # cv2.imwrite("dilation.jpg", dilation)
    #
    #
    # smoothed_image = gaussian_smooth(dilation)
    # # show(smoothed_image, "Gaussian Smooth")
    #
    # cv2.imwrite("smoothed_image.jpg", smoothed_image)
    #
    #
    # edges = canny(smoothed_image)
    # # show(edges, 'edge detection')
    #
    # cv2.imwrite("edges.jpg", edges)
    #
    # im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(image, contours, -1, (0,255,0), 3)
    #
    # print(len(contours))
    #
    # # show(image, "countours")
    # cv2.imwrite("countours.jpg", image)
    #
    #
    # # length = len(contours)
    # # status = np.zeros((length, 1))
    # #
    # # for i, cnt1 in enumerate(contours):
    # #     x = 1
    # #     if i != length-1:
    # #         for j, cnt2 in enumerate(contours[i+1:]):
    # #             x = x + 1
    # #             dist = find_close(cnt1, cnt2)
    # #             if dist is True:
    # #                 val = min(status[i], status[x])
    # #                 status[x] = status[i] = val
    # #             else:
    # #                 try:
    # #                     if status[x] == status[i]:
    # #                         status[x] = i+1
    # #                 except Exception as e:
    # #                     print(e)
    # # unified = []
    # # maximum = int(status.max()) + 1
    # # for i in range(maximum):
    # #     pos = np.where(status == i)[0]
    # #     if pos.size != 0:
    # #         cont = np.vstack(contours[i] for i in pos)
    # #         hull = cv2.convexHull(cont)
    # #         unified.append(hull)
    #
    # # cv2.drawContours(image, unified, -1, (0,255,0), 2)
    # # show(image, "countours")
    # cv2.imwrite("countours.jpg", image)
    #
    #
    #
    #
    #
    # # lines = hough_lines(edges)
    # # for i in lines:
    # #     x1, y1, x2, y2 = i[0]
    # #
    # #     cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
    #
    # # print(lines)
    # # teste = []
    # # for i in lines:
    # #     x1, y1, x2, y2 = i[0]
    # #     teste.append([x1, y1])
    # #     teste.append([x2, y2])
    #
    # # print(teste)
    # # teste = (np.sort(teste, axis=1))
    # # teste = np.int32([teste])
    # # cv2.polylines(image, np.array(teste), 3, (0, 255, 0))
    #
    # # show(image, "lines")
    #
    #
    # c = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # # show(cv2.drawContours(image, c[1], 1, (0, 255, 0), 3), 'teste')
    #
    # # for x1, y1, x2, y2 in lines:
    # #     cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0) ,2)
    # #     show(image, "lines")
    # # print(gradient_orientation(edges))
    # # teste = (general_hough_closure(edges))
    # # fct = general_hough_closure(edges, image)
    #
    # # test_general_hough(general_hough_closure, image)

    cv2.destroyAllWindows()

