import cv2
import numpy as np


class SugarCanePreprocessing():
    def __init__(self, base_image):
        if isinstance(base_image, str):
            base_image = cv2.imread(base_image, cv2.IMREAD_COLOR)
            
        self.base_image = base_image
    
    def morphological_transformation(self, kernel_shp, iterations, fct):
        if not isinstance(kernel_shp, tuple) and not isinstance(kernel_shp, int):
            raise TypeError("shape must be int or tuple")

        if kernel_shp is int:
            kernel_shp = (kernel_shp, kernel_shp)

        kernel = np.ones(kernel_shp, np.uint8)

        transformed_img = fct(self.base_image, kernel=kernel, iterations=iterations)
        return transformed_img

    def to_gray_scale(self, img=self.base_image):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def gaussian_smooth(self, img=self.base_image, kernel_shape=15):
        return cv2.GaussianBlur(img, (kernel_shape, kernel_shape), 0)

    def canny(self, img=self.base_image, low_thress=0, high_thress=10):
        return cv2.Canny(img, low_thress, high_thress)

    def show(self, img, name):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, 1200, 1200)
        cv2.imshow(name, img)
        cv2.waitKey()