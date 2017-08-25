import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from functools import wraps


def base_path():
    return os.path.dirname(os.path.abspath(__file__))


def save(func):
    __path = "{}/{}/".format(base_path(), "output_images")

    @wraps(func)
    def image_wrapper(self, *args, **kwargs):
        img = func(self, *args, **kwargs)
        if self.save_mode == True:
            img_name = self.name.split(".")[0]
            f_name = "{}{}_{}.jpg".format(__path, img_name, func.__name__)
            cv2.imwrite(f_name, img)
        return img
    return image_wrapper


class SugarCaneProcessingBase(object):

    __input_path = "{}/{}/".format(base_path(), "base_images")
    __output_path = "{}/{}/".format(base_path(), "output_images")

    def __init__(self, base_image, name, save_mode=False):
        if not os.path.exists(self.__output_path):
            os.makedirs(self.__output_path)

        if isinstance(base_image, str):
            img_path = self.__input_path + base_image
            base_image = cv2.imread(img_path, cv2.IMREAD_COLOR)

        self.base_image = base_image
        self.name = name
        self.save_mode = save_mode

    @classmethod
    def multiple_processor(cls, iterator, save_mode=False):
        for image in iterator:
            if isinstance(image["img"], str):
                img_path = cls.__input_path + image[0]
                image["img"] = cv2.imread(img_path, cv2.IMREAD_COLOR)
            yield cls(image["img"], image["name"], save_mode)

    def opencv_show(self, img=None, name=None, kill_all=True):
        if img is None:
            img = self.base_image
            name = self.name
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, 300, 300)
        cv2.imshow(name, img)
        cv2.waitKey()
        if kill_all:
            cv2.destroyAllWindows()
    
    def matplotlib_show(self, img=None):
        if img is None:
            img = self.base_image
        plt.imshow(img)
        plt.show()


class SugarCanePreProcessing(SugarCaneProcessingBase):

    @save
    def select_rgb_green(self):
        lower = np.uint8([65, 190, 0])
        upper = np.uint8([178, 255, 175])
        mask = cv2.inRange(self.base_image, lower, upper)
        return mask

    @save
    def morphological_transformation(self, img, kernel_shp, iterations, fct):
        if not isinstance(kernel_shp, tuple) and not isinstance(kernel_shp, int):
            raise TypeError("shape must be int or tuple")

        if kernel_shp is int:
            kernel_shp = (kernel_shp, kernel_shp)

        kernel = np.ones(kernel_shp, np.uint8)

        transformed_img = fct(img, kernel=kernel, iterations=iterations)
        return transformed_img

    @save
    def to_gray_scale(self, img=None):
        if img is None:
            img = self.base_image
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    @save
    def gaussian_smooth(self, img=None, kernel_shape=15):
        if img is None:
            img = self.base_image
        return cv2.GaussianBlur(img, (kernel_shape, kernel_shape), 0)

    def __call__(self):
        green_mask = self.select_rgb_green()
        erosion = self.morphological_transformation(green_mask, 1, 1, cv2.erode)
        dilation = self.morphological_transformation(erosion, 3, 1, cv2.dilate)
        return self.gaussian_smooth(dilation)


class SugarCaneLineFinder(SugarCaneProcessingBase):

    def __init__(self, gauss_filtered_image, name, save_mode=False):
        self.__plants = 0
        SugarCaneProcessingBase.__init__(self, gauss_filtered_image, name, save_mode)

    @save
    def canny(self, low_thress=0, high_thress=10):
        return cv2.Canny(self.base_image, low_thress, high_thress)

    @save
    def get_lines(self, output_image):
        __input_path = "{}/{}/".format(os.path.dirname(os.path.abspath(__file__)), "base_images")
        if isinstance(output_image, str):
            img_path = __input_path + output_image
            output_image = cv2.imread(img_path, cv2.IMREAD_COLOR)

        img = self.canny()
        _, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.__plants = len(contours)
        cv2.drawContours(output_image, contours, -1, (0,255,0), 3)
        return output_image

    def __call__(self, output_image):
        return self.get_lines(output_image)
