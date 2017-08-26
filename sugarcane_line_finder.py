"""""Sugar Cane line finder

Copyright 2017, Lab804

.. module: sugarcane_line_finder
   :platform: Unix, Windows, macOS
   :synopsis: An openCV application for crop rows detection

.. moduleauthor:: João Trevizoli Esteves <joao@lab804.com.br>
"""

import cv2
import os
import tabulate
import numpy as np
from matplotlib import pyplot as plt
from functools import wraps


def base_path():
    """Gets base file path

    Returns
    -------
    :param base_path:
        str with this file path.
    :rtype: str
    """
    return os.path.dirname(os.path.abspath(__file__))


def save(func):
    """Image saving decorator.

    Decorator for wrapping openCV functionalities and save them
    at the folder output_images

    Parameters
    ----------
    :param func:
        Function pointer.
    :type: func

    Returns
    -------
    :returns image_wrapper:
        Function pointer returns
    :rtype: numpy array

    Notes
    -----
    The image names are going to be renamed based on the wrapped method or
    function

    """
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
    """Processors base class

    Parameters
    ----------
    :param base_image:
        OpenCV base image file name or numpy array
    :type: str or numpy array

    :param name:
        Output file name
    :type: str

    :param save_mode:
        Image logging, default=False
    :type: bool
    """
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

    def opencv_show(self, img=None, name=None, kill_all=True):
        """Display image through openCV.

        Display images using openCV resources

        Parameters
        ----------
        :param img:
            opened openCV image, default=None
        :type: numpy array

        :param kill_all:
            Kill all windows on click
        :type: bool

        :param name:
            Window name
        :type: str
        """
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
        """Display image through matplotlib.

        Display images using matplotlib resources

        Parameters
        ----------
        :param img:
            opened openCV image
        :type: numpy array

        """
        if img is None:
            img = self.base_image
        plt.imshow(img)
        plt.show()


class SugarCanePreProcessing(SugarCaneProcessingBase):
    """Crop images pre-processor.

    Pre-processor for crop images

    """
    @classmethod
    def multiple_processor(cls, iterator, save_mode=False):
        """Class method to receive iterators.

        classmethod for receiving and dealing with iterators.

        Parameters
        ----------
        :param iterator:
            iterator with dict in the form:
                {"img": numpy array,
                 "name": str
                 }
        :type: list, tuple or iterator

        :param save_mode:
            Image logging, default=False
            :type: bool

        Yield
        -------
        :yield: cls

        .. warning:: The dict style should be used, otherwise
                     this method will not work
        """
        for image in iterator:
            if isinstance(image["img"], str):
                img_path = cls.__input_path + image[0]
                image["img"] = cv2.imread(img_path, cv2.IMREAD_COLOR)
            yield cls(image["img"], image["name"], save_mode)

    @save
    def select_rgb_green(self):
        """ RGB mask.

        Method used to segregate the green spectrum

        Return
        -------
        :return: mask
        :type: numpy array

        """
        lower = np.uint8([65, 190, 0])
        upper = np.uint8([178, 255, 175])
        mask = cv2.inRange(self.base_image, lower, upper)
        return mask

    @save
    def morphological_transformation(self, img, kernel_shp, iterations, func):
        """ Morphological transformation wrapper.

        wraps all openCV morphological transformations

        Parameters
        ----------
        :param img:
            opened openCV image, default=None
        :type: numpy array

        :param kernel_shp:
            Square matrix shape
        :type: int or tuple

        :param iterations:
            number of iterations
        :type: int

        :param func:
            openCV morphological operator function
        :type: method or function

        Return
        -------
        :return transformed_img
        :type: numpy array

        """
        if not isinstance(kernel_shp, tuple) and not isinstance(kernel_shp, int):
            raise TypeError("shape must be int or tuple")

        if kernel_shp is int:
            kernel_shp = (kernel_shp, kernel_shp)

        kernel = np.ones(kernel_shp, np.uint8)

        transformed_img = func(img, kernel=kernel, iterations=iterations)
        return transformed_img

    @save
    def to_gray_scale(self, img=None):
        """Convert image to gray scale.

        Parameters
        ----------
        :param img:
            opened openCV image, default=None
        :type: numpy array

        Return
        -------
        :return transformed_img
        :type: numpy array
        """
        if img is None:
            img = self.base_image
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    @save
    def gaussian_smooth(self, img=None, kernel_shape=15):
        """Apply Gaussian smooth

        Apply the gaussian smooth to an image

        Parameters
        ----------
        :param img:
            opened openCV image, default=None
        :type: numpy array

        :param kernel_shp:
            Square matrix shape
        :type: int or tuple

        Return
        -------
        :return blured image
        :type: numpy array
        """
        if img is None:
            img = self.base_image
        return cv2.GaussianBlur(img, (kernel_shape, kernel_shape), 0)

    def __call__(self):
        """Call in correct order

        Call all the methods in the correct order and
        return smoothed image

        Return
        -------
        :return blured image
        :type: numpy array
        """
        green_mask = self.select_rgb_green()
        erosion = self.morphological_transformation(green_mask, 1, 1, cv2.erode)
        dilation = self.morphological_transformation(erosion, 3, 1, cv2.dilate)
        return self.gaussian_smooth(dilation)


class SugarCaneLineFinder(SugarCaneProcessingBase):
    """Sugar Cane processor

    Finds the crop rows into an image
    """
    def __init__(self, gauss_filtered_image, name, save_mode=False):
        self.__plants = 0
        SugarCaneProcessingBase.__init__(self, gauss_filtered_image, name, save_mode)

    @save
    def canny(self, low_thress=0, high_thress=10):
        """ Edge detector

        Apply the Canny algorithm for edge detection

        Parameters
        ----------
        :param low_thress:
        :type: int

        :param high_thress:
        :type: int

        Return
        -------
        :return: bitmap with draw edges
        :rtype: numpy array
        """
        return cv2.Canny(self.base_image, low_thress, high_thress)

    @save
    def get_lines(self, output_image):
        """Find all the contours

        Find all the contours by applying the Suzuki's topological structural 
        analysis of digitized binary images by border following algorithm

        Parameters
        ----------
        :param output_image:
            image to receive the contours
        :type: numpy array
        """
        __input_path = "{}/{}/".format(os.path.dirname(os.path.abspath(__file__)), "base_images")
        if isinstance(output_image, str):
            img_path = __input_path + output_image
            output_image = cv2.imread(img_path, cv2.IMREAD_COLOR)

        img = self.canny()
        _, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.__plants = len(contours)
        cv2.drawContours(output_image, contours, - 1, (0,255,0), 3)
        return output_image

    def __call__(self, output_image):
        """Call methods in order
        
        Call all the methods in the correct order
        
        Parameters
        ----------
        :param output_image:
            image to receive the contours
 
        """
        return self.get_lines(output_image)

    def __str__(self):
        """Print table
        
        Print a table with an report
        
        :return: table
        :rtype: str
        """
        img = self.canny()
        _, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        body = [["Image Resolution", self.base_image.shape[0:2], "numpy array"],
                ["Chanels", 1, "int"],
                ["Image size", self.base_image.size/8192, "kb"],
                ["Sugarcane clumps", len(contours), "int"]]
        headers = ["Feature", "Report", "Type"]
        table = tabulate.tabulate(body,
                                  headers,
                                  tablefmt="fancy_grid")
        return table
