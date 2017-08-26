
if __name__ == '__main__':
    from sugarcane_line_finder import *
    import re

    b_path = base_path()
    images = os.listdir("{}/{}".format(b_path, "base_images"))
    allowed_extensions = ("jpg", "JPG", "png", "img", "gif", "bmp")
    pattern = re.compile("(cana)")
    image_path = ["{}/base_images/{}".format(b_path, i) for i in images if
                  i.split(".")[1] in allowed_extensions and pattern.search(i)]
    image_names = [i for i in images if i.split(".")[1] in allowed_extensions and pattern.search(i)]

    sugarcane_images = [{"img": cv2.imread(i[0]), "name": i[1]} for i in zip(image_path, image_names)]

    pre_processors = SugarCanePreProcessing.multiple_processor(sugarcane_images, True)
    for i, j in zip(pre_processors, sugarcane_images):
        pre_processed = i()
        image = j["img"].copy()
        SugarCaneLineFinder(pre_processed, j["name"], True)(image)