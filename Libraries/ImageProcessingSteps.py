import cv2
import numpy as np

from Libraries.Core import ImageInfo


class ImageProcessingStep:
    """Шаг обработки изображений"""

    def process(self,info):
        """Выполнить обработку"""
        pass

class MedianBlurProcessingStep(ImageProcessingStep):
    """Шаг, отвечающий за предобработку типа Медианная фильтрация"""

    def __init__(self,ksize):
        """Конструктор
        ksize - размер ядра фильтра"""

        self.ksize=ksize

    def process(self,info):
        """Выполнить обработку"""

        median_image = cv2.medianBlur(info.image, self.ksize)
        return ImageInfo(median_image)

class ThresholdProcessingStep(ImageProcessingStep):
    """Шаг, отвечающий за бинаризацию"""

    def process(self,info):
        """Выполнить обработку"""

        gray = cv2.cvtColor(info.image, cv2.COLOR_BGR2GRAY)
        image = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]
        return ImageInfo(image)

class ContourProcessingStep(ImageProcessingStep):
    """Шаг, отвечающий за выделение контуров"""

    def process(self,info):
        """Выполнить обработку"""

        contours, hierarchy  = cv2.findContours(info.image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        height, width = info.image.shape[:2]
        contours_image = np.zeros((height, width, 3), dtype=np.uint8)

        # отображаем контуры
        cv2.drawContours(contours_image, contours, -1, (255, 0, 0), 1, cv2.LINE_AA, hierarchy, 1)

        #Заполним данные
        new_info=ImageInfo(contours_image)
        new_info.contours=contours
        new_info.hierarchy=hierarchy

        return new_info

class ContourApproximationProcessingStep(ImageProcessingStep):
    """Шаг, отвечающий за апроксимацию контуров"""

    def __init__(self,eps = 0.005, filter=None):
        """Конструктор
        eps - размер элемента контура от размера общей дуги"""

        self.eps=eps
        self.filter=filter

    def process(self, info):
        """Выполнить обработку"""

        approx_countours=[]
        img_contours = np.uint8(np.zeros((info.image.shape[0], info.image.shape[1])))
        for countour in info.contours:
            arclen = cv2.arcLength(countour, True)
            epsilon = arclen * self.eps
            approx = cv2.approxPolyDP(countour, epsilon, True)
            append=False
            if not(self.filter is None):
                if self.filter(approx):
                    append=True
            else:
                append=True
            if append:
                approx_countours.append(approx)

        cv2.drawContours(img_contours, approx_countours, -1, (255, 255, 255), 1)

        #Заполним данные
        new_info=ImageInfo(img_contours)
        new_info.contours=approx_countours

        return new_info