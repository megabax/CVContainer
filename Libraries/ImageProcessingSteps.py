import cv2

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