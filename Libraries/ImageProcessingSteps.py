import cv2

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
        info.filtered_image=median_image
        return info