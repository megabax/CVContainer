class Engine:
    """"Движок"""

    def __init__(self):
        """Конструктор"""

        self.steps=[]

    def process(self, image):
        """Выполнить обработку.
        image - изображение"""

        current_info=ImageInfo(image)
        for step in self.steps:
            current_info=step.process(current_info)
        return current_info

class ImageInfo:
    """Содержимое картинки, включая результаты обработки"""

    def __init__(self,image):
        """Конструктор"""

        self.image=image