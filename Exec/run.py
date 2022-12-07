import cv2

from Libraries.Core import Engine
from Libraries.ImageProcessingSteps import MedianBlurProcessingStep

my_photo = cv2.imread('../Photos/MyPhoto1.jpg')
core=Engine()
core.steps.append(MedianBlurProcessingStep(5))
info=core.process(my_photo)

cv2.imshow('origin', info.image) # выводим исходное изображение в окно
cv2.imshow('res', info.filtered_image) # выводим итоговое изображение в окно

cv2.waitKey()
cv2.destroyAllWindows()