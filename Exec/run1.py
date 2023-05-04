import cv2

from Libraries.Core import Engine
from Libraries.ImageProcessingSteps import MedianBlurProcessingStep, ThresholdProcessingStep

my_photo = cv2.imread('../Photos/car.jpg')
core=Engine()
core.steps.append(MedianBlurProcessingStep(5))
core.steps.append(ThresholdProcessingStep())
res,history=core.process(my_photo)

i=1
for info in history:
    cv2.imshow('image'+str(i), info.image) # выводим изображение в окно
    i=i+1
cv2.imshow('res', res.image)

cv2.waitKey()
cv2.destroyAllWindows()