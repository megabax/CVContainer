import cv2

from Libraries.Core import Engine
from Libraries.ImageProcessingSteps import MedianBlurProcessingStep, ThresholdProcessingStep, ContourProcessingStep, \
    ContourApproximationProcessingStep
from Libraries.Utils import dist, get_rect, show_history


def my_filter(approx):
    if len(approx)==4:
        if abs(approx[2,0,0]-approx[0,0,0])<10:
            return False
        if abs(approx[2,0,1]-approx[0,0,1])<10:
            return False
        if abs(dist(approx[0,0],approx[1,0])/dist(approx[2,0],approx[3,0])-1)>0.4:
            return False
        if abs(dist(approx[0,0],approx[3,0])/dist(approx[1,0],approx[2,0])-1)>0.4:
            return False
        return True
    return False


my_photo = cv2.imread('../Photos/6108249.jpg')
#my_photo = cv2.imread('../Photos/car.jpg')
core=Engine()
core.steps.append(MedianBlurProcessingStep(5))
core.steps.append(ThresholdProcessingStep())
core.steps.append(ContourProcessingStep())
#core.steps.append(ContourApproximationProcessingStep(0.02))
core.steps.append(ContourApproximationProcessingStep(0.02,my_filter))
res,history=core.process(my_photo)

show_history(res,history)

finish_result = history[0].image.copy()

for item in res.contours:
    p1, p2 = get_rect(item)
    cv2.rectangle(finish_result, p1, p2, (255, 0, 0), 3)
cv2.imshow('Finish', finish_result)


cv2.waitKey()
cv2.destroyAllWindows()