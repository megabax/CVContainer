import math
import cv2

def get_rect(countur_item):
    x1 = countur_item[0][0][0]
    y1 = countur_item[0][0][1]
    x2 = countur_item[2][0][0]
    y2 = countur_item[2][0][1]
    return (x1,y1), (x2,y2)

def dist(point1,point2):
    d1 = point1[0] - point2[0]
    d2 = point1[1] - point2[1]
    return math.sqrt(d1*d1+d2*d2)

def show_history(res,history):
    i = 1
    for info in history:
        cv2.imshow('image' + str(i), info.image)  # выводим изображение в окно
        i = i + 1
    cv2.imshow('res', res.image)