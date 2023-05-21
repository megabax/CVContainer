import cv2
from keras import models

my_photo = cv2.imread('imgs/Digit0.png',cv2.IMREAD_GRAYSCALE) #загрузим изображение

#приведем изображение к формату для нейросети
normal_photo=my_photo/255.0
input=normal_photo.reshape(1,28,28)

#скормим изображение нейросетке и получим результат
model = models.load_model('mnist_model.bin')
result=model.predict(input)

print(result)
