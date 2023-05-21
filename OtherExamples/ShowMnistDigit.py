from keras.datasets import mnist
import cv2

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_labels)

cv2.imshow("Цифра", train_images[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
