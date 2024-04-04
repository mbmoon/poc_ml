from digits_dataset import split_images, split_data
from cifar_dataset import load_images

import cv2


# Load the digits image
img, sub_imgs = split_images('Images/digits.png', 20)

# cv2.imshow("Resized image", sub_imgs[6][2])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.waitKey(1)

# Obtain training and testing datasets from the digits image
digits_train_imgs, digits_train_labels, digits_test_imgs, digits_test_labels = split_data(20, sub_imgs, 0.8)

# Obtain training and testing datasets from the CIFAR-10 dataset
cifar_train_imgs, cifar_train_labels, cifar_test_imgs, cifar_test_labels = load_images('Images/cifar-10-batches-py/')