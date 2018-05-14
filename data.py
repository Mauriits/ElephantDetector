import cv2
import os
from os import listdir
import hog as descriptor


def get_train_data(max_img=20000):
    positive_img_path = "images/train/positive"
    negative_img_path = "images/train/negative"

    # Get the names of the images
    positive_img_names = [img for img in listdir(positive_img_path)]
    negative_img_names = [img for img in listdir(negative_img_path)]

    train_data = []
    train_data_mirrored = []
    train_labels = []

    max_img_amount = max_img

    hog = descriptor.HOGDescriptor()

    # For each positive image, compute its HOG and append a 1 to the labels list
    for img_name in positive_img_names:
        img = cv2.imread(positive_img_path + "/" + img_name)

        if (img is not None):
            flipped = cv2.flip(img, 1)
            train_data_mirrored.append(hog.calc_hog(flipped))

            train_data.append(hog.calc_hog(img))
            train_labels.append(1)

        # Stop adding images if desired amount of images is reached
        if len(train_labels) == int(max_img_amount / 2):
            break

    # For each negative image, compute its HOG and append a 0 to the labels list
    for img_name in negative_img_names:
        img = cv2.imread(negative_img_path + "/" + img_name)

        if (img is not None):
            img_hog = hog.calc_hog(img)
            train_data.append(img_hog)
            train_data_mirrored.append(img_hog)
            train_labels.append(-1)

        # Stop adding images if desired amount of images is reached
        if len(train_labels) == max_img_amount:
            break

    return train_data, train_data_mirrored, train_labels

def get_images(file_name):
    image = cv2.imread(file_name)
    images = []

    if image is None:
        # Assume it is a directory
        if os.path.exists(file_name):
            img_names = [img for img in listdir(file_name)]
            for img_name in img_names:
                img = cv2.imread(file_name + "/" + img_name)
                if img is not None:
                    images.append(img)
    else:
        images.append(image)

    if len(images) == 0:
        print("'" + file_name + "' is not an image and couldn't find any images in directory '" + file_name + "'")

    return images