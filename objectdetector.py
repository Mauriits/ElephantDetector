import cv2
import math
import hog
import nms
import numpy as np


class ObjectDetector:

    def __init__(self, scales, scaling, stride, detection_threshold, overlap_threshold):
        self.hog = hog.HOGDescriptor()

        # Save scores for quick detection when the image is the same
        self.window_scores = []
        self.current_image = np.zeros((0, 0, 3), np.uint8)

        # Set detection parameters
        self.w_size = self.hog.win_size
        self.w_scales = scales
        self.w_scaling = scaling
        self.w_stride = stride
        self.detection_threshold = detection_threshold
        self.overlap_threshold = overlap_threshold

    # Finds objects given an image and a two trained SVMs
    # output: tuple(array of bounding boxes, total amount of windows)
    def find_object(self, image, svm, mirrored_svm):

        if self.image_is_equal(image, self.current_image):
            return self.find_object_quick()

        self.current_image = image.copy()
        self.window_scores = []

        pyramid = self.create_pyramid(image)

        # Calculate amount of windows
        hor_amounts = []
        ver_amounts = []

        for i, img in enumerate(pyramid):
            h, w = img.shape[:2]

            hor_amounts.append(math.floor((w - self.w_size[0]) / self.w_stride[0]) + 1)
            ver_amounts.append(math.floor((h - self.w_size[1]) / self.w_stride[1]) + 1)

        totalIter = 0
        counter = 0
        for i in range(len(pyramid)):
            totalIter += hor_amounts[i] * ver_amounts[i]

        # Search object in each image of the pyramid
        found_objects = []
        for index, img in enumerate(pyramid):
            hor = hor_amounts[index]
            ver = ver_amounts[index]

            # Perform sliding window
            for i in range(ver):
                for j in range(hor):

                    if counter % 120 == 0:
                        print("Searching for object...", str(math.ceil(counter / totalIter * 100)) + "%", end='\r')

                    # Calculate current position of sliding window
                    x = j * self.w_stride[0]
                    y = i * self.w_stride[1]

                    window_hog = self.hog.calc_hog(img, (x, y))

                    score_normal = svm.classify([window_hog])[0][0]
                    score_mirrored = mirrored_svm.classify([window_hog])[0][0]
                    score = max(score_normal, score_mirrored)

                    self.window_scores.append((x, y, index, score))

                    if score >= self.detection_threshold:
                        scale_corr = math.pow(self.w_scaling, index)

                        x1 = int(x / scale_corr)
                        y1 = int(y / scale_corr)

                        x2 = x1 + int(self.w_size[0] / scale_corr)
                        y2 = y1 + int(self.w_size[1] / scale_corr)

                        found_objects.append((x1, y1, x2, y2, score))

                    counter += 1

        print("")

        filtered_bboxes = nms.non_maximum_suppression(found_objects, self.overlap_threshold)

        return filtered_bboxes, totalIter

    # Shows objects with a rectangle on the image, given their locations (x1, y1, x2, y2)
    # output: void, rectangle is drawn on the image
    def show_objects(self, image, locations, color=(0, 255, 255)):
        for pos in locations:
            cv2.rectangle(image, (pos[0], pos[1]), (pos[2], pos[3]), color, 2)
            cv2.rectangle(image, (pos[0] + 4, pos[1] - 20), (pos[0] + 80, pos[1]), color, -1)
            cv2.putText(image, "elephant", (pos[0] + 6, pos[1] - 6), 1, 1, (0, 0, 0), 0, cv2.LINE_AA)

    # Creates an image pyramid given an image
    # output: array of down scaled and smoothed images
    def create_pyramid(self, image):
        pyramid = [image]

        for i in range(self.w_scales - 1):
            image = cv2.resize(image, (0,0), fx=self.w_scaling, fy=self.w_scaling)
            image = cv2.GaussianBlur(image, (5, 5), 0)
            #image = cv2.pyrDown(image)

            if image.shape[0] < self.w_size[1] or image.shape[1] < self.w_size[0]:
                break
            else:
                pyramid.append(image)

        return pyramid

    def image_is_equal(self, image1, image2):
        return image1.shape == image2.shape and not (np.bitwise_xor(image1, image2).any())

    # Uses stored scores in window_scores to detect objects without having to classify
    # output: tuple(array of bounding boxes, total amount of windows)
    def find_object_quick(self):
        found_objects = []

        totalIter = 0
        for window in self.window_scores:
            totalIter += 1

            x = window[0]
            y = window[1]
            index = window[2]
            score = window[3]

            if score >= self.detection_threshold:
                scale_corr = math.pow(self.w_scaling, index)

                x1 = int(x / scale_corr)
                y1 = int(y / scale_corr)

                x2 = x1 + int(self.w_size[0] / scale_corr)
                y2 = y1 + int(self.w_size[1] / scale_corr)

                found_objects.append((x1, y1, x2, y2, score))

        filtered_bboxes = nms.non_maximum_suppression(found_objects, self.overlap_threshold)

        return filtered_bboxes, totalIter