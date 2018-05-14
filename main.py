import cv2
import data
import hog
import svm as SupportVectorMachine
import objectdetector as od
import evaluate
import pickle
import math

train_data, train_data_mirrored, train_labels = data.get_train_data()

svm = SupportVectorMachine.SVM("trained_svm.data")
if not svm.isTrained:
    svm.train(train_data, train_labels)

mirrored_svm = SupportVectorMachine.SVM("trained_mirrored_svm.data")
if not mirrored_svm.isTrained:
    mirrored_svm.train(train_data_mirrored, train_labels)

hog = hog.HOGDescriptor()

objectDetector = od.ObjectDetector(scales=10, scaling=0.8, stride=(6, 5), detection_threshold=0.6, overlap_threshold=0.5)

while True:
    command = input("\nType a command\n")
    arguments = command.split()

    if arguments[0] == "classify":
        images = data.get_images(arguments[1])

        if len(images) != 0:
            hogs = []
            for i in images:
                hogs.append(hog.calc_hog(i))

            labels = svm.classify(hogs)
            print(str(labels))

    if arguments[0] == "find":

        images = data.get_images(arguments[1])

        for img in images:
            result = objectDetector.find_object(img, svm, mirrored_svm)
            objectDetector.show_objects(img, result[0])
            cv2.imshow("Found objects", img)
            print("Press key to continue")
            cv2.waitKey(0)

        cv2.destroyAllWindows()

    if arguments[0] == "test_thresholds":

        with open("images/test/positive/bounding_boxes", 'rb') as f:
            correct_bboxes = pickle.load(f)

        images_pos = data.get_images("images/test/positive")
        images_neg = data.get_images("images/test/negative")
        pos_neg_images = [images_pos, images_neg]

        thresholds = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]
        threshold_tp = []
        threshold_fn = []
        threshold_fp = []

        # Initialize arrays with a 0 for each threshold
        for i in range(len(thresholds)):
            threshold_tp.append(0)
            threshold_fn.append(0)
            threshold_fp.append(0)

        counter = 0

        for c in range(2):
            for j, img in enumerate(pos_neg_images[c]):
                for i, t in enumerate(thresholds):
                    objectDetector.detection_threshold = t
                    result = objectDetector.find_object(img, svm, mirrored_svm)

                    if c == 0:
                        measure = evaluate.evaluate_detection(result[0].copy(), result[1], correct_bboxes[j])
                    else:
                        measure = evaluate.evaluate_detection(result[0].copy(), result[1])

                    # Accumulate TPs, FPs, FNs for each threshold
                    threshold_tp[i] += measure[0]
                    threshold_fn[i] += measure[1]
                    threshold_fp[i] += measure[2]

                counter += 1
                print(str(counter) + "/100 \r")

        print("")

        print(threshold_tp)
        print(threshold_fn)
        print(threshold_fp)

        with open("pr_graph_data", 'wb') as file:
            pickle.dump((threshold_tp, threshold_fn, threshold_fp), file)

        # Show Precision-recall graph
        evaluate.show_pr_graph(threshold_tp, threshold_fn, threshold_fp)

        cv2.waitKey(0)

    if arguments[0] == "test_parameters":

        train_data, train_data_mirrored, train_labels = data.get_train_data(int(arguments[1]))

        mirrored_svm.svm.setC(float(arguments[2]))
        mirrored_svm.svm.setP(float(arguments[3]))
        svm.svm.setC(float(arguments[2]))
        svm.svm.setP(float(arguments[3]))

        mirrored_svm.train(train_data, train_labels)
        svm.train(train_data_mirrored, train_labels)

        true_pos = 0
        false_neg = 0
        false_pos = 0

        with open("images/test/positive/bounding_boxes", 'rb') as f:
            correct_bboxes = pickle.load(f)

        images_pos = data.get_images("images/test/positive")
        images_neg = data.get_images("images/test/negative")
        pos_neg_images = [images_pos, images_neg]

        counter = 0
        for c in range(2):
            for j, img in enumerate(pos_neg_images[c]):
                result = objectDetector.find_object(img, svm, mirrored_svm)

                measure = evaluate.evaluate_detection(result[0], result[1], correct_bboxes[j])

                true_pos += measure[0]
                false_neg += measure[1]
                false_pos += measure[2]

                counter += 1
                print(str(counter) + "/100 \r")


        fmeasure = evaluate.calc_fmeasure(true_pos, false_neg, false_pos)

        print("F-Measure:", fmeasure)

    if arguments[0] == "show_graph":

        with open("pr_graph_data", 'rb') as f:
            threshold_tp, threshold_fn, threshold_fp = pickle.load(f)
            evaluate.show_pr_graph(threshold_tp, threshold_fn, threshold_fp)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    else:
        print(arguments[0] + " is not a command")
