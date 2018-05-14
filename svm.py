import cv2
import numpy as np
import os.path
import evaluate


class SVM:
    isTrained = False
    cross_val_folds = 10

    def __init__(self, save_file_name):
        self.save_file_name = save_file_name

        if os.path.isfile(self.save_file_name):
            self.svm = cv2.ml.SVM_load(self.save_file_name)

            self.isTrained = True
            print("SVM loaded from file '" + self.save_file_name + "'")
        else:
            self.svm = cv2.ml.SVM_create()

            # Set SVM parameters
            self.svm.setType(cv2.ml.SVM_EPS_SVR)
            self.svm.setKernel(cv2.ml.SVM_LINEAR)
            self.svm.setC(0.1)
            self.svm.setP(0.4)
            self.svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 1000000, 1.e-20))

    # Train the SVM given an array with data and an array with correct labels (-1 or 1)
    # output: void, trained svm (self.svm)
    def train(self, data, labels, cross_val=False):

        if not cross_val:
            print("Training...")
            self.svm.train(np.array(data, np.float32), cv2.ml.ROW_SAMPLE, np.array([labels], np.int32))
            self.isTrained = True
            self.svm.save(self.save_file_name)

        else:
            # Divide data over folds
            fold_index = 0
            data_folds = []
            labels_folds = []
            for i, d in enumerate(data):
                if len(labels_folds) != self.cross_val_folds:
                    data_folds.append([])
                    labels_folds.append([])

                data_folds[fold_index].append(d)
                labels_folds[fold_index].append(labels[i])

                fold_index += 1
                if fold_index == self.cross_val_folds:
                    fold_index = 0

            for x in range(len(data_folds)):
                print(len(data_folds[x]))

            # Different parameters settings for validation
            C_range = [0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]
            p_range = [0.01, 0.2, 0.4, 0.6, 0.8]

            best_average = 0
            best_setting = ()

            # Perform cross-validation
            stop = False
            for C in C_range:
                for p in p_range:
                    val_score = 0
                    self.svm.setC(C)
                    self.svm.setP(p)

                    for i in range(self.cross_val_folds):
                        val_data_set = data_folds[i]
                        val_label_set = labels_folds[i]

                        # Merge other folds and use it as train data
                        train_data_set = []
                        train_label_set = []
                        for j in range(self.cross_val_folds):
                            if j == i:
                                continue
                            train_data_set += data_folds[j]
                            train_label_set += labels_folds[j]

                        # Train model
                        print(len(train_data_set))
                        print("Training... [" + str(i + 1) + "/" + str(self.cross_val_folds) + "]")#, end='\r')
                        self.svm.train(np.array(train_data_set, np.float32), cv2.ml.ROW_SAMPLE, np.array([train_label_set], np.int32))
                        self.isTrained = True

                        # Validate model
                        guesses = self.classify(val_data_set)
                        scr = evaluate.calc_fscore(guesses, val_label_set)
                        val_score += scr
                        print(scr)

                    # Calculate average validation score and update best parameter settings
                    average_val_score = val_score / self.cross_val_folds
                    print("average",average_val_score)
                    if average_val_score > best_average:
                        best_setting = (C, p)
                        best_average = average_val_score
                        print("bestC", self.svm.getC())
                        print("bestP", self.svm.getP())

                        if average_val_score == 1:
                            stop = True
                    if stop:
                        break
                if stop:
                    break

            print("")

            # Train with all data
            self.svm.setC(best_setting[0])
            self.svm.setP(best_setting[1])
            self.svm.train(np.array(data, np.float32), cv2.ml.ROW_SAMPLE, np.array([labels], np.int32))
            self.svm.save(self.save_file_name)

            print(self.svm.getC())
            print(self.svm.getP())

    # Classify an array of data
    # output: array of distances to the hyperplane, [-X.X] for negatives [X.X] for positives
    def classify(self, data):
        data = np.array(data, np.float32)
        return self.svm.predict(data, True)[1]