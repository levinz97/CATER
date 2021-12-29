import json
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm, metrics


class Data_hsv():
    def __init__(self):
        pass

    # The statistical color and material information is included in the hsv.json file
    def get_label_dict(self):
        input_path = './hsv'
        with open(input_path, 'r', encoding='UTF-8') as input_file:
            dictionary = json.load(input_file)
            input_file.close()
        number = len(dictionary)
        #print(a)

        material_list = ['r','m']
        color_list = ['red', 'blue', 'cyan', 'green', 'gold', 'purple', 'yellow', 'gray', 'brown']
        label_dict = {}
        label_list = []

        for i in color_list:
            for j in material_list:
                name = i + '_' + j
                label_dict[name] = {}
                label_dict[name]['hsv'] = []
                label_dict[name]['rgb'] = []
                label_list.append(name)

        for i in range(number):
            for j in label_list:
                label_dict[j]['hsv'].extend(dictionary[i][j]['hsv'])
                label_dict[j]['rgb'].extend(dictionary[i][j]['rgb'])

        #print(label_dict)
        return label_dict


class Svm():
    def __init__(self, label_dict):
        self.label_dict = label_dict

    # The label selected for training can be color*material c*m or pure color c or pure material m
    def class_choice(self, class_label):
        if class_label == 'c*m':
            return self.color_material
        elif class_label == 'c':
            return self.color
        elif class_label == 'm':
            return self.material
        else:
            print('wrong input')

    def color_material(self):
        feature = []
        label = []
        i = 1
        for key, value in self.label_dict.items():
            for hsv in value['hsv']:
                feature.append(hsv)
                label.append(i)
            i += 1
        return feature, label

    def color(self):
        feature = []
        label = []
        i = 1
        k = 0
        for key, value in self.label_dict.items():
            for hsv in value['hsv']:
                feature.append(hsv)
                label.append(i)
            k += 1
            if k % 2 == 0:
                i += 1
        return feature, label

    def material(self):
        feature = []
        label = []
        for key, value in self.label_dict.items():
            if 'm' in key:
                i = 1
            else:
                i = 0
            for hsv in value['hsv']:
                feature.append(hsv)
                label.append(i)
        return feature, label

    def train(self, feature, label):
        random_state = np.random.RandomState(0)
        # test_size is the proportion of the training set and can be modified
        X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.2, random_state=0)
        # Kernel function can choose 'rbf' Gaussian kernel or 'poly' polynomial kernel
        # Result of 'linear' kernel is very poor
        model = OneVsRestClassifier(svm.SVC(kernel='poly', probability=True, random_state = random_state))
        print("[INFO] Successfully initialize a new model !")
        print("[INFO] Training the model…… ")
        clt = model.fit(X_train, y_train)
        print("[INFO] Model training completed !")
        y_test_pred = clt.predict(X_test)
        ov_acc = metrics.accuracy_score(y_test_pred, y_test)
        print("overall accuracy: %f" % (ov_acc))
        print("===========================================")
        acc_for_each_class = metrics.precision_score(y_test, y_test_pred, average=None)
        print("acc_for_each_class:\n", acc_for_each_class)
        print("===========================================")
        avg_acc = np.mean(acc_for_each_class)
        print("average accuracy:%f" % (avg_acc))


def main():
    warnings.filterwarnings('ignore')
    dataset = Data_hsv()
    label_dict = dataset.get_label_dict()
    class_label = input('input your class label: c*m or c or m   ')
    svm1 = Svm(label_dict)
    label = svm1.class_choice(class_label)
    feature, label = label()
    print(feature)
    print(label)
    feature = np.array(feature)
    label = np.array(label)
    svm1.train(feature, label)

if __name__ == "__main__":
    main()


