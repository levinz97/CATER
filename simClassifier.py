import json
import numpy as np
import warnings
from scipy.sparse.construct import rand
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from scipy import stats


class Data_hsv():
    def __init__(self):
        pass

    # The statistical color and material information is included in the hsv.json file
    def get_label_dict(self):
        input_path = './hsv.json'
        with open(input_path, 'r', encoding='UTF-8') as input_file:
            dictionary = json.load(input_file)
            input_file.close()
        number = len(dictionary)
        # print(a)

        material_list = ['r', 'm']
        color_list = ['red', 'blue', 'cyan', 'green',
                      'gold', 'purple', 'yellow', 'gray', 'brown']
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

        # print(label_dict)
        return label_dict

# TODO refactor needed
class SimClassifier():
    def __init__(self, label_dict):
        self.label_dict = label_dict
        # TODO: model construct here: self.models

    # The label selected for training can be color*material c*m or pure color c or pure material m
    def class_choice(self, class_label):
        if class_label == 'c*m':
            return self.color_material
        elif class_label == 'c':
            return self.color
        elif class_label == 'm':
            return self.material
        else:
            # print('wrong input')
            return self.color_material

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
    def eval(self):
        # TODO evaluate model accuracy 
        pass
    def predict(self):
        # TODO predict 
        pass

    def train(self, feature, label):
        random_state = np.random.RandomState(np.random.randint(0, 1000))
        # random_state = np.random.RandomState(0)
        # test_size is the proportion of the training set and can be modified
        X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.2, random_state=0)
        # Kernel function can choose 'rbf' Gaussian kernel or 'poly' polynomial kernel
        # Result of 'linear' kernel is very poor

        models = []
        model = OneVsRestClassifier(svm.SVC(kernel='poly', probability=True, random_state=random_state))
        models.append(model)
        model  = RandomForestClassifier(n_estimators=80, random_state=random_state)
        # models.append(model)#85
        model  = RandomForestClassifier(n_estimators=120, random_state=random_state)
        # models.append(model)#85
        model1 = RandomForestClassifier(n_estimators=350, random_state=random_state)
        models.append(model1)#86
        model2 = MLPClassifier(solver='lbfgs', alpha=1e-6, hidden_layer_sizes=(5, 18, 10, 5), random_state=random_state)
        # models.append(model2)
        model3 = MLPClassifier(solver='sgd', activation='relu',learning_rate_init=0.0005, learning_rate='adaptive', alpha=1e-6, hidden_layer_sizes=(150), random_state=random_state)
        # models.append(model3) # for m 
        model4 = MLPClassifier(solver='adam', activation='relu', learning_rate_init=1e-3, alpha=1e-3,hidden_layer_sizes=(20,40,60,80), random_state=random_state)
        models.append(model4)
        model5 = MLPClassifier(solver='sgd', activation='relu',learning_rate='adaptive',shuffle=False, alpha=1e-6,hidden_layer_sizes=(150), batch_size=32, random_state=random_state)
        # models.append(model5)
        model6 = OneVsRestClassifier(svm.SVC(kernel='rbf', probability=True, random_state=random_state))
        # models.append(model6)
        assert len(models) > 0, "no model added!"
        print("[INFO] Successfully initialize a new model !")
        print("[INFO] Training the model…… ")

        def ensembleLearning(model):
            # pipe = make_pipeline(StandardScaler(),model)
            pipe = model
            pipe.fit(X_train, y_train)
            print("[INFO] Model training completed !")
            y_test_pred = pipe.predict(X_test)
            return y_test_pred
        y_test_pred = 0
        cnt = 0
        for m in models:
            cnt += 1
            y_p = ensembleLearning(m)
            if isinstance(y_test_pred, int):
                y_test_pred = y_p
            else:
                if np.max(y_p) != np.min(y_p):
                    y_test_pred = np.vstack((y_test_pred, y_p))
                    # if m == model5  or m == model4:
                    #     y_test_pred = np.vstack((y_test_pred,y_p))
                else:
                    print(
                        f"[{'warning'.upper()}] model Failed:\n{m} collapses\n with output{y_p}")

        y_test_pred = y_test_pred.astype(np.uint8)
        print(y_test_pred)
        # find the mode of output 
        if len(y_test_pred.shape) > 1:
            y_test_pred = y_test_pred.transpose()
            y_test_pred, _ = stats.mode(y_test_pred, axis=1)
        print(y_test_pred.transpose())
        print(f"GT:\n {y_test}")
        ov_acc = metrics.accuracy_score(y_test_pred, y_test)
        # ov_acc = pipe.score(X_test, y_test)
        print("overall accuracy: %f" % (ov_acc))
        print("===========================================")
        acc_for_each_class = metrics.precision_score(
            y_test, y_test_pred, average=None)
        print("acc_for_each_class:\n", acc_for_each_class)
        print("===========================================")
        avg_acc = np.mean(acc_for_each_class)
        print("average accuracy:%f" % (avg_acc))
        
        return ov_acc


def main():
    warnings.filterwarnings('ignore')
    dataset = Data_hsv()
    label_dict = dataset.get_label_dict()
    class_label = input('input your class label: c*m or c or m   ')
    svm1 = SimClassifier(label_dict)
    label = svm1.class_choice(class_label)
    feature, label = label()
    # print(feature)
    # print(label)
    feature = np.array(feature)
    label = np.array(label)
    acc_list = []
    for i in range(100):
        acc_list.append(svm1.train(feature, label))
    print(f"\noverall average accuracy is {np.mean(acc_list)}")
    print(f"standard deviation of accuracy is {np.std(acc_list)}")
    

if __name__ == "__main__":
    main()
