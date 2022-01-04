import json
import numpy as np
import warnings
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
        input_path = './hsv'
        input_path = './raw_data/hsv.json'
        with open(input_path, 'r', encoding='UTF-8') as input_file:
            dictionary = json.load(input_file)
            input_file.close()
        number = len(dictionary)
        # print(a)

        material_list = ['r', 'm']
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

        # print(label_dict)
        return label_dict


class SimClassifier():
    def __init__(self, class_label="c*m"):
        self.class_label = class_label
        dataset = Data_hsv()
        self.label_dict = dataset.get_label_dict()
        self.models = []
        self.trained_models = []
        self.label_convert_dict = {}
        # model construct here: self.models
        self.label_convert_dict = self.dict_choice()

    def get_label_dict(self):
        return self.label_convert_dict

    # The label selected for training can be color*material c*m or pure color c or pure material m
    def class_choice(self):
        if self.class_label == 'c*m':
            return self.color_material
        elif self.class_label == 'c':
            return self.color
        elif self.class_label == 'm':
            return self.material
        else:
            # print('wrong input')
            return self.color_material

    def dict_choice(self):  # Convert the output to the corresponding attribute
        material_list = ['rubber', 'metal']
        color_list = ['red', 'blue', 'cyan', 'green', 'gold', 'purple', 'yellow', 'gray', 'brown']
        material_dict = {0: 'rubber', 1: 'metal'}
        color_dict = {}
        color_material_dict = {}
        i = 1
        for color in color_list:
            color_dict[i] = color
            i += 1
        j = 1
        for color in color_list:
            for material in material_list:
                color_material_dict[j] = [color, material]
                j += 1
        if self.class_label == 'c*m':
            return color_material_dict
        elif self.class_label == 'c':
            return color_dict
        elif self.class_label == 'm':
            return material_dict
        else:
            return color_material_dict

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

    def data_initial(self, feature, label):
        # test_size is the proportion of the training set and can be modified
        feature = np.array(feature)
        label = np.array(label)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(feature, label, test_size=0.2,
                                                                                random_state=0)
        # Kernel function can choose 'rbf' Gaussian kernel or 'poly' polynomial kernel
        # Result of 'linear' kernel is very poor
        return self.X_train, self.X_test, self.y_train, self.y_test

    def model_initial(self):
        # models = []
        random_state = np.random.RandomState(np.random.randint(0, 1000))
        model = OneVsRestClassifier(svm.SVC(kernel='poly', probability=True, random_state=random_state))
        self.models.append(model)
        model = RandomForestClassifier(n_estimators=80, random_state=random_state)
        # self.models.append(model)#85
        # model = make_pipeline(StandardScaler(),model)
        self.models.append(model)
        model = RandomForestClassifier(n_estimators=80, random_state=random_state)
        self.models.append(model)#85
        model = RandomForestClassifier(n_estimators=120, random_state=random_state)
        # self.models.append(model)#85
        model1 = RandomForestClassifier(n_estimators=350, random_state=random_state)
        self.models.append(model1)  # 86
        model2 = MLPClassifier(solver='lbfgs', alpha=1e-6, hidden_layer_sizes=(5, 18, 10, 5), random_state=random_state)
        # self.models.append(model2)
        model3 = MLPClassifier(solver='sgd', activation='relu', learning_rate_init=0.0005, learning_rate='adaptive',
                               alpha=1e-6, hidden_layer_sizes=(150), random_state=random_state)
        # self.models.append(model3) # for m
                               alpha=1e-6, hidden_layer_sizes=(150),max_iter=1000, random_state=random_state)
        self.models.append(model3) # for m
        model4 = MLPClassifier(solver='adam', activation='relu', learning_rate_init=1e-3, alpha=1e-3,
                               hidden_layer_sizes=(20, 40, 60, 80), max_iter=1000, random_state=random_state)
        self.models.append(model4)
        model5 = MLPClassifier(solver='sgd', activation='relu', learning_rate='adaptive', shuffle=False, alpha=1e-6,
                               hidden_layer_sizes=(150), batch_size=32, random_state=random_state)
        # self.models.append(model5)
        model6 = OneVsRestClassifier(svm.SVC(kernel='rbf', probability=True, random_state=random_state))
        # self.models.append(model6)
        assert len(self.models) > 0, "no model added!"
        print("[INFO] Successfully initialize a new model !")
        #return self.models
        # print("[INFO] Training the model…… ")

    def train(self):
        feature, label = self.class_choice()()
        self.label_convert_dict = self.dict_choice()
        self.data_initial(feature, label)
        self.model_initial()
        for model in self.models:
            pipe = model
            pipe.fit(self.X_train, self.y_train)
            self.trained_models.append(pipe)
        print("[INFO] Model training completed !")

    def ensembleLearning(self, X_test):
        # pipe = make_pipeline(StandardScaler(),model)
        y_test_pred = 0
        cnt = 0
        for trained_model in self.trained_models:

    def train(self):
        feature, label = self.class_choice()()

        self.data_initial(feature, label)
        self.model_initial()
        for model in self.models:
            model.fit(self.X_train, self.y_train)
        print("[INFO] Model training completed !")

    def ensembleLearning(self, X_test):
        y_test_pred = 0
        cnt = 0
        for trained_model in self.models:
            cnt += 1
            y_p = trained_model.predict(X_test)  # 预测结果
            if isinstance(y_test_pred, int):
                y_test_pred = y_p  # 如果是int 预测结果传给他
            else:
                if np.max(y_p) != np.min(y_p):  # 最大值不等于最小值
                    y_test_pred = np.vstack((y_test_pred, y_p))  # 把它堆起来
                    # if m == model5  or m == model4:
                    # if trained_model == self.models.model5  or trained_model == self.models.model4:
                    #     y_test_pred = np.vstack((y_test_pred,y_p))
                else:
                    print(
                        f"[{'warning'.upper()}] model Failed:\n{trained_model} collapses\n with output{y_p}")
        y_test_pred = y_test_pred.astype(np.uint8)
        all_pred = y_test_pred
        print(y_test_pred)
        # find the mode of output
        if len(y_test_pred.shape) > 1:
            y_test_pred, _ = stats.mode(y_test_pred, axis=0)
            y_test_pred = y_test_pred[0]
        print(y_test_pred)
        if X_test.all() == self.X_test.all():
            print(f"GT:\n {self.y_test}")
        return all_pred, y_test_pred  # pred for ensembleLearning models

    def predict(self, X_test):  # X_test: input of hsv:list
        all_pred, y_test_pred = self.ensembleLearning(X_test)
        return all_pred, y_test_pred, self.label_convert_dict
        # pred for each model

    def eval(self, y_test_pred, y_test):
        # evaluate model accuracy
        ov_acc = metrics.accuracy_score(y_test_pred, y_test)
        # ov_acc = pipe.score(X_test, y_test)
        print(f"final decision:\n {y_test_pred}")
        return all_pred.transpose(), y_test_pred  # pred for ensembleLearning models

    def predict(self, X_test):  # X_test: input of hsv:list
        all_pred, y_test_pred = self.ensembleLearning(X_test)
        return all_pred, y_test_pred
        # pred for each model

    def eval(self, y_test_pred):
        # TODO evaluate model accuracy
        print(f"GT:\n {self.y_test}")
        ov_acc = metrics.accuracy_score(y_test_pred, self.y_test)
        print("overall accuracy: %f" % (ov_acc))
        print("===========================================")
        acc_for_each_class = metrics.precision_score(
            self.y_test, y_test_pred, average=None, zero_division=1)
        print("acc_for_each_class:\n", acc_for_each_class)
        print("===========================================")
        avg_acc = np.mean(acc_for_each_class)
        print("average accuracy:%f" % (avg_acc))
        print("average class accuracy:%f" % (avg_acc))
        return ov_acc

    def convert_m(self, data):
        material = np.copy(data)
        for i in range(len(data)):
            if data[i] % 2 == 0:
                material[i] = 1
            else:
                material[i] = 0
        return material

    def convert_c(self, data):
        color = np.copy(data)
        convert_dict = {}
        k = 1
        for i in range(1, 19):
            convert_dict[i] = k
            if i % 2 == 0:
                k += 1
        print('检测1')
        print(convert_dict)
        for i in range(len(data)):
            color[i] = int(convert_dict[data[i]])
        return color

def main():
    warnings.filterwarnings('ignore')
    # class_label = input('input your class label: c*m or c or m   ')
    # simc = SimClassifier()
    # label = simc.class_choice(class_label)
    # feature, label = label()
    # attribut_dict = simc.dict_choice(class_label)  # can be used later for attributes search
    # simc.data_initial(feature, label)
    # simc.model_initial()
    # trained_model_list = simc.model_train()
    # _, y_test_pred = simc.ensembleLearning(trained_model_list, simc.X_test)
    # simc.eval(y_test_pred)
    # # X_test input your data hsv value
    # # all_pred, y_test_pred = simc.ensembleLearning(trained_model_list, X_test)
    # # print(all_pred)
    # # print(y_test_pred)
    clf = SimClassifier('c*m')  # create classifier
    clf.train()
    all_pred, y_test_pred, _ = clf.predict(clf.X_test)
    print("---------------c*m------------------")
    clf.eval(y_test_pred, clf.y_test)
    y_test_pred_m = clf.convert_m(y_test_pred)
    y_test_m = clf.convert_m(clf.y_test)
    print("---------------m------------------")
    clf.eval(y_test_pred_m, y_test_m)
    y_test_pred_c = clf.convert_c(y_test_pred)
    y_test_c = clf.convert_c(clf.y_test)
    print("---------------c------------------")
    clf.eval(y_test_pred_c, y_test_c)
    print('no improvement')
    # warnings.filterwarnings('ignore')
    acc_list = []
    modal = input("choose modality, c, m or c*m  ")
    for i in range(1):
        clf = SimClassifier(modal)  # create classifier
        clf.train()
        hsv_dict = clf.get_label_dict()
        all_pred_val, predict_val = clf.predict([[111.52783964, 100.06458797,  26.69042316], [166.86111111, 134.04678363, 122.21418129], [ 71.74397032, 169.5868893,   58.91032777],
        [17.85132297, 148.36539269,  80.93910122]])
        for i in all_pred_val[3]:
            print(f"prediction is {hsv_dict[i]}")
        # all_pred, y_test_pred, _ = clf.predict(clf.X_test)
        # acc_list.append(clf.eval(y_test_pred))
    # print(f"overall average acc: {np.mean(acc_list)}")
    # print(f"overall average std: {np.std(acc_list)}")


if __name__ == "__main__":
    main()
