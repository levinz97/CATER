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
        input_path = './hsv'
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
    def __init__(self):
        dataset = Data_hsv()
        self.label_dict = dataset.get_label_dict()
        self.models = []
        # model construct here: self.models

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

    def dict_choice(self, class_label):  # Convert the output to the corresponding attribute
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
        if class_label == 'c*m':
            return color_material_dict
        elif class_label == 'c':
            return color_dict
        elif class_label == 'm':
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
        model = RandomForestClassifier(n_estimators=120, random_state=random_state)
        # self.models.append(model)#85
        model1 = RandomForestClassifier(n_estimators=350, random_state=random_state)
        self.models.append(model1)  # 86
        model2 = MLPClassifier(solver='lbfgs', alpha=1e-6, hidden_layer_sizes=(5, 18, 10, 5), random_state=random_state)
        # self.models.append(model2)
        model3 = MLPClassifier(solver='sgd', activation='relu', learning_rate_init=0.0005, learning_rate='adaptive',
                               alpha=1e-6, hidden_layer_sizes=(150), random_state=random_state)
        # self.models.append(model3) # for m
        model4 = MLPClassifier(solver='adam', activation='relu', learning_rate_init=1e-3, alpha=1e-3,
                               hidden_layer_sizes=(20, 40, 60, 80), random_state=random_state)
        self.models.append(model4)
        model5 = MLPClassifier(solver='sgd', activation='relu', learning_rate='adaptive', shuffle=False, alpha=1e-6,
                               hidden_layer_sizes=(150), batch_size=32, random_state=random_state)
        # self.models.append(model5)
        model6 = OneVsRestClassifier(svm.SVC(kernel='rbf', probability=True, random_state=random_state))
        # self.models.append(model6)
        assert len(self.models) > 0, "no model added!"
        print("[INFO] Successfully initialize a new model !")
        return self.models
        # print("[INFO] Training the model…… ")

    def model_train(self):
        trained_model_list = []
        for model in self.models:
            pipe = model
            pipe.fit(self.X_train, self.y_train)
            trained_model_list.append(pipe)
        print("[INFO] Model training completed !")
        return trained_model_list

    def predict(self, trained_model, X_test):
        # predict for each model
        pipe = trained_model
        y_test_pred = pipe.predict(X_test)
        return y_test_pred  # pred for each model

    def ensembleLearning(self, trained_model_list, X_test):
        # pipe = make_pipeline(StandardScaler(),model)
        y_test_pred = 0
        cnt = 0
        for trained_model in trained_model_list:
            cnt += 1
            y_p = self.predict(trained_model, X_test)  # 预测结果
            if isinstance(y_test_pred, int):
                y_test_pred = y_p  # 如果是int 预测结果传给他
            else:
                if np.max(y_p) != np.min(y_p):  # 最大值不等于最小值
                    y_test_pred = np.vstack((y_test_pred, y_p))  # 把它堆起来
                    # if m == model5  or m == model4:
                    #     y_test_pred = np.vstack((y_test_pred,y_p))
                else:
                    print(
                        f"[{'warning'.upper()}] model Failed:\n{m} collapses\n with output{y_p}")
        y_test_pred = y_test_pred.astype(np.uint8)
        all_pred = y_test_pred
        print(y_test_pred)
        # find the mode of output
        if len(y_test_pred.shape) > 1:
            y_test_pred, _ = stats.mode(y_test_pred, axis=0)
            y_test_pred = y_test_pred[0]
        print(y_test_pred)
        print(f"GT:\n {self.y_test}")
        return all_pred, y_test_pred  # pred for ensembleLearning models

    def eval(self, y_test_pred):
        # TODO evaluate model accuracy
        ov_acc = metrics.accuracy_score(y_test_pred, self.y_test)
        # ov_acc = pipe.score(X_test, y_test)
        print("overall accuracy: %f" % (ov_acc))
        print("===========================================")
        acc_for_each_class = metrics.precision_score(
            self.y_test, y_test_pred, average=None)
        print("acc_for_each_class:\n", acc_for_each_class)
        print("===========================================")
        avg_acc = np.mean(acc_for_each_class)
        print("average accuracy:%f" % (avg_acc))
        return ov_acc


def main():
    warnings.filterwarnings('ignore')
    class_label = input('input your class label: c*m or c or m   ')
    simc = SimClassifier()
    label = simc.class_choice(class_label)
    feature, label = label()
    attribut_dict = simc.dict_choice(class_label)  # can be used later for attributes search
    simc.data_initial(feature, label)
    simc.model_initial()
    trained_model_list = simc.model_train()
    _, y_test_pred = simc.ensembleLearning(trained_model_list, simc.X_test)
    simc.eval(y_test_pred)
    # X_test input your data hsv value
    # all_pred, y_test_pred = simc.ensembleLearning(trained_model_list, X_test)
    # print(all_pred)
    # print(y_test_pred)


if __name__ == "__main__":
    main()
