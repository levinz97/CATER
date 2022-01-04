import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from scipy import stats

class Data_size:
    def __init__(self):
        pass

    def feature_label(self):
        input_path = './data_final'
        with open(input_path, 'r', encoding='UTF-8') as input_file:
            dictionary = json.load(input_file)
            input_file.close()

        data_dict = dictionary[0]
        material_list = ['r', 'm']
        color_list = ['red', 'blue', 'cyan', 'green', 'gold', 'purple', 'yellow', 'gray', 'brown']
        shape_list = ['cub', 'con', 'spl', 'sph', 'cyl']
        size_list = ['s', 'm', 'l']

        name_list = []
        for i in color_list:
            for j in material_list:
                name = i + '_' + j
                name_list.append(name)

        area = []
        length = []
        center = []
        shape = []
        size = []
        for name in name_list:
            area.extend(data_dict[name]['area'])
            length.extend(data_dict[name]['length'])
            center.extend(data_dict[name]['center'])
            shape.extend(data_dict[name]['shape'])
            size.extend(data_dict[name]['size'])

        #change to number label
        shape_dict = {}
        size_dict = {}

        shape_cnt = 0
        for k in shape_list:
            shape_dict[k] = shape_cnt
            shape_cnt += 1

        size_cnt = 0
        for k in size_list:
            size_dict[k] = size_cnt
            size_cnt += 1
        print(size_dict)

        for i in range(len(shape)):
            shape[i] = shape_dict[shape[i]]

        for i in range(len(size)):
            size[i] = size_dict[size[i]]

        area_length_feature = []
        for i in range(len(area)):
            list = [area[i], length[i], area[i]/length[i]]
            area_length_feature.append(list)
        #print(area_length_feature)

        # feature = np.array(area_length_feature)
        # label = np.array(shape)

        area_center_feature = []
        for i in range(len(area)):
            list = []
            list.append(area[i])
            list.append(length[i])
            list.extend(center[i])
            area_center_feature.append(list)
        #print(area_center_feature)


        feature = np.array(area_center_feature)
        label = np.array(size)
        return feature, label

class SizeClassifier:
    def __init__(self):
        dataset_size = Data_size()
        self.feature, self.label = dataset_size.feature_label()
        self.model_list = []
        self.label_convert_size_dict = {0: 'small', 1: 'medium', 2: 'large'}

    def get_label_size_dict(self):
        return self.label_convert_size_dict

    def train(self):
        for i in range(10):
            random_state = np.random.RandomState(np.random.randint(0, 1000))
            x_train, x_test, y_train, y_test = train_test_split(self.feature,self.label, test_size=0.2, random_state=random_state)

            random_state = np.random.RandomState(np.random.randint(0, 1000))
            model = RandomForestClassifier(n_estimators=450, random_state=random_state)

            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            #print(y_pred)  # 预测值
            #print(y_test)  # 真实值
            acc = metrics.accuracy_score(y_pred, y_test)
            print("准确率：", acc)  # 计算准确率
            if acc > 0.9:
                self.model_list.append(model)

    def size_pred(self, x_test):
        y_test_pred = 0
        cnt = 0
        for model in self.model_list:
            cnt += 1
            y_p = model.predict(x_test)  # 预测结果
            if isinstance(y_test_pred, int):
                y_test_pred = y_p  # 如果是int 预测结果传给他
            else:
                if np.max(y_p) != np.min(y_p):  # 最大值不等于最小值
                    y_test_pred = np.vstack((y_test_pred, y_p))  # 把它堆起来
                    # if trained_model == self.models.model5  or trained_model == self.models.model4:
                    #     y_test_pred = np.vstack((y_test_pred,y_p))
                else:
                    print(
                        f"[{'warning'.upper()}] model Failed:\n{model} collapses\n with output{y_p}")
        y_test_pred = y_test_pred.astype(np.uint8)
        all_pred = y_test_pred
        #print(y_test_pred)
        # find the mode of output
        if len(y_test_pred.shape) > 1:
            y_test_pred, _ = stats.mode(y_test_pred, axis=0)
            y_test_pred = y_test_pred[0]
        print(f"final decision:\n {y_test_pred}")
        return all_pred.transpose(), y_test_pred  # pred for ensembleLearning models


if __name__ == "__main__":
    for i in range(1):
        clf = SizeClassifier()  # create classifier
        clf.train()
        all_pred, y_pred = clf.size_pred(clf.feature[3:8])
        print(all_pred)
        print(y_pred)
        print(clf.label[3:8])

