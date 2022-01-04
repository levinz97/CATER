from numpy.random.mtrand import f
from prepare_data import PrepareData
from simClassifier import SimClassifier
from utils import dispImg

import numpy as np
import os
import cv2

dirname = os.path.join('.','raw_data','first_frame', 'all_actions_first_frame')
class generateDataset:
    def __init__(self, dirname:str):
        self.dirname = dirname
        self.pd = PrepareData(need_visualization=False)
        # train a simple classifier for color, material prediction
        self.clf = SimClassifier(class_label="c*m")
        self.clf.train()
        self.label_convert_hsv_dict = self.clf.get_label_hsv_dict()
        self.label_convert_size_dict = self.clf.get_label_size_dict()


        
    def getImage(self):
        i = np.random.randint(0,5501)
        filenum = str(i)
        # filenum = "005192"
        # filenum = "004167"
        while len(filenum) < 6:
            filenum = '0'+ filenum
        filename = "CATER_new_{}.png".format(filenum)
        filename = os.path.join(self.dirname, filename)
        if not os.path.isfile(filename):
            filename = "test.png"
        print('\n',5*'>>>>>>>>','open file: '+ filename.format(str(i*10)))
        img = cv2.imread(filename.format(str(i*10)))
        raw_img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        return raw_img, filenum
    
    def getDict(self):
        raw_img, filenum = self.getImage()
        contours, bbox_list, attr_list = self.pd.getContoursWithBbox(raw_img)
        attr_val_hsv = []
        attr_val_size = []
        hsv_list = []
        center_list = []
        size_list = []
        for attr in attr_list:
            list1 = []
            list1.append(attr[0])
            list1.append(attr[4])
            list1.extend(attr[3])
            size_list.append(list1)
            hsv_list.append(attr[1])
            center_list.append(attr[3])
        all_pred_val_size, predict_val_size = self.clf.predict_size(size_list)
        for val in predict_val_size:
            attr_val_size.append(self.label_convert_size_dict[val])

        all_pred_val_hsv, predict_val_hsv = self.clf.predict_hsv(hsv_list)
        for val in predict_val_hsv:
            attr_val_hsv.append(self.label_convert_hsv_dict[val])

        single_dict = {filenum: dict(color_material = attr_val_hsv,
                                      all_prediction = all_pred_val_hsv,
                                      center = center_list,
                                      bbox = bbox_list,
                                      contours = contours,
                                      size = attr_val_size)}

        cnt = 0
        for c,b in zip(single_dict[filenum]["contours"], single_dict[filenum]["bbox"]):
            print( single_dict[filenum]["color_material"][cnt])
            print( single_dict[filenum]["size"][cnt])
            b = np.array(b).reshape(1, -1)
            print(b)
            self.pd._drawBboxOnImg(raw_img, b)
            dispImg("raw",raw_img)
            # self.pd._dispAllContours(raw_img,c,b)
            cnt += 1
        
        # print(single_dict[filename]["color_material"])
        return single_dict


if __name__ == "__main__":
    gd = generateDataset(dirname)
    for i in range(10):
        d = gd.getDict()
    # print(d)
