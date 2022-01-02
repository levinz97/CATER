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
        self.label_convert_dict = self.clf.get_label_dict()
        
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
        attr_val = []
        hsv_list = []
        center_list = []
        size_list = []
        for attr in attr_list:
            if attr[0] > 1000:
                size_list.append("large")
            elif attr[0] < 400:
                size_list.append("small")
            else:
                size_list.append("medium")
            hsv_list.append(attr[1])
            center_list.append(attr[3])
        all_pred_val, predict_val = self.clf.predict(hsv_list)
        for val in predict_val:
            attr_val.append(self.label_convert_dict[val])

        single_dict = {filenum: dict(color_material = attr_val, 
                                      all_prediction = all_pred_val,
                                      center = center_list,
                                      bbox = bbox_list,
                                      contours = contours,
                                      size = size_list)}

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
