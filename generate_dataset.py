from matplotlib import pyplot as plt
from numpy.random.mtrand import f
from prepare_data import PrepareData
from simClassifier import SimClassifier
from utils import dispImg, move_figure

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

        keep_idx = []
        cnt = 0
        def press(event):
            if event.key == 'c':
                print("clear all val in keep")
                keep.clear()
            if event.key == 'r':
                print(f"save {cnt} annotation")
                keep.append(1)
            if event.key == 'd':
                print(f"delete {cnt} annotation")
                keep.append(-1)
            if event.key == 'v':
                print(f"annotations to be kept = {True if np.sum(keep) > 0 else False}")
        for c,b in zip(contours, bbox_list):
            keep = [True]
            print(attr_val[cnt])
            print(size_list[cnt])
            b = np.array(b).reshape(1, -1)
            print(b)
            screen = raw_img.copy()
            # self.pd._drawBboxOnImg(screen, b)
            # dispImg("raw",screen)
            self.pd._dispAllContours(screen, [c], b, on_press=press)
            if np.sum(keep) > 0:
                print("annotation saved")
                keep_idx.append(cnt)
            cnt += 1
        assert len(keep_idx) > 0, "no annotations for this image"
        single_dict = {filenum: dict(color_material = list(np.array(attr_val)[keep_idx]), 
                                all_prediction = list(np.array(all_pred_val)[keep_idx]),
                                center = list(np.array(center_list)[keep_idx]),
                                bbox = list(np.array(bbox_list)[keep_idx]),
                                contours = list(np.array(contours, dtype=object)[keep_idx]),
                                size = list(np.array(size_list)[keep_idx]))}
        self.pd._dispAllContours(raw_img, single_dict[filenum]['contours'], single_dict[filenum]['bbox'])

        # print(single_dict[filename]["color_material"])
        return single_dict


if __name__ == "__main__":
    gd = generateDataset(dirname)
    for i in range(10):
        d = gd.getDict()
        print(d)
    # print(d)
