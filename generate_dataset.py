from matplotlib import pyplot as plt
from numpy.random.mtrand import f
from prepare_data import PrepareData
from simClassifier import SimClassifier
from utils import dispImg, move_figure
from Coco import Coco

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
        
    def getImage(self, filenum: int):
        i = np.random.randint(0,5501)
        
        '''
        filenum = str(i)
        filenum = "005192"
        '''
        while len(filenum) < 6:
            filenum = '0'+ filenum
        filename = "CATER_new_{}.png".format(filenum)
        filename = os.path.join(self.dirname, filename)
        #if not os.path.isfile(filename):
        #    filename = "test.png"
        print('\n',5*'>>>>>>>>','open file: '+ filename.format(str(i*10)))
        img = cv2.imread(filename.format(str(i*10)))
        raw_img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        return raw_img, filenum
    
    def getDict(self, filenum: int):
        raw_img, filenum = self.getImage(filenum)
        contours, bbox_list, attr_list = self.pd.getContoursWithBbox(raw_img)
        attr_val = []
        hsv_list = []
        center_list = []
        size_list = []
        area_list = []
        feature_size_list = []
        for attr in attr_list:
            area_list.append(attr[0]) # add area
            list1 = []
            list1.append(attr[0])
            list1.append(attr[4])
            list1.extend(attr[3])
            feature_size_list.append(list1)
            hsv_list.append(attr[1])  # add HSV
            center_list.append(attr[3]) # add center location [x,y]

        _, predict_val_size = self.clf.predict_size(feature_size_list)
        for val in predict_val_size:
            size_list.append(self.label_convert_size_dict[val])

        all_pred_val, predict_val = self.clf.predict_hsv(hsv_list)
        for val in predict_val:
            attr_val.append(self.label_convert_hsv_dict[val])

        keep_idx = []
        cnt = 0
        start_shape_annotation = dict(start=True)
        all_shape_list = [] # to store all objects' shape
        tmp_shape_list = [] # to store the single object shape
        def press(event):
            a = ['cub', 'con', 'spl', 'sph', 'cyl']
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
                print(f"shape is {tmp_shape_list[-1] if len(tmp_shape_list) > 0 else None}")
            if event.key == 'a':
                start_shape_annotation.update(dict(start=True))
                print("start annotation of shape, only the last selection will be saved")
                print([i for i in zip(range(len(a)), a)])
            if start_shape_annotation['start'] and event.key in ["{:1d}".format(x) for x in range(len(a))]:
                shape = a[int(event.key)]
                print(f'{event.key} is pressed, shape is {shape}')
                tmp_shape_list.append(shape)
            if event.key == 'i':
                dispImg("show origin image", screen, move_dist=[1200, 200])

        for c,b in zip(contours, bbox_list):
            keep = [True]
            print('\n')
            print(attr_val[cnt], end=' ')
            print(size_list[cnt], end=' ')
            b = np.array(b).reshape(1, -1)
            print(b)
            shapes = ['cub', 'con', 'spl', 'sph', 'cyl']
            print(f"pls input the shape {[i for i in zip(range(len(shapes)), shapes)]}")
            screen = raw_img.copy()
            # self.pd._drawBboxOnImg(screen, b)
            # dispImg("raw",screen)
            self.pd._dispAllContours(screen, [c], b, on_press=press)
            if np.sum(keep) > 0:
                print("annotation saved")
                if len(tmp_shape_list) > 0:
                    print(f"shape {tmp_shape_list[-1]} is saved")
                    all_shape_list.append(tmp_shape_list[-1])
                else:
                    print("[WARNING] no shape input")
                    all_shape_list.append('Unknown')
                if len(tmp_shape_list) > 0:
                    # keep the shape value as default for next object
                    tmp_shape_list = [tmp_shape_list[-1]]
                keep_idx.append(cnt)
            cnt += 1
        assert len(keep_idx) > 0, "no annotations for this image"
        assert len(all_shape_list) == len(keep_idx), "annotations of shape do not correspond with other annos!"

        contours = list(np.array(contours, dtype=object)[keep_idx])
        list_contours = [ list(i.reshape(-1,)) for i in contours]
        single_dict = {filenum: dict(
                                shape = all_shape_list,
                                area = area_list,
                                color_material = list(np.array(attr_val)[keep_idx]),
                                all_prediction = list(np.array(all_pred_val)[keep_idx]),
                                center = list(np.array(center_list)[keep_idx]),
                                bbox = list(np.array(bbox_list)[keep_idx]),
                                contours = list_contours,
                                size = list(np.array(size_list)[keep_idx]))}
        self.pd._dispAllContours(raw_img, contours, single_dict[filenum]['bbox'])

        # print(single_dict[filename]["color_material"])
        print(single_dict[filenum]['contours'])
        return single_dict


if __name__ == "__main__":
    # 初始化coco，最初只执行一次即可
    coco = Coco()
    
    # 此处添加循环，每导出一张图片，执行一次该命令
    filenum_list = [5748, 5749,5750]
    for filenum in filenum_list: 
        filenum = str(filenum)
        zeng =  {
                    'file_name': "CATER_new_00{}.png".format(str(filenum)),
                    'labels': []
                }

        coco.add_image(zeng)
        '''
        {   'segmentation': [120,45,36,48,99],
            'area': 111,
            'bbox': [15,20,25,5],
            'shape': 'spl',
            'color': 'purple', 
            'size': 'large',
            'material': 'metal',
            'coordination_X': 0,    #坐标需要是int类型而不是str
            'coordination_Y': -2,
            'coordination_Z': 0}
        '''
        dirname = '.\\all_actions_first_frame'
        gd = generateDataset(dirname)

        raw_img, filenum = gd.getImage(filenum)
        d = gd.getDict(filenum)

        length = len(d[filenum]['color_material'])
        for i in range(length):
            label = {}
            label['segmentation'] = []
            label['segmentation'].append(d[filenum]['contours'][i].flatten().tolist())
            label['area'] = d[filenum]['area']
            label['bbox'] = d[filenum]['bbox'][i].tolist()
            label['shape'] = d[filenum]['shape']
            label['color'] = d[filenum]['color_material'][i][0]
            label['size'] = d[filenum]['size'][i]
            label['material'] = d[filenum]['color_material'][i][1]
            label['coordination_X'] = 0
            label['coordination_Y'] = 0
            label['coordination_Z'] = 0
            zeng['labels'].append(label)
            
        coco.add_image_with_annotation(zeng)

    # 保存Coco文件，最后只执行一次即可
    output = './update.json'
    coco.save(output)
    print('success')
