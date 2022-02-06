import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.image import imread
import math
import json

class Twins:

    def __init__(self, file: int, frame: str):

        self.file = file
        self.frame = frame

        opencv_path = 'D://文档//12_Vorlesungen_Stuttgart//21WS_Practical_Course_ML_CV_for_HCI/TWIN/opencv_data/CATER_new_00{}.json'.format(str(self.file))
        with open(opencv_path, 'r', encoding = 'UTF-8') as opencv_file:
            self.opencv = json.load(opencv_file)
            opencv_file.close()

        self.image_name_to_id = {'000':7, '050':6, '100':5, '150':4, '200':3, '250':2, '300':1}
        self.bbox_list = []

    def get_all_bbox(self):

        bbox_list = []
        def get_label(category_id: int):
            label = self.opencv['categories'][category_id-1]['name']
            return label
        
        def get_bbox(annotation: dict):
            bbox = {}
            bbox['object_id'] = annotation['id']
            bbox['label'] = get_label(annotation['category_id'])
            bbox['bbox'] = annotation['bbox']
            bbox['coordination_X'] = round(float(annotation['attributes']['coordination_X']),2)
            bbox['coordination_Y'] = round(float(annotation['attributes']['coordination_Y']),2)
            bbox['coordination_Z'] = round(float(annotation['attributes']['coordination_Z']),2)
            return bbox
        
        image_id = self.image_name_to_id[self.frame]

        for annotation in self.opencv['annotations']:
            if annotation['image_id'] == image_id:
                bbox = get_bbox(annotation)
                bbox_list.append(bbox)
        
        return bbox_list
        '''
        [   {'object_id': 21, 'label': 'cone_purple_small_metal', 'bbox': [117, 91, 14, 18], 
            'coordination_X': -1.46, 'coordination_Y': -0.38, 'coordination_Z': 0.43},
         ......]
         bbox : [x,y,w,h] (x,y)左上角坐标
        '''    
    
    def get_twins_bbox(self):
        
        bbox_list = self.get_all_bbox()

        twins_in_json_path = r'D:/文档/12_Vorlesungen_Stuttgart/21WS_Practical_Course_ML_CV_for_HCI/TWIN/twins_in_opencv.json'
        with open(twins_in_json_path, 'r', encoding = 'UTF-8') as opencv_file:
            self.twins = json.load(opencv_file)
            opencv_file.close()
        print(self.twins['CATER_new_00'+str(self.file)])

        for bbox in bbox_list:
            if bbox['label'] in self.twins['CATER_new_00'+str(self.file)]:
                self.bbox_list.append(bbox)

class Bbox:

    def __init__(self, file: int, frame: str, bbox_list: list):
        
        self.bbox_list = bbox_list
        self.file= file
        self.frame = frame
        
        image_path = r'D:/文档/12_Vorlesungen_Stuttgart/21WS_Practical_Course_ML_CV_for_HCI/TWIN/images/CATER_new_00{}/CATER_new_00{}_{}.png'.format(str(self.file), str(self.file), frame)
        self.image = imread(image_path)
        plt.figure(figsize=(10, 10))
        plt.imshow(self.image)

    def open_file(self):

        twins_in_opencv_path = r'D:/文档/12_Vorlesungen_Stuttgart/21WS_Practical_Course_ML_CV_for_HCI/TWIN/twins_in_opencv.json'.format(str(self.file))
        with open(twins_in_opencv_path, 'r', encoding = 'UTF-8') as twins_in_opencv_file:
            twins_in_opencv = json.load(twins_in_opencv_file)
            twins_in_opencv_file.close()
        self.twins_in_opencv = twins_in_opencv['CATER_new_00{}'.format(str(self.file))]

        twins_in_json_path = r'D:/文档/12_Vorlesungen_Stuttgart/21WS_Practical_Course_ML_CV_for_HCI/TWIN/twins_in_json.json'.format(str(self.file))
        with open(twins_in_json_path, 'r', encoding = 'UTF-8') as twins_in_json_file:
            twins_in_json = json.load(twins_in_json_file)
            twins_in_json_file.close()
        self.twins_in_json = twins_in_json['CATER_new_00{}'.format(str(self.file))]

    def draw_rectangle( self,
                        axis,        # currentAxis，坐标轴，通过plt.gca()获取
                        bbox,               # bbox，边界框，包含四个数值的list， [x1, y1, x2, y2]
                        edgecolor='k',      # edgecolor，边框线条颜色
                        facecolor='y',      # facecolor，填充颜色
                        fill=False,         # fill, 是否填充
                        linestyle='-'):     # linestype，边框线型

        rect=patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1,
                           edgecolor=edgecolor,facecolor=facecolor,fill=fill, linestyle=linestyle)
        axis.add_patch(rect)
        # patches.Rectangle((x,y), width, height,linewidth,edgecolor,facecolor,fill, linestyle)
        # (x,y):左上角坐标; width:矩形框的宽; height:矩形框的高; linewidth:线宽; edgecolor:边界颜色; facecolor:填充颜色; fill:是否填充; linestyle:线断类型

    def add_bbox(self):
        axis=plt.gca()  # 根据图片大小自动生成坐标
        #plt.text(0, 20, str(self.file)+'_'+self.frame, color='red', fontsize=20)
        #plt.title('all cropped img')

        for bbox in self.bbox_list:
            print(bbox['bbox'])
            self.draw_rectangle(axis, bbox['bbox'], edgecolor='r')
            plt.text(bbox['bbox'][0], bbox['bbox'][1], bbox['object_id'], color='black', fontsize=12)
            #plt.text(bbox['bbox'][0], bbox['bbox'][1]+5, bbox['label'], color='black', fontsize=11)
    
    def add_real_location(self):
        self.open_file()

        count = 150
        for label_opencv in self.twins_in_opencv:    # 此处label表示其名称字母，如 label = "cone_purple_small_metal"
            for label_json in self.twins_in_json:     
                try:
                    coordination = label_json[label_opencv][str(int(self.frame))]
                    for i in range(3):
                        coordination[i] = round(coordination[i],2)
                    plt.text(0, count, label_opencv, color='blue', fontsize=10)
                    plt.text(0, count+10, coordination, color='black', fontsize=10)
                    count = count+20
                except KeyError:
                    pass
        
    def add_info(self):
        self.add_bbox()
        self.add_real_location()

    def show_image(self):
        plt.show()

    def close_image(self):
        plt.close()

if __name__ == "__main__":
    
    file = 5430
    #for frame in ['300','250','200','150','100','050','000']:
    for frame in ['300']:
        t = Twins(file, frame)
        t.get_twins_bbox()
        print(t.bbox_list)

        bbox = Bbox(file, frame, t.bbox_list)
        bbox.add_info()
        bbox.show_image()
        bbox.close_image()
        print('success')

'''
# 绘制锚框
def draw_anchor_box(center, length, scales, ratios, img_height, img_width):
    """
    以center为中心，产生一系列锚框
    其中length指定了一个基准的长度
    scales是包含多种尺寸比例的list
    ratios是包含多种长宽比的list
    img_height和img_width是图片的尺寸，生成的锚框范围不能超出图片尺寸之外
    """
    bboxes = []
    for scale in scales:
        for ratio in ratios:
            h = length*scale*math.sqrt(ratio)
            w = length*scale/math.sqrt(ratio) 
            x1 = max(center[0] - w/2., 0.)
            y1 = max(center[1] - h/2., 0.)
            x2 = min(center[0] + w/2. - 1.0, img_width - 1.0)
            y2 = min(center[1] + h/2. - 1.0, img_height - 1.0)
            print(center[0], center[1], w, h)
            bboxes.append([x1, y1, x2, y2])

    for bbox in bboxes:
        draw_rectangle(currentAxis, bbox, edgecolor = 'b')

img_height = im.shape[0]
img_width = im.shape[1] 
# 绘制锚框
draw_anchor_box([300., 500.], 100., [2.0], [0.5, 1.0, 2.0], img_height, img_width)
'''