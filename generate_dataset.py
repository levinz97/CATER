from prepare_data import PrepareData
from simClassifier import SimClassifier
from utils import dispImg, StatusLogger
from Coco import Coco_annotation

import numpy as np
import os
import cv2

# dirname = os.path.join('.','raw_data','first_frame', 'all_actions_first_frame')
dirname = os.path.join('.','raw_data', 'raw_data_from_005200_to_005699_sort')

class GenerateDataset:
    def __init__(self, dirname:str, reset_logger = False):
        self.dirname = dirname
        self.pd = PrepareData(need_visualization=False)
        # train a simple classifier for color, material prediction
        self.clf = SimClassifier(class_label="c*m")
        self.clf.train()
        self.label_convert_hsv_dict = self.clf.get_label_hsv_dict()
        self.label_convert_size_dict = self.clf.get_label_size_dict()
        self.status_logger = StatusLogger(file='status.json')
        if reset_logger:
            self.status_logger.reset_logger()

    @staticmethod
    def getFullFilenum(filenum):
        while len(filenum) < 6:
            filenum = '0'+ filenum
        return filenum

    def getImage(self, filename: str):
        '''
        filename: image path, image must in RGB format
        return concise filenum, 
        for example: 
        filename = "/some paths/CATER_new_005192_000.png"
        filenum  = "005192_000"
        '''
        filenum = filename[-14:-4]
        if not os.path.isfile(filename):
            print(f'no file found for {filename}')
            filename = "test.png"
        print('\n[INFO][generate_dataset]','open file: '+ filename)
        filename = './raw_data/raw_data_from_005200_to_005699_sort/005200-005299_sort/CATER_new_005202/CATER_new_005202_100.png'
        img = cv2.imread(filename)
        raw_img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        return raw_img, filenum
    
    def getDict(self, raw_img, filenum: int):
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
        tmp_shape_list = ['cube'] # to store the single object shape, default as cubic
        def press(event):
            a = ['cube', 'cone', 'spl', 'sphere', 'cylinder']
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
            print(attr_val[cnt], end=' ')
            print(size_list[cnt], end=' ')
            b = np.array(b).reshape(1, -1)
            print(b)
            shapes = ['cube', 'cone', 'spl', 'sphere', 'cylinder']
            print(f"\npls input the shape {[i for i in zip(range(len(shapes)), shapes)]}")
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
                    print("[WARNING][generate_dataset] no shape input")
                    all_shape_list.append('Unknown')
                if len(tmp_shape_list) > 0:
                    # keep the shape value as default for next object
                    tmp_shape_list = [tmp_shape_list[-1]]
                keep_idx.append(cnt)
            cnt += 1
        assert len(keep_idx) > 0, "no annotations for this image"
        assert len(all_shape_list) == len(keep_idx), "annotations of shape do not correspond with other annos!"
        print(f"bbox_list has {len(bbox_list)} items, contours has {len(contours)}")
        contours_keep = [contours[i] for i in range(len(contours)) if i in keep_idx]
        # contours = np.array(contours, dtype=object)[keep_idx]
        list_contours = [ i.reshape(-1,).astype('int').tolist() for i in contours_keep]
        single_dict = {filenum: dict(
                                shape = all_shape_list,
                                area = area_list,
                                color_material = list(np.array(attr_val)[keep_idx]),
                                all_prediction = list(np.array(all_pred_val)[keep_idx]),
                                center = list(np.array(center_list)[keep_idx]),
                                bbox = list(np.array(bbox_list)[keep_idx]),
                                contours = list_contours,
                                size = list(np.array(size_list)[keep_idx]))}
        self.pd._dispAllContours(raw_img, contours_keep, single_dict[filenum]['bbox'])
        # print(type(list_contours[0][0]))
        # print(single_dict[filename]["color_material"])
        # print(single_dict[filenum]['contours'])
        return single_dict

    def generate(self):
        # dirname is the top directory containing all raw images
        for subdir in sorted(os.listdir(dirname)):
            _, finished_secdir, _ = self.status_logger.get_status()
            if subdir in finished_secdir:
                continue
            secdir = os.path.join(dirname, subdir)
            print(f"[INFO][generate_dataset] start to annotate in folder {subdir}")
            if not os.path.isdir(secdir):
                continue
            for subsubdir in sorted(os.listdir(secdir)):
                _, _, finished_thirdir = self.status_logger.get_status()
                if subsubdir in finished_thirdir:
                    continue
                thirdir = os.path.join(dirname, subdir, subsubdir)
                print(f"[INFO][generate_dataset] annotate video {subsubdir}")
                self.status_logger.update_status(current_dir=subsubdir)
                if not os.path.isdir(thirdir):
                    continue
                self.generate_in_video_folder(thirdir)
                
                print(f'[INFO][generate_dataset] finished annotating video {subsubdir}')
                self.status_logger.update_status(finished_thirdir=subsubdir)

            print(f'[INFO][generate_dataset] finished annotating folder {subdir}')
            self.status_logger.update_status(finished_secdir=subdir)

    def generate_in_video_folder(self, thirdir):
        """
            thridir: the folder for all images in a single video
        """
        IMG_EXTENSIONS = ['.png','.jpg']
        is_image_file = lambda filename : any(filename.endswith(ext) for ext in IMG_EXTENSIONS)
        for root, _, fnames in sorted(os.walk(thirdir)):
            coco = Coco_annotation() # 初始 化coco，最初只执行一次即可
            for fn in sorted(fnames):
                if is_image_file(fn):
                    path = os.path.join(root, fn)
                    raw_img, filenum = self.getImage(path)
                    d = self.getDict(raw_img, filenum)
                    zeng =  {
                                'file_name': "CATER_new_{}.png".format(str(filenum)),
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
                    length = len(d[filenum]['color_material'])
                    print('+++++++++++++++++++++++++++++++++++++++'*3)
                    print(f"{length} annotations in the image {filenum}")
                    for i in range(length):
                        label = {}
                        label['segmentation'] = []
                        label['segmentation'].append(d[filenum]['contours'][i])
                        label['area'] = d[filenum]['area'][i]
                        label['bbox'] = d[filenum]['bbox'][i].tolist()
                        label['shape'] = d[filenum]['shape'][i]
                        label['color'] = d[filenum]['color_material'][i][0]
                        label['size'] = d[filenum]['size'][i]
                        label['material'] = d[filenum]['color_material'][i][1]
                        label['coordination_X'] = 0
                        label['coordination_Y'] = 0
                        label['coordination_Z'] = 0
                        zeng['labels'].append(label)
                        # print(zeng)
                    coco.add_image_with_annotation(zeng)
            # 保存Coco文件，最后只执行一次即可
            output = 'dataset/{}.json'.format(thirdir[-16:])
            coco.save(output)
            print(f'success saved to {output}')

if __name__ == "__main__":
    cv2.setUseOptimized(True)
    cv2.setUseOptimized(4)
    dataset_generator = GenerateDataset(dirname)
    dataset_generator.generate()