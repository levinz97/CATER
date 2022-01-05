import json
import numpy as np
#from generate_dataset import generateDataset

class Coco:

    def __init__(self):
        
        self.instance = {}
        self.instance["licenses"] = [{"name": "", "id": 0, "url": ""}]

        self.instance["info"] = {"contributor": "", 
                                 "date_created": "", 
                                 "description": "", 
                                 "url": "", 
                                 "version": "", 
                                 "year": ""}

        self.instance["categories"] = [{"id": 1, "name": "spl_yellow_large_metal", "supercategory": ""}, {"id": 2, "name": "spl_yellow_large_rubber", "supercategory": ""}, {"id": 3, "name": "spl_yellow_small_metal", "supercategory": ""}, {"id": 4, "name": "spl_yellow_small_rubber", "supercategory": ""}, {"id": 5, "name": "spl_yellow_medium_metal", "supercategory": ""}, {"id": 6, "name": "spl_yellow_medium_rubber", "supercategory": ""}, {"id": 7, "name": "spl_gold_large_metal", "supercategory": ""}, {"id": 8, "name": "spl_gold_large_rubber", "supercategory": ""}, {"id": 9, "name": "spl_gold_small_metal", "supercategory": ""}, {"id": 10, "name": "spl_gold_small_rubber", "supercategory": ""}, {"id": 11, "name": "spl_gold_medium_metal", "supercategory": ""}, {"id": 12, "name": "spl_gold_medium_rubber", "supercategory": ""}, {"id": 13, "name": "spl_green_large_metal", "supercategory": ""}, {"id": 14, "name": "spl_green_large_rubber", "supercategory": ""}, {"id": 15, "name": "spl_green_small_metal", "supercategory": ""}, {"id": 16, "name": "spl_green_small_rubber", "supercategory": ""}, {"id": 17, "name": "spl_green_medium_metal", "supercategory": ""}, {"id": 18, "name": "spl_green_medium_rubber", "supercategory": ""}, {"id": 19, "name": "spl_red_large_metal", "supercategory": ""}, {"id": 20, "name": "spl_red_large_rubber", "supercategory": ""}, {"id": 21, "name": "spl_red_small_metal", "supercategory": ""}, {"id": 22, "name": "spl_red_small_rubber", "supercategory": ""}, {"id": 23, "name": "spl_red_medium_metal", "supercategory": ""}, {"id": 24, "name": "spl_red_medium_rubber", "supercategory": ""}, {"id": 25, "name": "spl_brown_large_metal", "supercategory": ""}, {"id": 26, "name": "spl_brown_large_rubber", "supercategory": ""}, {"id": 27, "name": "spl_brown_small_metal", "supercategory": ""}, {"id": 28, "name": "spl_brown_small_rubber", "supercategory": ""}, {"id": 29, "name": "spl_brown_medium_metal", "supercategory": ""}, {"id": 30, "name": "spl_brown_medium_rubber", "supercategory": ""}, {"id": 31, "name": "spl_purple_large_metal", "supercategory": ""}, {"id": 32, "name": "spl_purple_large_rubber", "supercategory": ""}, {"id": 33, "name": "spl_purple_small_metal", "supercategory": ""}, {"id": 34, "name": "spl_purple_small_rubber", "supercategory": ""}, {"id": 35, "name": "spl_purple_medium_metal", "supercategory": ""}, {"id": 36, "name": "spl_purple_medium_rubber", "supercategory": ""}, {"id": 37, "name": "spl_blue_large_metal", "supercategory": ""}, {"id": 38, "name": "spl_blue_large_rubber", "supercategory": ""}, {"id": 39, "name": "spl_blue_small_metal", "supercategory": ""}, {"id": 40, "name": "spl_blue_small_rubber", "supercategory": ""}, {"id": 41, "name": "spl_blue_medium_metal", "supercategory": ""}, {"id": 42, "name": "spl_blue_medium_rubber", "supercategory": ""}, {"id": 43, "name": "spl_cyan_large_metal", "supercategory": ""}, {"id": 44, "name": "spl_cyan_large_rubber", "supercategory": ""}, {"id": 45, "name": "spl_cyan_small_metal", "supercategory": ""}, {"id": 46, "name": "spl_cyan_small_rubber", "supercategory": ""}, {"id": 47, "name": "spl_cyan_medium_metal", "supercategory": ""}, {"id": 48, "name": "spl_cyan_medium_rubber", "supercategory": ""}, {"id": 49, "name": "spl_gray_large_metal", "supercategory": ""}, {"id": 50, "name": "spl_gray_large_rubber", "supercategory": ""}, {"id": 51, "name": "spl_gray_small_metal", "supercategory": ""}, {"id": 52, "name": "spl_gray_small_rubber", "supercategory": ""}, {"id": 53, "name": "spl_gray_medium_metal", "supercategory": ""}, {"id": 54, "name": "spl_gray_medium_rubber", "supercategory": ""}, {"id": 55, "name": "cone_yellow_large_metal", "supercategory": ""}, {"id": 56, "name": "cone_yellow_large_rubber", "supercategory": ""}, {"id": 57, "name": "cone_yellow_small_metal", "supercategory": ""}, {"id": 58, "name": "cone_yellow_small_rubber", "supercategory": ""}, {"id": 59, "name": "cone_yellow_medium_metal", "supercategory": ""}, {"id": 60, "name": "cone_yellow_medium_rubber", "supercategory": ""}, {"id": 61, "name": "cone_gold_large_metal", "supercategory": ""}, {"id": 62, "name": "cone_gold_large_rubber", "supercategory": ""}, {"id": 63, "name": "cone_gold_small_metal", "supercategory": ""}, {"id": 64, "name": "cone_gold_small_rubber", "supercategory": ""}, {"id": 65, "name": "cone_gold_medium_metal", "supercategory": ""}, {"id": 66, "name": "cone_gold_medium_rubber", "supercategory": ""}, {"id": 67, "name": "cone_green_large_metal", "supercategory": ""}, {"id": 68, "name": "cone_green_large_rubber", "supercategory": ""}, {"id": 69, "name": "cone_green_small_metal", "supercategory": ""}, {"id": 70, "name": "cone_green_small_rubber", "supercategory": ""}, {"id": 71, "name": "cone_green_medium_metal", "supercategory": ""}, {"id": 72, "name": "cone_green_medium_rubber", "supercategory": ""}, {"id": 73, "name": "cone_red_large_metal", "supercategory": ""}, {"id": 74, "name": "cone_red_large_rubber", "supercategory": ""}, {"id": 75, "name": "cone_red_small_metal", "supercategory": ""}, {"id": 76, "name": "cone_red_small_rubber", "supercategory": ""}, {"id": 77, "name": "cone_red_medium_metal", "supercategory": ""}, {"id": 78, "name": "cone_red_medium_rubber", "supercategory": ""}, {"id": 79, "name": "cone_brown_large_metal", "supercategory": ""}, {"id": 80, "name": "cone_brown_large_rubber", "supercategory": ""}, {"id": 81, "name": "cone_brown_small_metal", "supercategory": ""}, {"id": 82, "name": "cone_brown_small_rubber", "supercategory": ""}, {"id": 83, "name": "cone_brown_medium_metal", "supercategory": ""}, {"id": 84, "name": "cone_brown_medium_rubber", "supercategory": ""}, {"id": 85, "name": "cone_purple_large_metal", "supercategory": ""}, {"id": 86, "name": "cone_purple_large_rubber", "supercategory": ""}, {"id": 87, "name": "cone_purple_small_metal", "supercategory": ""}, {"id": 88, "name": "cone_purple_small_rubber", "supercategory": ""}, {"id": 89, "name": "cone_purple_medium_metal", "supercategory": ""}, {"id": 90, "name": "cone_purple_medium_rubber", "supercategory": ""}, {"id": 91, "name": "cone_blue_large_metal", "supercategory": ""}, {"id": 92, "name": "cone_blue_large_rubber", "supercategory": ""}, {"id": 93, "name": "cone_blue_small_metal", "supercategory": ""}, {"id": 94, "name": "cone_blue_small_rubber", "supercategory": ""}, {"id": 95, "name": "cone_blue_medium_metal", "supercategory": ""}, {"id": 96, "name": "cone_blue_medium_rubber", "supercategory": ""}, {"id": 97, "name": "cone_cyan_large_metal", "supercategory": ""}, {"id": 98, "name": "cone_cyan_large_rubber", "supercategory": ""}, {"id": 99, "name": "cone_cyan_small_metal", "supercategory": ""}, {"id": 100, "name": "cone_cyan_small_rubber", "supercategory": ""}, {"id": 101, "name": "cone_cyan_medium_metal", "supercategory": ""}, {"id": 102, "name": "cone_cyan_medium_rubber", "supercategory": ""}, {"id": 103, "name": "cone_gray_large_metal", "supercategory": ""}, {"id": 104, "name": "cone_gray_large_rubber", "supercategory": ""}, {"id": 105, "name": "cone_gray_small_metal", "supercategory": ""}, {"id": 106, "name": "cone_gray_small_rubber", "supercategory": ""}, {"id": 107, "name": "cone_gray_medium_metal", "supercategory": ""}, {"id": 108, "name": "cone_gray_medium_rubber", "supercategory": ""}, {"id": 109, "name": "cube_yellow_large_metal", "supercategory": ""}, {"id": 110, "name": "cube_yellow_large_rubber", "supercategory": ""}, {"id": 111, "name": "cube_yellow_small_metal", "supercategory": ""}, {"id": 112, "name": "cube_yellow_small_rubber", "supercategory": ""}, {"id": 113, "name": "cube_yellow_medium_metal", "supercategory": ""}, {"id": 114, "name": "cube_yellow_medium_rubber", "supercategory": ""}, {"id": 115, "name": "cube_gold_large_metal", "supercategory": ""}, {"id": 116, "name": "cube_gold_large_rubber", "supercategory": ""}, {"id": 117, "name": "cube_gold_small_metal", "supercategory": ""}, {"id": 118, "name": "cube_gold_small_rubber", "supercategory": ""}, {"id": 119, "name": "cube_gold_medium_metal", "supercategory": ""}, {"id": 120, "name": "cube_gold_medium_rubber", "supercategory": ""}, {"id": 121, "name": "cube_green_large_metal", "supercategory": ""}, {"id": 122, "name": "cube_green_large_rubber", "supercategory": ""}, {"id": 123, "name": "cube_green_small_metal", "supercategory": ""}, {"id": 124, "name": "cube_green_small_rubber", "supercategory": ""}, {"id": 125, "name": "cube_green_medium_metal", "supercategory": ""}, {"id": 126, "name": "cube_green_medium_rubber", "supercategory": ""}, {"id": 127, "name": "cube_red_large_metal", "supercategory": ""}, {"id": 128, "name": "cube_red_large_rubber", "supercategory": ""}, {"id": 129, "name": "cube_red_small_metal", "supercategory": ""}, {"id": 130, "name": "cube_red_small_rubber", "supercategory": ""}, {"id": 131, "name": "cube_red_medium_metal", "supercategory": ""}, {"id": 132, "name": "cube_red_medium_rubber", "supercategory": ""}, {"id": 133, "name": "cube_brown_large_metal", "supercategory": ""}, {"id": 134, "name": "cube_brown_large_rubber", "supercategory": ""}, {"id": 135, "name": "cube_brown_small_metal", "supercategory": ""}, {"id": 136, "name": "cube_brown_small_rubber", "supercategory": ""}, {"id": 137, "name": "cube_brown_medium_metal", "supercategory": ""}, {"id": 138, "name": "cube_brown_medium_rubber", "supercategory": ""}, {"id": 139, "name": "cube_purple_large_metal", "supercategory": ""}, {"id": 140, "name": "cube_purple_large_rubber", "supercategory": ""}, {"id": 141, "name": "cube_purple_small_metal", "supercategory": ""}, {"id": 142, "name": "cube_purple_small_rubber", "supercategory": ""}, {"id": 143, "name": "cube_purple_medium_metal", "supercategory": ""}, {"id": 144, "name": "cube_purple_medium_rubber", "supercategory": ""}, {"id": 145, "name": "cube_blue_large_metal", "supercategory": ""}, {"id": 146, "name": "cube_blue_large_rubber", "supercategory": ""}, {"id": 147, "name": "cube_blue_small_metal", "supercategory": ""}, {"id": 148, "name": "cube_blue_small_rubber", "supercategory": ""}, {"id": 149, "name": "cube_blue_medium_metal", "supercategory": ""}, {"id": 150, "name": "cube_blue_medium_rubber", "supercategory": ""}, {"id": 151, "name": "cube_cyan_large_metal", "supercategory": ""}, {"id": 152, "name": "cube_cyan_large_rubber", "supercategory": ""}, {"id": 153, "name": "cube_cyan_small_metal", "supercategory": ""}, {"id": 154, "name": "cube_cyan_small_rubber", "supercategory": ""}, {"id": 155, "name": "cube_cyan_medium_metal", "supercategory": ""}, {"id": 156, "name": "cube_cyan_medium_rubber", "supercategory": ""}, {"id": 157, "name": "cube_gray_large_metal", "supercategory": ""}, {"id": 158, "name": "cube_gray_large_rubber", "supercategory": ""}, {"id": 159, "name": "cube_gray_small_metal", "supercategory": ""}, {"id": 160, "name": "cube_gray_small_rubber", "supercategory": ""}, {"id": 161, "name": "cube_gray_medium_metal", "supercategory": ""}, {"id": 162, "name": "cube_gray_medium_rubber", "supercategory": ""}, {"id": 163, "name": "sphere_yellow_large_metal", "supercategory": ""}, {"id": 164, "name": "sphere_yellow_large_rubber", "supercategory": ""}, {"id": 165, "name": "sphere_yellow_small_metal", "supercategory": ""}, {"id": 166, "name": "sphere_yellow_small_rubber", "supercategory": ""}, {"id": 167, "name": "sphere_yellow_medium_metal", "supercategory": ""}, {"id": 168, "name": "sphere_yellow_medium_rubber", "supercategory": ""}, {"id": 169, "name": "sphere_gold_large_metal", "supercategory": ""}, {"id": 170, "name": "sphere_gold_large_rubber", "supercategory": ""}, {"id": 171, "name": "sphere_gold_small_metal", "supercategory": ""}, {"id": 172, "name": "sphere_gold_small_rubber", "supercategory": ""}, {"id": 173, "name": "sphere_gold_medium_metal", "supercategory": ""}, {"id": 174, "name": "sphere_gold_medium_rubber", "supercategory": ""}, {"id": 175, "name": "sphere_green_large_metal", "supercategory": ""}, {"id": 176, "name": "sphere_green_large_rubber", "supercategory": ""}, {"id": 177, "name": "sphere_green_small_metal", "supercategory": ""}, {"id": 178, "name": "sphere_green_small_rubber", "supercategory": ""}, {"id": 179, "name": "sphere_green_medium_metal", "supercategory": ""}, {"id": 180, "name": "sphere_green_medium_rubber", "supercategory": ""}, {"id": 181, "name": "sphere_red_large_metal", "supercategory": ""}, {"id": 182, "name": "sphere_red_large_rubber", "supercategory": ""}, {"id": 183, "name": "sphere_red_small_metal", "supercategory": ""}, {"id": 184, "name": "sphere_red_small_rubber", "supercategory": ""}, {"id": 185, "name": "sphere_red_medium_metal", "supercategory": ""}, {"id": 186, "name": "sphere_red_medium_rubber", "supercategory": ""}, {"id": 187, "name": "sphere_brown_large_metal", "supercategory": ""}, {"id": 188, "name": "sphere_brown_large_rubber", "supercategory": ""}, {"id": 189, "name": "sphere_brown_small_metal", "supercategory": ""}, {"id": 190, "name": "sphere_brown_small_rubber", "supercategory": ""}, {"id": 191, "name": "sphere_brown_medium_metal", "supercategory": ""}, {"id": 192, "name": "sphere_brown_medium_rubber", "supercategory": ""}, {"id": 193, "name": "sphere_purple_large_metal", "supercategory": ""}, {"id": 194, "name": "sphere_purple_large_rubber", "supercategory": ""}, {"id": 195, "name": "sphere_purple_small_metal", "supercategory": ""}, {"id": 196, "name": "sphere_purple_small_rubber", "supercategory": ""}, {"id": 197, "name": "sphere_purple_medium_metal", "supercategory": ""}, {"id": 198, "name": "sphere_purple_medium_rubber", "supercategory": ""}, {"id": 199, "name": "sphere_blue_large_metal", "supercategory": ""}, {"id": 200, "name": "sphere_blue_large_rubber", "supercategory": ""}, {"id": 201, "name": "sphere_blue_small_metal", "supercategory": ""}, {"id": 202, "name": "sphere_blue_small_rubber", "supercategory": ""}, {"id": 203, "name": "sphere_blue_medium_metal", "supercategory": ""}, {"id": 204, "name": "sphere_blue_medium_rubber", "supercategory": ""}, {"id": 205, "name": "sphere_cyan_large_metal", "supercategory": ""}, {"id": 206, "name": "sphere_cyan_large_rubber", "supercategory": ""}, {"id": 207, "name": "sphere_cyan_small_metal", "supercategory": ""}, {"id": 208, "name": "sphere_cyan_small_rubber", "supercategory": ""}, {"id": 209, "name": "sphere_cyan_medium_metal", "supercategory": ""}, {"id": 210, "name": "sphere_cyan_medium_rubber", "supercategory": ""}, {"id": 211, "name": "sphere_gray_large_metal", "supercategory": ""}, {"id": 212, "name": "sphere_gray_large_rubber", "supercategory": ""}, {"id": 213, "name": "sphere_gray_small_metal", "supercategory": ""}, {"id": 214, "name": "sphere_gray_small_rubber", "supercategory": ""}, {"id": 215, "name": "sphere_gray_medium_metal", "supercategory": ""}, {"id": 216, "name": "sphere_gray_medium_rubber", "supercategory": ""}, {"id": 217, "name": "cylinder_yellow_large_metal", "supercategory": ""}, {"id": 218, "name": "cylinder_yellow_large_rubber", "supercategory": ""}, {"id": 219, "name": "cylinder_yellow_small_metal", "supercategory": ""}, {"id": 220, "name": "cylinder_yellow_small_rubber", "supercategory": ""}, {"id": 221, "name": "cylinder_yellow_medium_metal", "supercategory": ""}, {"id": 222, "name": "cylinder_yellow_medium_rubber", "supercategory": ""}, {"id": 223, "name": "cylinder_gold_large_metal", "supercategory": ""}, {"id": 224, "name": "cylinder_gold_large_rubber", "supercategory": ""}, {"id": 225, "name": "cylinder_gold_small_metal", "supercategory": ""}, {"id": 226, "name": "cylinder_gold_small_rubber", "supercategory": ""}, {"id": 227, "name": "cylinder_gold_medium_metal", "supercategory": ""}, {"id": 228, "name": "cylinder_gold_medium_rubber", "supercategory": ""}, {"id": 229, "name": "cylinder_green_large_metal", "supercategory": ""}, {"id": 230, "name": "cylinder_green_large_rubber", "supercategory": ""}, {"id": 231, "name": "cylinder_green_small_metal", "supercategory": ""}, {"id": 232, "name": "cylinder_green_small_rubber", "supercategory": ""}, {"id": 233, "name": "cylinder_green_medium_metal", "supercategory": ""}, {"id": 234, "name": "cylinder_green_medium_rubber", "supercategory": ""}, {"id": 235, "name": "cylinder_red_large_metal", "supercategory": ""}, {"id": 236, "name": "cylinder_red_large_rubber", "supercategory": ""}, {"id": 237, "name": "cylinder_red_small_metal", "supercategory": ""}, {"id": 238, "name": "cylinder_red_small_rubber", "supercategory": ""}, {"id": 239, "name": "cylinder_red_medium_metal", "supercategory": ""}, {"id": 240, "name": "cylinder_red_medium_rubber", "supercategory": ""}, {"id": 241, "name": "cylinder_brown_large_metal", "supercategory": ""}, {"id": 242, "name": "cylinder_brown_large_rubber", "supercategory": ""}, {"id": 243, "name": "cylinder_brown_small_metal", "supercategory": ""}, {"id": 244, "name": "cylinder_brown_small_rubber", "supercategory": ""}, {"id": 245, "name": "cylinder_brown_medium_metal", "supercategory": ""}, {"id": 246, "name": "cylinder_brown_medium_rubber", "supercategory": ""}, {"id": 247, "name": "cylinder_purple_large_metal", "supercategory": ""}, {"id": 248, "name": "cylinder_purple_large_rubber", "supercategory": ""}, {"id": 249, "name": "cylinder_purple_small_metal", "supercategory": ""}, {"id": 250, "name": "cylinder_purple_small_rubber", "supercategory": ""}, {"id": 251, "name": "cylinder_purple_medium_metal", "supercategory": ""}, {"id": 252, "name": "cylinder_purple_medium_rubber", "supercategory": ""}, {"id": 253, "name": "cylinder_blue_large_metal", "supercategory": ""}, {"id": 254, "name": "cylinder_blue_large_rubber", "supercategory": ""}, {"id": 255, "name": "cylinder_blue_small_metal", "supercategory": ""}, {"id": 256, "name": "cylinder_blue_small_rubber", "supercategory": ""}, {"id": 257, "name": "cylinder_blue_medium_metal", "supercategory": ""}, {"id": 258, "name": "cylinder_blue_medium_rubber", "supercategory": ""}, {"id": 259, "name": "cylinder_cyan_large_metal", "supercategory": ""}, {"id": 260, "name": "cylinder_cyan_large_rubber", "supercategory": ""}, {"id": 261, "name": "cylinder_cyan_small_metal", "supercategory": ""}, {"id": 262, "name": "cylinder_cyan_small_rubber", "supercategory": ""}, {"id": 263, "name": "cylinder_cyan_medium_metal", "supercategory": ""}, {"id": 264, "name": "cylinder_cyan_medium_rubber", "supercategory": ""}, {"id": 265, "name": "cylinder_gray_large_metal", "supercategory": ""}, {"id": 266, "name": "cylinder_gray_large_rubber", "supercategory": ""}, {"id": 267, "name": "cylinder_gray_small_metal", "supercategory": ""}, {"id": 268, "name": "cylinder_gray_small_rubber", "supercategory": ""}, {"id": 269, "name": "cylinder_gray_medium_metal", "supercategory": ""}, {"id": 270, "name": "cylinder_gray_medium_rubber", "supercategory": ""}]
        self.instance["images"] = []
        self.instance["annotations"] = []
        
        self.image_id = 0
        self.annotation_id = 0

    def image(self, zeng: dict):
        
        file_name = zeng['file_name']

        self.image_id = self.image_id + 1
        image = { "id": self.image_id, 
                  "width": 320, 
                  "height": 240, 
                  "file_name": file_name, 
                  "license": 0, 
                  "flickr_url": "", 
                  "coco_url": "", 
                  "date_captured": 0}
        return image

    def annotation(self, label: dict):

        self.annotation_id = self.annotation_id + 1
        annotation = {'id': self.annotation_id,
                      'image_id': self.image_id}
        
        category_name = label['shape']+'_'+label['color']+'_'+label['size']+'_'+label['material']
        for catagory in self.instance["categories"]:
            if catagory['name'] == category_name:
                category_id = catagory['id']
                break 

        annotation['category_id'] = category_id
        annotation['segmentation'] = label['segmentation']
        annotation['area'] = label['area']
        annotation['bbox'] = label['bbox']
        annotation['iscrowd'] = 0
        annotation['attributes'] = {}
        annotation['attributes']['shape'] = label['shape']
        annotation['attributes']['color'] = label['color']
        annotation['attributes']['size'] = label['size']
        annotation['attributes']['material'] = label['material']
        annotation['attributes']['coordination_X'] = label['coordination_X']
        annotation['attributes']['coordination_Y'] = label['coordination_Y']
        annotation['attributes']['coordination_Z'] = label['coordination_Z']
        annotation['attributes']['occluded'] = False 
        return annotation

    def add_image(self, zeng: dict):
        image = self.image(zeng)
        self.instance["images"].append(image)

    def add_image_with_annotation(self, zeng: dict):

        for label in zeng['labels']:
            annotation = self.annotation(label)
            self.instance["annotations"].append(annotation)

    def save(self, output: str):
        instances = json.dumps(self.instance)

        file = open(output, 'w', encoding = 'UTF-8')
        file.write(instances)
        file.close()

if __name__=='__main__':
    
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
        print('+++++++++++++++++++++++++++++++++++++++')
        print(length)
        label = {}
        for i in range(length):
            label['segmentation'] = []
            label['segmentation'].append(d[filenum]['contours'][i].flatten().tolist())
            label['area'] = 2022
            label['bbox'] = d[filenum]['bbox'][i].tolist()
            label['shape'] = 'spl'
            label['color'] = d[filenum]['color_material'][i][0]
            label['size'] = d[filenum]['size'][i]
            label['material'] = d[filenum]['color_material'][i][1]
            label['coordination_X'] = 0
            label['coordination_Y'] = 0
            label['coordination_Z'] = 0
            zeng['labels'].append(label)
            print(zeng)
            
        coco.add_image_with_annotation(zeng)

    # 保存Coco文件，最后只执行一次即可
    output = './update.json'
    coco.save(output)
    print('success')