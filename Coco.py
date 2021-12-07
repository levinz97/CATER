import json

class Coco:

    def __init__(self):
        
        self.instance = {}
        self.instance["licenses"] = [{"name": "", "id": 0, "url": ""}]
        self.instance["info"] = {"contributor": "", 
                                 "date_created": "", 
                                 "description": "", 
                                 "url": "", 
                                 "version": "", 
                                 "year": ""}, 
        self.instance["categories"] = [ {'id': 1, 'name': 'spl_gold_small_metal'},
                                        {'id': 2, 'name': 'spl_gold_small_rubber'}, 
                                        {'id': 3, 'name': 'spl_gold_medium_metal'}, 
                                        {'id': 4, 'name': 'spl_gold_medium_rubber'},
                                        {'id': 5, 'name': 'spl_gold_large_metal'},  
                                        {'id': 6, 'name': 'spl_gold_large_rubber'}, 
                                        {'id': 7, 'name': 'spl_blue_small_metal'},  
                                        {'id': 8, 'name': 'spl_blue_small_rubber'}, 
                                        {'id': 9, 'name': 'spl_blue_medium_metal'},
                                        {'id': 10, 'name': 'spl_blue_medium_rubber'},
                                        {'id': 11, 'name': 'spl_blue_large_metal'},
                                        {'id': 12, 'name': 'spl_blue_large_rubber'},
                                        {'id': 13, 'name': 'spl_yellow_small_metal'},
                                        {'id': 14, 'name': 'spl_yellow_small_rubber'},
                                        {'id': 15, 'name': 'spl_yellow_medium_metal'},
                                        {'id': 16, 'name': 'spl_yellow_medium_rubber'},
                                        {'id': 17, 'name': 'spl_yellow_large_metal'},
                                        {'id': 18, 'name': 'spl_yellow_large_rubber'},
                                        {'id': 19, 'name': 'spl_red_small_metal'},
                                        {'id': 20, 'name': 'spl_red_small_rubber'},
                                        {'id': 21, 'name': 'spl_red_medium_metal'},
                                        {'id': 22, 'name': 'spl_red_medium_rubber'},
                                        {'id': 23, 'name': 'spl_red_large_metal'},
                                        {'id': 24, 'name': 'spl_red_large_rubber'},
                                        {'id': 25, 'name': 'spl_brown_small_metal'},
                                        {'id': 26, 'name': 'spl_brown_small_rubber'},
                                        {'id': 27, 'name': 'spl_brown_medium_metal'},
                                        {'id': 28, 'name': 'spl_brown_medium_rubber'},
                                        {'id': 29, 'name': 'spl_brown_large_metal'},
                                        {'id': 30, 'name': 'spl_brown_large_rubber'},
                                        {'id': 31, 'name': 'spl_green_small_metal'},
                                        {'id': 32, 'name': 'spl_green_small_rubber'},
                                        {'id': 33, 'name': 'spl_green_medium_metal'},
                                        {'id': 34, 'name': 'spl_green_medium_rubber'},
                                        {'id': 35, 'name': 'spl_green_large_metal'},
                                        {'id': 36, 'name': 'spl_green_large_rubber'},
                                        {'id': 37, 'name': 'spl_cyan_small_metal'},
                                        {'id': 38, 'name': 'spl_cyan_small_rubber'},
                                        {'id': 39, 'name': 'spl_cyan_medium_metal'},
                                        {'id': 40, 'name': 'spl_cyan_medium_rubber'},
                                        {'id': 41, 'name': 'spl_cyan_large_metal'},
                                        {'id': 42, 'name': 'spl_cyan_large_rubber'},
                                        {'id': 43, 'name': 'spl_purple_small_metal'},
                                        {'id': 44, 'name': 'spl_purple_small_rubber'},
                                        {'id': 45, 'name': 'spl_purple_medium_metal'},
                                        {'id': 46, 'name': 'spl_purple_medium_rubber'},
                                        {'id': 47, 'name': 'spl_purple_large_metal'},
                                        {'id': 48, 'name': 'spl_purple_large_rubber'},
                                        {'id': 49, 'name': 'spl_gray_small_metal'},
                                        {'id': 50, 'name': 'spl_gray_small_rubber'},
                                        {'id': 51, 'name': 'spl_gray_medium_metal'},
                                        {'id': 52, 'name': 'spl_gray_medium_rubber'},
                                        {'id': 53, 'name': 'spl_gray_large_metal'},
                                        {'id': 54, 'name': 'spl_gray_large_rubber'},
                                        {'id': 55, 'name': 'cone_gold_small_metal'},
                                        {'id': 56, 'name': 'cone_gold_small_rubber'},
                                        {'id': 57, 'name': 'cone_gold_medium_metal'},
                                        {'id': 58, 'name': 'cone_gold_medium_rubber'},
                                        {'id': 59, 'name': 'cone_gold_large_metal'},
                                        {'id': 60, 'name': 'cone_gold_large_rubber'},
                                        {'id': 61, 'name': 'cone_blue_small_metal'},
                                        {'id': 62, 'name': 'cone_blue_small_rubber'},
                                        {'id': 63, 'name': 'cone_blue_medium_metal'},
                                        {'id': 64, 'name': 'cone_blue_medium_rubber'},
                                        {'id': 65, 'name': 'cone_blue_large_metal'},
                                        {'id': 66, 'name': 'cone_blue_large_rubber'},
                                        {'id': 67, 'name': 'cone_yellow_small_metal'},
                                        {'id': 68, 'name': 'cone_yellow_small_rubber'},
                                        {'id': 69, 'name': 'cone_yellow_medium_metal'},
                                        {'id': 70, 'name': 'cone_yellow_medium_rubber'},
                                        {'id': 71, 'name': 'cone_yellow_large_metal'},
                                        {'id': 72, 'name': 'cone_yellow_large_rubber'},
                                        {'id': 73, 'name': 'cone_red_small_metal'},
                                        {'id': 74, 'name': 'cone_red_small_rubber'},
                                        {'id': 75, 'name': 'cone_red_medium_metal'},
                                        {'id': 76, 'name': 'cone_red_medium_rubber'},
                                        {'id': 77, 'name': 'cone_red_large_metal'},
                                        {'id': 78, 'name': 'cone_red_large_rubber'},
                                        {'id': 79, 'name': 'cone_brown_small_metal'},
                                        {'id': 80, 'name': 'cone_brown_small_rubber'},
                                        {'id': 81, 'name': 'cone_brown_medium_metal'},
                                        {'id': 82, 'name': 'cone_brown_medium_rubber'},
                                        {'id': 83, 'name': 'cone_brown_large_metal'},
                                        {'id': 84, 'name': 'cone_brown_large_rubber'},
                                        {'id': 85, 'name': 'cone_green_small_metal'},
                                        {'id': 86, 'name': 'cone_green_small_rubber'},
                                        {'id': 87, 'name': 'cone_green_medium_metal'},
                                        {'id': 88, 'name': 'cone_green_medium_rubber'},
                                        {'id': 89, 'name': 'cone_green_large_metal'},
                                        {'id': 90, 'name': 'cone_green_large_rubber'},
                                        {'id': 91, 'name': 'cone_cyan_small_metal'},
                                        {'id': 92, 'name': 'cone_cyan_small_rubber'},
                                        {'id': 93, 'name': 'cone_cyan_medium_metal'},
                                        {'id': 94, 'name': 'cone_cyan_medium_rubber'},
                                        {'id': 95, 'name': 'cone_cyan_large_metal'},
                                        {'id': 96, 'name': 'cone_cyan_large_rubber'},
                                        {'id': 97, 'name': 'cone_purple_small_metal'},
                                        {'id': 98, 'name': 'cone_purple_small_rubber'},
                                        {'id': 99, 'name': 'cone_purple_medium_metal'},
                                        {'id': 100, 'name': 'cone_purple_medium_rubber'},
                                        {'id': 101, 'name': 'cone_purple_large_metal'},
                                        {'id': 102, 'name': 'cone_purple_large_rubber'},
                                        {'id': 103, 'name': 'cone_gray_small_metal'},
                                        {'id': 104, 'name': 'cone_gray_small_rubber'},
                                        {'id': 105, 'name': 'cone_gray_medium_metal'},
                                        {'id': 106, 'name': 'cone_gray_medium_rubber'},
                                        {'id': 107, 'name': 'cone_gray_large_metal'},
                                        {'id': 108, 'name': 'cone_gray_large_rubber'},
                                        {'id': 109, 'name': 'sphere_gold_small_metal'},
                                        {'id': 110, 'name': 'sphere_gold_small_rubber'},
                                        {'id': 111, 'name': 'sphere_gold_medium_metal'},
                                        {'id': 112, 'name': 'sphere_gold_medium_rubber'},
                                        {'id': 113, 'name': 'sphere_gold_large_metal'},
                                        {'id': 114, 'name': 'sphere_gold_large_rubber'},
                                        {'id': 115, 'name': 'sphere_blue_small_metal'},
                                        {'id': 116, 'name': 'sphere_blue_small_rubber'},
                                        {'id': 117, 'name': 'sphere_blue_medium_metal'},
                                        {'id': 118, 'name': 'sphere_blue_medium_rubber'},
                                        {'id': 119, 'name': 'sphere_blue_large_metal'},
                                        {'id': 120, 'name': 'sphere_blue_large_rubber'},
                                        {'id': 121, 'name': 'sphere_yellow_small_metal'},
                                        {'id': 122, 'name': 'sphere_yellow_small_rubber'},
                                        {'id': 123, 'name': 'sphere_yellow_medium_metal'},
                                        {'id': 124, 'name': 'sphere_yellow_medium_rubber'},
                                        {'id': 125, 'name': 'sphere_yellow_large_metal'},
                                        {'id': 126, 'name': 'sphere_yellow_large_rubber'},
                                        {'id': 127, 'name': 'sphere_red_small_metal'},
                                        {'id': 128, 'name': 'sphere_red_small_rubber'},
                                        {'id': 129, 'name': 'sphere_red_medium_metal'},
                                        {'id': 130, 'name': 'sphere_red_medium_rubber'},
                                        {'id': 131, 'name': 'sphere_red_large_metal'},
                                        {'id': 132, 'name': 'sphere_red_large_rubber'},
                                        {'id': 133, 'name': 'sphere_brown_small_metal'},
                                        {'id': 134, 'name': 'sphere_brown_small_rubber'},
                                        {'id': 135, 'name': 'sphere_brown_medium_metal'},
                                        {'id': 136, 'name': 'sphere_brown_medium_rubber'},
                                        {'id': 137, 'name': 'sphere_brown_large_metal'},
                                        {'id': 138, 'name': 'sphere_brown_large_rubber'},
                                        {'id': 139, 'name': 'sphere_green_small_metal'},
                                        {'id': 140, 'name': 'sphere_green_small_rubber'},
                                        {'id': 141, 'name': 'sphere_green_medium_metal'},
                                        {'id': 142, 'name': 'sphere_green_medium_rubber'},
                                        {'id': 143, 'name': 'sphere_green_large_metal'},
                                        {'id': 144, 'name': 'sphere_green_large_rubber'},
                                        {'id': 145, 'name': 'sphere_cyan_small_metal'},
                                        {'id': 146, 'name': 'sphere_cyan_small_rubber'},
                                        {'id': 147, 'name': 'sphere_cyan_medium_metal'},
                                        {'id': 148, 'name': 'sphere_cyan_medium_rubber'},
                                        {'id': 149, 'name': 'sphere_cyan_large_metal'},
                                        {'id': 150, 'name': 'sphere_cyan_large_rubber'},
                                        {'id': 151, 'name': 'sphere_purple_small_metal'},
                                        {'id': 152, 'name': 'sphere_purple_small_rubber'},
                                        {'id': 153, 'name': 'sphere_purple_medium_metal'},
                                        {'id': 154, 'name': 'sphere_purple_medium_rubber'},
                                        {'id': 155, 'name': 'sphere_purple_large_metal'},
                                        {'id': 156, 'name': 'sphere_purple_large_rubber'},
                                        {'id': 157, 'name': 'sphere_gray_small_metal'},
                                        {'id': 158, 'name': 'sphere_gray_small_rubber'},
                                        {'id': 159, 'name': 'sphere_gray_medium_metal'},
                                        {'id': 160, 'name': 'sphere_gray_medium_rubber'},
                                        {'id': 161, 'name': 'sphere_gray_large_metal'},
                                        {'id': 162, 'name': 'sphere_gray_large_rubber'},
                                        {'id': 163, 'name': 'cylinder_gold_small_metal'},
                                        {'id': 164, 'name': 'cylinder_gold_small_rubber'},
                                        {'id': 165, 'name': 'cylinder_gold_medium_metal'},
                                        {'id': 166, 'name': 'cylinder_gold_medium_rubber'},
                                        {'id': 167, 'name': 'cylinder_gold_large_metal'},
                                        {'id': 168, 'name': 'cylinder_gold_large_rubber'},
                                        {'id': 169, 'name': 'cylinder_blue_small_metal'},
                                        {'id': 170, 'name': 'cylinder_blue_small_rubber'},
                                        {'id': 171, 'name': 'cylinder_blue_medium_metal'},
                                        {'id': 172, 'name': 'cylinder_blue_medium_rubber'},
                                        {'id': 173, 'name': 'cylinder_blue_large_metal'},
                                        {'id': 174, 'name': 'cylinder_blue_large_rubber'},
                                        {'id': 175, 'name': 'cylinder_yellow_small_metal'},
                                        {'id': 176, 'name': 'cylinder_yellow_small_rubber'},
                                        {'id': 177, 'name': 'cylinder_yellow_medium_metal'},
                                        {'id': 178, 'name': 'cylinder_yellow_medium_rubber'},
                                        {'id': 179, 'name': 'cylinder_yellow_large_metal'},
                                        {'id': 180, 'name': 'cylinder_yellow_large_rubber'},
                                        {'id': 181, 'name': 'cylinder_red_small_metal'},
                                        {'id': 182, 'name': 'cylinder_red_small_rubber'},
                                        {'id': 183, 'name': 'cylinder_red_medium_metal'},
                                        {'id': 184, 'name': 'cylinder_red_medium_rubber'},
                                        {'id': 185, 'name': 'cylinder_red_large_metal'},
                                        {'id': 186, 'name': 'cylinder_red_large_rubber'},
                                        {'id': 187, 'name': 'cylinder_brown_small_metal'},
                                        {'id': 188, 'name': 'cylinder_brown_small_rubber'},
                                        {'id': 189, 'name': 'cylinder_brown_medium_metal'},
                                        {'id': 190, 'name': 'cylinder_brown_medium_rubber'},
                                        {'id': 191, 'name': 'cylinder_brown_large_metal'},
                                        {'id': 192, 'name': 'cylinder_brown_large_rubber'},
                                        {'id': 193, 'name': 'cylinder_green_small_metal'},
                                        {'id': 194, 'name': 'cylinder_green_small_rubber'},
                                        {'id': 195, 'name': 'cylinder_green_medium_metal'},
                                        {'id': 196, 'name': 'cylinder_green_medium_rubber'},
                                        {'id': 197, 'name': 'cylinder_green_large_metal'},
                                        {'id': 198, 'name': 'cylinder_green_large_rubber'},
                                        {'id': 199, 'name': 'cylinder_cyan_small_metal'},
                                        {'id': 200, 'name': 'cylinder_cyan_small_rubber'},
                                        {'id': 201, 'name': 'cylinder_cyan_medium_metal'},
                                        {'id': 202, 'name': 'cylinder_cyan_medium_rubber'},
                                        {'id': 203, 'name': 'cylinder_cyan_large_metal'},
                                        {'id': 204, 'name': 'cylinder_cyan_large_rubber'},
                                        {'id': 205, 'name': 'cylinder_purple_small_metal'},
                                        {'id': 206, 'name': 'cylinder_purple_small_rubber'},
                                        {'id': 207, 'name': 'cylinder_purple_medium_metal'},
                                        {'id': 208, 'name': 'cylinder_purple_medium_rubber'},
                                        {'id': 209, 'name': 'cylinder_purple_large_metal'},
                                        {'id': 210, 'name': 'cylinder_purple_large_rubber'},
                                        {'id': 211, 'name': 'cylinder_gray_small_metal'},
                                        {'id': 212, 'name': 'cylinder_gray_small_rubber'},
                                        {'id': 213, 'name': 'cylinder_gray_medium_metal'},
                                        {'id': 214, 'name': 'cylinder_gray_medium_rubber'},
                                        {'id': 215, 'name': 'cylinder_gray_large_metal'},
                                        {'id': 216, 'name': 'cylinder_gray_large_rubber'},
                                        {'id': 217, 'name': 'cube_gold_small_metal'},
                                        {'id': 218, 'name': 'cube_gold_small_rubber'},
                                        {'id': 219, 'name': 'cube_gold_medium_metal'},
                                        {'id': 220, 'name': 'cube_gold_medium_rubber'},
                                        {'id': 221, 'name': 'cube_gold_large_metal'},
                                        {'id': 222, 'name': 'cube_gold_large_rubber'},
                                        {'id': 223, 'name': 'cube_blue_small_metal'},
                                        {'id': 224, 'name': 'cube_blue_small_rubber'},
                                        {'id': 225, 'name': 'cube_blue_medium_metal'},
                                        {'id': 226, 'name': 'cube_blue_medium_rubber'},
                                        {'id': 227, 'name': 'cube_blue_large_metal'},
                                        {'id': 228, 'name': 'cube_blue_large_rubber'},
                                        {'id': 229, 'name': 'cube_yellow_small_metal'},
                                        {'id': 230, 'name': 'cube_yellow_small_rubber'},
                                        {'id': 231, 'name': 'cube_yellow_medium_metal'},
                                        {'id': 232, 'name': 'cube_yellow_medium_rubber'},
                                        {'id': 233, 'name': 'cube_yellow_large_metal'},
                                        {'id': 234, 'name': 'cube_yellow_large_rubber'},
                                        {'id': 235, 'name': 'cube_red_small_metal'},
                                        {'id': 236, 'name': 'cube_red_small_rubber'},
                                        {'id': 237, 'name': 'cube_red_medium_metal'},
                                        {'id': 238, 'name': 'cube_red_medium_rubber'},
                                        {'id': 239, 'name': 'cube_red_large_metal'},
                                        {'id': 240, 'name': 'cube_red_large_rubber'},
                                        {'id': 241, 'name': 'cube_brown_small_metal'},
                                        {'id': 242, 'name': 'cube_brown_small_rubber'},
                                        {'id': 243, 'name': 'cube_brown_medium_metal'},
                                        {'id': 244, 'name': 'cube_brown_medium_rubber'},
                                        {'id': 245, 'name': 'cube_brown_large_metal'},
                                        {'id': 246, 'name': 'cube_brown_large_rubber'},
                                        {'id': 247, 'name': 'cube_green_small_metal'},
                                        {'id': 248, 'name': 'cube_green_small_rubber'},
                                        {'id': 249, 'name': 'cube_green_medium_metal'},
                                        {'id': 250, 'name': 'cube_green_medium_rubber'},
                                        {'id': 251, 'name': 'cube_green_large_metal'},
                                        {'id': 252, 'name': 'cube_green_large_rubber'},
                                        {'id': 253, 'name': 'cube_cyan_small_metal'},
                                        {'id': 254, 'name': 'cube_cyan_small_rubber'},
                                        {'id': 255, 'name': 'cube_cyan_medium_metal'},
                                        {'id': 256, 'name': 'cube_cyan_medium_rubber'},
                                        {'id': 257, 'name': 'cube_cyan_large_metal'},
                                        {'id': 258, 'name': 'cube_cyan_large_rubber'},
                                        {'id': 259, 'name': 'cube_purple_small_metal'},
                                        {'id': 260, 'name': 'cube_purple_small_rubber'},
                                        {'id': 261, 'name': 'cube_purple_medium_metal'},
                                        {'id': 262, 'name': 'cube_purple_medium_rubber'},
                                        {'id': 263, 'name': 'cube_purple_large_metal'},
                                        {'id': 264, 'name': 'cube_purple_large_rubber'},
                                        {'id': 265, 'name': 'cube_gray_small_metal'},
                                        {'id': 266, 'name': 'cube_gray_small_rubber'},
                                        {'id': 267, 'name': 'cube_gray_medium_metal'},
                                        {'id': 268, 'name': 'cube_gray_medium_rubber'},
                                        {'id': 269, 'name': 'cube_gray_large_metal'},
                                        {'id': 270, 'name': 'cube_gray_large_rubber'} ]                 
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
        annotation['segementation'] = label['segementation']
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

    def add_image_with_annotation(self, zeng: dict):

        image = self.image(zeng)
        self.instance["images"].append(image)

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
    
    # 此处添加循环，每导出一张图片，执行一次该命令。
    zeng =  {
                'file_name': "image/frame_000217.PNG",
                'labels': [ 
                            {'segementation': [100,25,30],
                            'area': 100,
                            'bbox': [10,20,20,10],
                            'shape': 'cone',
                            'color': 'gold', 
                            'size': 'small',
                            'material': 'metal',
                            'coordination_X': 1,    #坐标需要是int类型而不是str
                            'coordination_Y': 1,
                            'coordination_Z': 1},

                            {'segementation': [120,45,36,48,99],
                            'area': 111,
                            'bbox': [15,20,25,5],
                            'shape': 'spl',
                            'color': 'purple', 
                            'size': 'large',
                            'material': 'metal',
                            'coordination_X': 0,    #坐标需要是int类型而不是str
                            'coordination_Y': -2,
                            'coordination_Z': 0,
                            }]
            }
    coco.add_image_with_annotation(zeng)
    '''
    对卓哥 Mask-RCNN 导出文件的格式要求：导出必须为字典dict格式,如:  
    zeng =  {
                'file_name': "image/frame_000217.PNG"
                'labels':  [ 
                            {'segementation': [100,25,30],
                            'area': 100,
                            'bbox': [10,20,20,10],
                            'shape': 'cone',
                            'color': 'gold', 
                            'size': 'small',
                            'material': 'metal',
                            'coordination_X': 1,    坐标需要是int类型而不是str
                            'coordination_Y': 1,
                            'coordination_Z': 1},

                            {'segementation': [120,45,36,48,99],
                            'area': 111,
                            'bbox': [15,20,25,5],
                            'shape': 'spl',
                            'color': 'purple', 
                            'size': 'large',
                            'material': 'metal',
                            'coordination_X': 0,    坐标需要是int类型而不是str
                            'coordination_Y': -2,
                            'coordination_Z': 0,
                            }]
            }
    '''

    # 保存Coco文件，最后只执行一次即可
    output = './03_Coco/instances_default.json'
    coco.save(output)