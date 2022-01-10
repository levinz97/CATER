import json

class Data:

    def __init__(self, input_data_path:str, json_path:str, output_data_path:str):
        
        self.input_data_path = input_data_path
        self.json_path = json_path

        self.output_data_path= output_data_path
        self.filenum = 'CATER_new_005200'
        self.modified_data = {'CATER_new_005200': []}

        with open(self.input_data_path, 'r', encoding = 'UTF-8') as cvat_file:
            self.dictionary = json.load(cvat_file)
            cvat_file.close()

        self.filename = self.dictionary['images'][0]['file_name'][0:16]+'.png'
        self.wrong = []
    
    def get_frame_number(self, frame_order: int):
        
        image_id = self.dictionary['images'][frame_order]['id']
        file_name = self.dictionary['images'][frame_order]['file_name']
        frame_number = int(file_name[-7:-4])

        return image_id, frame_number

    def import_json(self):        
        
        with open(self.json_path, 'r', encoding = 'UTF-8') as json_file:
            json_dict = json.load(json_file)
            json_file.close()
        objects = json_dict['objects']
        
        return objects

    def check(self):

        objects = self.import_json()

        for frame_order in range(7):
            image_id, frame_number = self.get_frame_number(frame_order)

            for antt in self.dictionary['annotations']:

                num = self.dictionary['annotations'].index(antt)
                if antt['image_id'] == image_id:
                
                    for obj in objects:
                        if obj['shape'] == self.dictionary['annotations'][num]['attributes']['shape']:
                            if obj['color'] == self.dictionary['annotations'][num]['attributes']['color']:
                                if obj['material'] == self.dictionary['annotations'][num]['attributes']['material']:
                                    self.dictionary['annotations'][num]['attributes']['size'] = obj['size']
                                    self.dictionary['annotations'][num]['attributes']['coordination_X'] = obj['locations'][str(frame_number)][0]
                                    self.dictionary['annotations'][num]['attributes']['coordination_Y'] = obj['locations'][str(frame_number)][1]
                                    self.dictionary['annotations'][num]['attributes']['coordination_Z'] = obj['locations'][str(frame_number)][2]
                                else:
                                    continue
                            else:
                                continue
                        else:
                            continue
                    
                    if self.dictionary['annotations'][num]['attributes']['coordination_X'] == 0:
                        category_id = self.dictionary['annotations'][num]['category_id']
                        wrong = {}
                        wrong['filename'] = self.filename
                        wrong['frame'] = frame_number
                        wrong['label'] = self.dictionary['categories'][category_id-1]['name']
                        self.wrong.append(wrong)
                    else:
                        pass
    
    def relabel(self):
               
        annotations_length = len(self.dictionary['annotations'])

        for annotation_num in range(annotations_length):

            shape = self.dictionary['annotations'][annotation_num]['attributes']['shape']
            color = self.dictionary['annotations'][annotation_num]['attributes']['color']
            size = self.dictionary['annotations'][annotation_num]['attributes']['size']
            material = self.dictionary['annotations'][annotation_num]['attributes']['material']

            new_label = shape+'_'+color+'_'+size+'_'+material

            for category in self.dictionary['categories']:
                if new_label == category['name']:
                    category_id = category['id']
                    self.dictionary['annotations'][annotation_num]['category_id'] = category_id
                else:
                    pass
    
    def final_relabel(self):

        categories = [
                        {
                            "id": 1,
                            "name": "spl_gold_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 2,
                            "name": "spl_gold_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 3,
                            "name": "spl_gold_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 4,
                            "name": "spl_gold_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 5,
                            "name": "spl_gold_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 6,
                            "name": "spl_gold_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 7,
                            "name": "spl_blue_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 8,
                            "name": "spl_blue_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 9,
                            "name": "spl_blue_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 10,
                            "name": "spl_blue_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 11,
                            "name": "spl_blue_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 12,
                            "name": "spl_blue_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 13,
                            "name": "spl_yellow_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 14,
                            "name": "spl_yellow_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 15,
                            "name": "spl_yellow_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 16,
                            "name": "spl_yellow_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 17,
                            "name": "spl_yellow_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 18,
                            "name": "spl_yellow_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 19,
                            "name": "spl_red_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 20,
                            "name": "spl_red_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 21,
                            "name": "spl_red_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 22,
                            "name": "spl_red_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 23,
                            "name": "spl_red_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 24,
                            "name": "spl_red_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 25,
                            "name": "spl_brown_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 26,
                            "name": "spl_brown_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 27,
                            "name": "spl_brown_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 28,
                            "name": "spl_brown_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 29,
                            "name": "spl_brown_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 30,
                            "name": "spl_brown_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 31,
                            "name": "spl_green_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 32,
                            "name": "spl_green_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 33,
                            "name": "spl_green_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 34,
                            "name": "spl_green_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 35,
                            "name": "spl_green_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 36,
                            "name": "spl_green_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 37,
                            "name": "spl_cyan_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 38,
                            "name": "spl_cyan_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 39,
                            "name": "spl_cyan_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 40,
                            "name": "spl_cyan_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 41,
                            "name": "spl_cyan_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 42,
                            "name": "spl_cyan_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 43,
                            "name": "spl_purple_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 44,
                            "name": "spl_purple_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 45,
                            "name": "spl_purple_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 46,
                            "name": "spl_purple_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 47,
                            "name": "spl_purple_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 48,
                            "name": "spl_purple_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 49,
                            "name": "spl_gray_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 50,
                            "name": "spl_gray_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 51,
                            "name": "spl_gray_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 52,
                            "name": "spl_gray_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 53,
                            "name": "spl_gray_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 54,
                            "name": "spl_gray_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 55,
                            "name": "cone_gold_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 56,
                            "name": "cone_gold_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 57,
                            "name": "cone_gold_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 58,
                            "name": "cone_gold_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 59,
                            "name": "cone_gold_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 60,
                            "name": "cone_gold_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 61,
                            "name": "cone_blue_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 62,
                            "name": "cone_blue_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 63,
                            "name": "cone_blue_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 64,
                            "name": "cone_blue_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 65,
                            "name": "cone_blue_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 66,
                            "name": "cone_blue_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 67,
                            "name": "cone_yellow_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 68,
                            "name": "cone_yellow_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 69,
                            "name": "cone_yellow_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 70,
                            "name": "cone_yellow_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 71,
                            "name": "cone_yellow_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 72,
                            "name": "cone_yellow_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 73,
                            "name": "cone_red_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 74,
                            "name": "cone_red_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 75,
                            "name": "cone_red_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 76,
                            "name": "cone_red_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 77,
                            "name": "cone_red_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 78,
                            "name": "cone_red_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 79,
                            "name": "cone_brown_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 80,
                            "name": "cone_brown_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 81,
                            "name": "cone_brown_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 82,
                            "name": "cone_brown_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 83,
                            "name": "cone_brown_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 84,
                            "name": "cone_brown_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 85,
                            "name": "cone_green_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 86,
                            "name": "cone_green_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 87,
                            "name": "cone_green_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 88,
                            "name": "cone_green_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 89,
                            "name": "cone_green_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 90,
                            "name": "cone_green_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 91,
                            "name": "cone_cyan_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 92,
                            "name": "cone_cyan_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 93,
                            "name": "cone_cyan_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 94,
                            "name": "cone_cyan_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 95,
                            "name": "cone_cyan_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 96,
                            "name": "cone_cyan_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 97,
                            "name": "cone_purple_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 98,
                            "name": "cone_purple_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 99,
                            "name": "cone_purple_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 100,
                            "name": "cone_purple_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 101,
                            "name": "cone_purple_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 102,
                            "name": "cone_purple_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 103,
                            "name": "cone_gray_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 104,
                            "name": "cone_gray_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 105,
                            "name": "cone_gray_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 106,
                            "name": "cone_gray_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 107,
                            "name": "cone_gray_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 108,
                            "name": "cone_gray_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 109,
                            "name": "cube_gold_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 110,
                            "name": "cube_gold_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 111,
                            "name": "cube_gold_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 112,
                            "name": "cube_gold_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 113,
                            "name": "cube_gold_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 114,
                            "name": "cube_gold_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 115,
                            "name": "cube_blue_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 116,
                            "name": "cube_blue_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 117,
                            "name": "cube_blue_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 118,
                            "name": "cube_blue_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 119,
                            "name": "cube_blue_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 120,
                            "name": "cube_blue_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 121,
                            "name": "cube_yellow_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 122,
                            "name": "cube_yellow_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 123,
                            "name": "cube_yellow_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 124,
                            "name": "cube_yellow_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 125,
                            "name": "cube_yellow_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 126,
                            "name": "cube_yellow_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 127,
                            "name": "cube_red_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 128,
                            "name": "cube_red_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 129,
                            "name": "cube_red_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 130,
                            "name": "cube_red_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 131,
                            "name": "cube_red_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 132,
                            "name": "cube_red_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 133,
                            "name": "cube_brown_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 134,
                            "name": "cube_brown_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 135,
                            "name": "cube_brown_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 136,
                            "name": "cube_brown_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 137,
                            "name": "cube_brown_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 138,
                            "name": "cube_brown_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 139,
                            "name": "cube_green_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 140,
                            "name": "cube_green_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 141,
                            "name": "cube_green_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 142,
                            "name": "cube_green_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 143,
                            "name": "cube_green_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 144,
                            "name": "cube_green_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 145,
                            "name": "cube_cyan_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 146,
                            "name": "cube_cyan_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 147,
                            "name": "cube_cyan_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 148,
                            "name": "cube_cyan_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 149,
                            "name": "cube_cyan_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 150,
                            "name": "cube_cyan_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 151,
                            "name": "cube_purple_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 152,
                            "name": "cube_purple_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 153,
                            "name": "cube_purple_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 154,
                            "name": "cube_purple_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 155,
                            "name": "cube_purple_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 156,
                            "name": "cube_purple_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 157,
                            "name": "cube_gray_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 158,
                            "name": "cube_gray_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 159,
                            "name": "cube_gray_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 160,
                            "name": "cube_gray_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 161,
                            "name": "cube_gray_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 162,
                            "name": "cube_gray_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 163,
                            "name": "sphere_gold_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 164,
                            "name": "sphere_gold_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 165,
                            "name": "sphere_gold_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 166,
                            "name": "sphere_gold_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 167,
                            "name": "sphere_gold_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 168,
                            "name": "sphere_gold_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 169,
                            "name": "sphere_blue_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 170,
                            "name": "sphere_blue_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 171,
                            "name": "sphere_blue_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 172,
                            "name": "sphere_blue_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 173,
                            "name": "sphere_blue_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 174,
                            "name": "sphere_blue_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 175,
                            "name": "sphere_yellow_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 176,
                            "name": "sphere_yellow_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 177,
                            "name": "sphere_yellow_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 178,
                            "name": "sphere_yellow_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 179,
                            "name": "sphere_yellow_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 180,
                            "name": "sphere_yellow_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 181,
                            "name": "sphere_red_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 182,
                            "name": "sphere_red_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 183,
                            "name": "sphere_red_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 184,
                            "name": "sphere_red_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 185,
                            "name": "sphere_red_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 186,
                            "name": "sphere_red_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 187,
                            "name": "sphere_brown_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 188,
                            "name": "sphere_brown_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 189,
                            "name": "sphere_brown_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 190,
                            "name": "sphere_brown_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 191,
                            "name": "sphere_brown_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 192,
                            "name": "sphere_brown_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 193,
                            "name": "sphere_green_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 194,
                            "name": "sphere_green_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 195,
                            "name": "sphere_green_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 196,
                            "name": "sphere_green_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 197,
                            "name": "sphere_green_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 198,
                            "name": "sphere_green_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 199,
                            "name": "sphere_cyan_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 200,
                            "name": "sphere_cyan_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 201,
                            "name": "sphere_cyan_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 202,
                            "name": "sphere_cyan_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 203,
                            "name": "sphere_cyan_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 204,
                            "name": "sphere_cyan_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 205,
                            "name": "sphere_purple_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 206,
                            "name": "sphere_purple_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 207,
                            "name": "sphere_purple_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 208,
                            "name": "sphere_purple_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 209,
                            "name": "sphere_purple_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 210,
                            "name": "sphere_purple_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 211,
                            "name": "sphere_gray_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 212,
                            "name": "sphere_gray_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 213,
                            "name": "sphere_gray_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 214,
                            "name": "sphere_gray_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 215,
                            "name": "sphere_gray_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 216,
                            "name": "sphere_gray_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 217,
                            "name": "cylinder_gold_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 218,
                            "name": "cylinder_gold_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 219,
                            "name": "cylinder_gold_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 220,
                            "name": "cylinder_gold_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 221,
                            "name": "cylinder_gold_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 222,
                            "name": "cylinder_gold_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 223,
                            "name": "cylinder_blue_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 224,
                            "name": "cylinder_blue_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 225,
                            "name": "cylinder_blue_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 226,
                            "name": "cylinder_blue_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 227,
                            "name": "cylinder_blue_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 228,
                            "name": "cylinder_blue_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 229,
                            "name": "cylinder_yellow_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 230,
                            "name": "cylinder_yellow_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 231,
                            "name": "cylinder_yellow_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 232,
                            "name": "cylinder_yellow_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 233,
                            "name": "cylinder_yellow_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 234,
                            "name": "cylinder_yellow_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 235,
                            "name": "cylinder_red_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 236,
                            "name": "cylinder_red_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 237,
                            "name": "cylinder_red_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 238,
                            "name": "cylinder_red_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 239,
                            "name": "cylinder_red_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 240,
                            "name": "cylinder_red_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 241,
                            "name": "cylinder_brown_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 242,
                            "name": "cylinder_brown_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 243,
                            "name": "cylinder_brown_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 244,
                            "name": "cylinder_brown_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 245,
                            "name": "cylinder_brown_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 246,
                            "name": "cylinder_brown_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 247,
                            "name": "cylinder_green_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 248,
                            "name": "cylinder_green_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 249,
                            "name": "cylinder_green_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 250,
                            "name": "cylinder_green_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 251,
                            "name": "cylinder_green_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 252,
                            "name": "cylinder_green_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 253,
                            "name": "cylinder_cyan_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 254,
                            "name": "cylinder_cyan_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 255,
                            "name": "cylinder_cyan_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 256,
                            "name": "cylinder_cyan_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 257,
                            "name": "cylinder_cyan_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 258,
                            "name": "cylinder_cyan_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 259,
                            "name": "cylinder_purple_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 260,
                            "name": "cylinder_purple_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 261,
                            "name": "cylinder_purple_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 262,
                            "name": "cylinder_purple_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 263,
                            "name": "cylinder_purple_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 264,
                            "name": "cylinder_purple_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 265,
                            "name": "cylinder_gray_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 266,
                            "name": "cylinder_gray_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 267,
                            "name": "cylinder_gray_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 268,
                            "name": "cylinder_gray_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 269,
                            "name": "cylinder_gray_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 270,
                            "name": "cylinder_gray_large_metal",
                            "supercategory": ""
                        }
                    ]

        annotations_length = len(self.dictionary['annotations'])

        for annotation_num in range(annotations_length):

            shape = self.dictionary['annotations'][annotation_num]['attributes']['shape']
            color = self.dictionary['annotations'][annotation_num]['attributes']['color']
            size = self.dictionary['annotations'][annotation_num]['attributes']['size']
            material = self.dictionary['annotations'][annotation_num]['attributes']['material']

            new_label = shape+'_'+color+'_'+size+'_'+material

            for category in categories:
                if new_label == category['name']:
                    category_id = category['id']
                    self.dictionary['annotations'][annotation_num]['category_id'] = category_id
                else:
                    pass
        
    def save(self):
        
        output = json.dumps(self.dictionary)
        output_file = open(self.output_data_path, 'w', encoding = 'UTF-8')
        output_file.write(output)
        output_file.close()

if __name__ == "__main__":
    
    wrong_dict = {}

    for i in range(5259,5286):
        
        input_data_path = './a_data/CATER_new_00{}.json'.format(str(i))
        json_path = './json/CATER_new_00{}.json'.format(str(i))
        output_data_path= './b_data/CATER_new_00{}.json'.format(str(i))
        
        data = Data(input_data_path, json_path, output_data_path)
        data.check()
        data.relabel()
        # print(data.dictionary)
        data.save()

        wrong_dict[str(i)] = data.wrong

    with open ('.wrong.json','w', encoding = 'UTF-8') as wrong_doc:
        wrong_dict = json.dumps(wrong_dict)
        wrong_doc.write(wrong_dict)
        wrong_doc.close()

    print('success!!!')


'''
with open(input_path, 'r', encoding = 'UTF-8') as input_file:
    print(input_path)
    dictionary = json.load(input_file)
    input_file.close()

with open(output_data_path, 'w', encoding = 'UTF-8') as output_file:
                output_file.write(cvat_json)
                output_file.close()

file = open(output, 'w', encoding = 'UTF-8')
    file.write(instances)
    file.close()


try: 
except FileNotFoundError:
            pass
'''