import json

categories = [{'id': 1, 'name': 'spl_gold_small_metal', 'supercategory': ''}, {'id': 2, 'name': 'cone_blue_large_metal', 'supercategory': ''}, {'id': 3, 'name': 'cone_blue_large_rubber', 'supercategory': ''}, {'id': 4, 'name': 'cone_blue_small_metal', 'supercategory': ''}, {'id': 5, 'name': 'cone_blue_small_rubber', 'supercategory': 
''}, {'id': 6, 'name': 'cone_blue_medium_metal', 'supercategory': ''}, {'id': 7, 'name': 'cone_blue_medium_rubber', 'supercategory': ''}, {'id': 8, 'name': 'cone_yellow_large_metal', 'supercategory': ''}, {'id': 9, 'name': 'cone_yellow_large_rubber', 'supercategory': ''}, {'id': 10, 'name': 'cone_yellow_small_metal', 'supercategory': ''}, {'id': 11, 'name': 'cone_yellow_small_rubber', 'supercategory': ''}, {'id': 12, 'name': 'cone_yellow_medium_metal', 'supercategory': ''}, {'id': 13, 'name': 'cone_yellow_medium_rubber', 'supercategory': ''}, {'id': 14, 'name': 'cone_green_large_metal', 'supercategory': ''}, {'id': 15, 'name': 'cone_green_large_rubber', 'supercategory': ''}, {'id': 16, 'name': 'cone_green_small_metal', 'supercategory': ''}, {'id': 17, 'name': 'cone_green_small_rubber', 'supercategory': ''}, {'id': 18, 'name': 'cone_green_medium_metal', 'supercategory': ''}, {'id': 19, 'name': 'cone_green_medium_rubber', 'supercategory': ''}, {'id': 20, 'name': 'cone_red_large_metal', 'supercategory': ''}, {'id': 21, 'name': 'cone_red_large_rubber', 'supercategory': ''}, {'id': 22, 'name': 'cone_red_small_metal', 'supercategory': ''}, {'id': 23, 'name': 'cone_red_small_rubber', 'supercategory': ''}, {'id': 24, 'name': 'cone_red_medium_metal', 'supercategory': ''}, {'id': 25, 'name': 'cone_red_medium_rubber', 'supercategory': ''}, {'id': 26, 'name': 'cone_brown_large_metal', 'supercategory': ''}, {'id': 27, 'name': 'cone_brown_large_rubber', 'supercategory': ''}, {'id': 28, 'name': 'cone_brown_small_metal', 'supercategory': ''}, {'id': 29, 'name': 'cone_brown_small_rubber', 'supercategory': ''}, {'id': 30, 'name': 'cone_brown_medium_metal', 'supercategory': ''}, {'id': 31, 'name': 'cone_brown_medium_rubber', 'supercategory': ''}, {'id': 32, 'name': 'cone_purple_large_metal', 'supercategory': ''}, {'id': 33, 'name': 'cone_purple_large_rubber', 'supercategory': ''}, {'id': 34, 'name': 'cone_purple_small_metal', 'supercategory': ''}, {'id': 35, 'name': 'cone_purple_small_rubber', 'supercategory': ''}, {'id': 36, 'name': 'cone_purple_medium_metal', 'supercategory': ''}, {'id': 37, 'name': 'cone_purple_medium_rubber', 'supercategory': ''}, {'id': 38, 'name': 'cone_cyan_large_metal', 'supercategory': ''}, {'id': 39, 'name': 'cone_cyan_large_rubber', 'supercategory': ''}, {'id': 40, 'name': 'cone_cyan_small_metal', 'supercategory': ''}, {'id': 41, 'name': 'cone_cyan_small_rubber', 
'supercategory': ''}, {'id': 42, 'name': 'cone_cyan_medium_metal', 'supercategory': ''}, {'id': 43, 'name': 'cone_cyan_medium_rubber', 'supercategory': ''}, {'id': 44, 'name': 'cone_gray_large_metal', 'supercategory': ''}, {'id': 45, 'name': 'cone_gray_large_rubber', 'supercategory': ''}, {'id': 46, 'name': 'cone_gray_small_metal', 'supercategory': ''}, {'id': 47, 'name': 'cone_gray_small_rubber', 'supercategory': ''}, {'id': 48, 'name': 'cone_gray_medium_metal', 'supercategory': ''}, {'id': 49, 'name': 'cone_gray_medium_rubber', 'supercategory': ''}, {'id': 50, 'name': 'cube_blue_large_metal', 'supercategory': ''}, {'id': 51, 'name': 'cube_blue_large_rubber', 'supercategory': ''}, {'id': 52, 'name': 'cube_blue_small_metal', 'supercategory': ''}, {'id': 53, 'name': 'cube_blue_small_rubber', 'supercategory': ''}, {'id': 54, 'name': 'cube_blue_medium_metal', 'supercategory': ''}, {'id': 55, 'name': 'cube_blue_medium_rubber', 'supercategory': ''}, {'id': 
56, 'name': 'cube_yellow_large_metal', 'supercategory': ''}, {'id': 57, 'name': 'cube_yellow_large_rubber', 'supercategory': ''}, {'id': 58, 'name': 'cube_yellow_small_metal', 'supercategory': ''}, {'id': 59, 'name': 'cube_yellow_small_rubber', 'supercategory': ''}, {'id': 60, 'name': 'cube_yellow_medium_metal', 'supercategory': ''}, {'id': 61, 'name': 'cube_yellow_medium_rubber', 'supercategory': ''}, {'id': 62, 'name': 'cube_green_large_metal', 'supercategory': ''}, {'id': 63, 'name': 'cube_green_large_rubber', 'supercategory': ''}, {'id': 64, 'name': 'cube_green_small_metal', 'supercategory': ''}, {'id': 65, 'name': 'cube_green_small_rubber', 'supercategory': ''}, {'id': 66, 'name': 'cube_green_medium_metal', 'supercategory': ''}, {'id': 67, 'name': 'cube_green_medium_rubber', 'supercategory': ''}, {'id': 68, 'name': 'cube_red_large_metal', 'supercategory': ''}, {'id': 69, 'name': 'cube_red_large_rubber', 'supercategory': ''}, {'id': 70, 'name': 'cube_red_small_metal', 'supercategory': ''}, {'id': 71, 'name': 'cube_red_small_rubber', 'supercategory': ''}, {'id': 72, 'name': 'cube_red_medium_metal', 'supercategory': ''}, {'id': 73, 'name': 'cube_red_medium_rubber', 'supercategory': ''}, {'id': 74, 'name': 'cube_brown_large_metal', 'supercategory': ''}, {'id': 75, 'name': 'cube_brown_large_rubber', 'supercategory': ''}, {'id': 76, 'name': 'cube_brown_small_metal', 'supercategory': ''}, {'id': 77, 'name': 'cube_brown_small_rubber', 'supercategory': ''}, {'id': 78, 'name': 'cube_brown_medium_metal', 'supercategory': ''}, {'id': 79, 'name': 'cube_brown_medium_rubber', 'supercategory': 
''}, {'id': 80, 'name': 'cube_purple_large_metal', 'supercategory': ''}, {'id': 81, 'name': 'cube_purple_large_rubber', 'supercategory': ''}, {'id': 82, 'name': 
'cube_purple_small_metal', 'supercategory': ''}, {'id': 83, 'name': 'cube_purple_small_rubber', 'supercategory': ''}, {'id': 84, 'name': 'cube_purple_medium_metal', 'supercategory': ''}, {'id': 85, 'name': 'cube_purple_medium_rubber', 'supercategory': ''}, {'id': 86, 'name': 'cube_cyan_large_metal', 'supercategory': ''}, {'id': 87, 'name': 'cube_cyan_large_rubber', 'supercategory': ''}, {'id': 88, 'name': 'cube_cyan_small_metal', 'supercategory': ''}, {'id': 89, 'name': 'cube_cyan_small_rubber', 'supercategory': ''}, {'id': 90, 'name': 'cube_cyan_medium_metal', 'supercategory': ''}, {'id': 91, 'name': 'cube_cyan_medium_rubber', 'supercategory': ''}, {'id': 92, 'name': 'cube_gray_large_metal', 'supercategory': ''}, {'id': 93, 'name': 'cube_gray_large_rubber', 'supercategory': ''}, {'id': 94, 'name': 'cube_gray_small_metal', 'supercategory': ''}, {'id': 95, 'name': 'cube_gray_small_rubber', 'supercategory': ''}, {'id': 96, 'name': 'cube_gray_medium_metal', 'supercategory': ''}, {'id': 97, 'name': 'cube_gray_medium_rubber', 'supercategory': ''}, {'id': 98, 'name': 'sphere_blue_large_metal', 'supercategory': ''}, 
{'id': 99, 'name': 'sphere_blue_large_rubber', 'supercategory': ''}, {'id': 100, 'name': 'sphere_blue_small_metal', 'supercategory': ''}, {'id': 101, 'name': 'sphere_blue_small_rubber', 'supercategory': ''}, {'id': 102, 'name': 'sphere_blue_medium_metal', 'supercategory': ''}, {'id': 103, 'name': 'sphere_blue_medium_rubber', 'supercategory': ''}, {'id': 104, 'name': 'sphere_yellow_large_metal', 'supercategory': ''}, {'id': 105, 'name': 'sphere_yellow_large_rubber', 'supercategory': ''}, {'id': 106, 'name': 'sphere_yellow_small_metal', 'supercategory': ''}, {'id': 107, 'name': 'sphere_yellow_small_rubber', 'supercategory': ''}, {'id': 108, 'name': 'sphere_yellow_medium_metal', 'supercategory': ''}, {'id': 109, 'name': 'sphere_yellow_medium_rubber', 'supercategory': ''}, {'id': 110, 'name': 'sphere_green_large_metal', 'supercategory': ''}, {'id': 111, 'name': 'sphere_green_large_rubber', 'supercategory': ''}, {'id': 112, 'name': 'sphere_green_small_metal', 'supercategory': ''}, {'id': 113, 'name': 'sphere_green_small_rubber', 'supercategory': ''}, {'id': 114, 'name': 'sphere_green_medium_metal', 'supercategory': ''}, {'id': 115, 'name': 'sphere_green_medium_rubber', 'supercategory': ''}, {'id': 116, 'name': 'sphere_red_large_metal', 'supercategory': ''}, {'id': 117, 'name': 'sphere_red_large_rubber', 'supercategory': ''}, {'id': 118, 'name': 'sphere_red_small_metal', 'supercategory': ''}, {'id': 119, 'name': 'sphere_red_small_rubber', 'supercategory': ''}, {'id': 120, 'name': 'sphere_red_medium_metal', 'supercategory': ''}, {'id': 121, 'name': 'sphere_red_medium_rubber', 'supercategory': ''}, {'id': 122, 'name': 'sphere_brown_large_metal', 'supercategory': ''}, {'id': 123, 'name': 'sphere_brown_large_rubber', 'supercategory': ''}, {'id': 124, 
'name': 'sphere_brown_small_metal', 'supercategory': ''}, {'id': 125, 'name': 'sphere_brown_small_rubber', 'supercategory': ''}, {'id': 126, 'name': 'sphere_brown_medium_metal', 'supercategory': ''}, {'id': 127, 'name': 'sphere_brown_medium_rubber', 'supercategory': ''}, {'id': 128, 'name': 'sphere_purple_large_metal', 'supercategory': ''}, {'id': 129, 'name': 'sphere_purple_large_rubber', 'supercategory': ''}, {'id': 130, 'name': 'sphere_purple_small_metal', 'supercategory': ''}, {'id': 131, 'name': 'sphere_purple_small_rubber', 'supercategory': ''}, {'id': 132, 'name': 'sphere_purple_medium_metal', 'supercategory': ''}, {'id': 133, 'name': 'sphere_purple_medium_rubber', 'supercategory': ''}, {'id': 134, 'name': 'sphere_cyan_large_metal', 'supercategory': ''}, {'id': 135, 'name': 'sphere_cyan_large_rubber', 'supercategory': ''}, {'id': 136, 'name': 'sphere_cyan_small_metal', 'supercategory': ''}, {'id': 137, 'name': 'sphere_cyan_small_rubber', 'supercategory': ''}, {'id': 138, 'name': 'sphere_cyan_medium_metal', 'supercategory': ''}, {'id': 139, 'name': 'sphere_cyan_medium_rubber', 'supercategory': ''}, {'id': 140, 'name': 'sphere_gray_large_metal', 'supercategory': ''}, {'id': 141, 'name': 'sphere_gray_large_rubber', 'supercategory': ''}, {'id': 142, 'name': 'sphere_gray_small_metal', 'supercategory': ''}, {'id': 143, 'name': 'sphere_gray_small_rubber', 'supercategory': ''}, {'id': 144, 'name': 'sphere_gray_medium_metal', 'supercategory': ''}, {'id': 145, 'name': 'sphere_gray_medium_rubber', 'supercategory': ''}, {'id': 146, 'name': 'cylinder_blue_large_metal', 'supercategory': ''}, {'id': 147, 'name': 'cylinder_blue_large_rubber', 'supercategory': ''}, {'id': 148, 'name': 'cylinder_blue_small_metal', 'supercategory': ''}, {'id': 149, 'name': 'cylinder_blue_small_rubber', 'supercategory': ''}, {'id': 150, 'name': 'cylinder_blue_medium_metal', 'supercategory': ''}, {'id': 151, 'name': 'cylinder_blue_medium_rubber', 'supercategory': ''}, {'id': 152, 'name': 'cylinder_yellow_large_metal', 'supercategory': ''}, {'id': 153, 'name': 'cylinder_yellow_large_rubber', 'supercategory': ''}, {'id': 154, 'name': 'cylinder_yellow_small_metal', 'supercategory': ''}, {'id': 155, 'name': 'cylinder_yellow_small_rubber', 'supercategory': ''}, {'id': 156, 'name': 'cylinder_yellow_medium_metal', 'supercategory': ''}, {'id': 157, 'name': 'cylinder_yellow_medium_rubber', 'supercategory': ''}, 
{'id': 158, 'name': 'cylinder_green_large_metal', 'supercategory': ''}, {'id': 159, 'name': 'cylinder_green_large_rubber', 'supercategory': ''}, {'id': 160, 'name': 'cylinder_green_small_metal', 'supercategory': ''}, {'id': 161, 'name': 'cylinder_green_small_rubber', 'supercategory': ''}, {'id': 162, 'name': 'cylinder_green_medium_metal', 'supercategory': ''}, {'id': 163, 'name': 'cylinder_green_medium_rubber', 'supercategory': ''}, {'id': 164, 'name': 'cylinder_red_large_metal', 'supercategory': ''}, {'id': 165, 'name': 'cylinder_red_large_rubber', 'supercategory': ''}, {'id': 166, 'name': 'cylinder_red_small_metal', 'supercategory': ''}, {'id': 167, 'name': 'cylinder_red_small_rubber', 'supercategory': ''}, {'id': 168, 'name': 'cylinder_red_medium_metal', 'supercategory': ''}, {'id': 169, 'name': 'cylinder_red_medium_rubber', 'supercategory': ''}, {'id': 170, 'name': 'cylinder_brown_large_metal', 'supercategory': ''}, {'id': 171, 'name': 'cylinder_brown_large_rubber', 'supercategory': ''}, {'id': 172, 'name': 'cylinder_brown_small_metal', 'supercategory': ''}, {'id': 173, 'name': 'cylinder_brown_small_rubber', 'supercategory': ''}, {'id': 174, 'name': 'cylinder_brown_medium_metal', 'supercategory': ''}, {'id': 175, 'name': 'cylinder_brown_medium_rubber', 'supercategory': ''}, {'id': 176, 'name': 'cylinder_purple_large_metal', 'supercategory': ''}, {'id': 177, 'name': 'cylinder_purple_large_rubber', 'supercategory': ''}, {'id': 178, 'name': 'cylinder_purple_small_metal', 'supercategory': ''}, {'id': 179, 'name': 'cylinder_purple_small_rubber', 'supercategory': ''}, {'id': 180, 'name': 'cylinder_purple_medium_metal', 'supercategory': ''}, {'id': 181, 'name': 'cylinder_purple_medium_rubber', 'supercategory': ''}, {'id': 182, 'name': 'cylinder_cyan_large_metal', 'supercategory': ''}, {'id': 183, 'name': 'cylinder_cyan_large_rubber', 'supercategory': ''}, {'id': 184, 'name': 'cylinder_cyan_small_metal', 'supercategory': ''}, {'id': 185, 'name': 'cylinder_cyan_small_rubber', 'supercategory': ''}, {'id': 186, 'name': 'cylinder_cyan_medium_metal', 'supercategory': ''}, {'id': 187, 'name': 'cylinder_cyan_medium_rubber', 'supercategory': ''}, {'id': 188, 'name': 'cylinder_gray_large_metal', 'supercategory': ''}, {'id': 189, 'name': 'cylinder_gray_large_rubber', 'supercategory': ''}, {'id': 190, 'name': 'cylinder_gray_small_metal', 'supercategory': ''}, {'id': 191, 'name': 'cylinder_gray_small_rubber', 'supercategory': ''}, {'id': 192, 'name': 'cylinder_gray_medium_metal', 'supercategory': ''}, {'id': 193, 'name': 'cylinder_gray_medium_rubber', 'supercategory': ''}]

class Relabel_opencv:

    def __init__(self, input_path: str, output_path: str):

        self.input_path = input_path
        self.output_path = output_path

        with open(input_path, 'r', encoding = 'UTF-8') as cvat_file:
            self.opencv = json.load(cvat_file)
            cvat_file.close()

        self.opencv['categories'] = categories

    def reorder(self):
        for annotation in self.opencv['annotations']:
            annotation['id'] = self.opencv['annotations'].index(annotation)+1   

    def relabel(self):
        
        annotations_length = len(self.opencv['annotations'])
        
        for annotation_num in range(annotations_length):

            shape = self.opencv['annotations'][annotation_num]['attributes']['shape']
            color = self.opencv['annotations'][annotation_num]['attributes']['color']
            size = self.opencv['annotations'][annotation_num]['attributes']['size']
            material = self.opencv['annotations'][annotation_num]['attributes']['material']

            new_label = shape+'_'+color+'_'+size+'_'+material

            for category in self.opencv['categories']: 
                if new_label == category['name']:
                    category_id = category['id']
                    self.opencv['annotations'][annotation_num]['category_id'] = category_id
                else:
                    pass
    
    def save(self):
       
        output = json.dumps(self.opencv)
        output_file = open(self.output_path, 'w', encoding = 'UTF-8')
        output_file.write(output)
        output_file.close()


class Addlabel_json:

    def __init__(self, input_path: str, output_path: str):

        self.input_path = input_path
        self.output_path = output_path

        with open(self.input_path, 'r', encoding = 'UTF-8') as cvat_file:
            self.origin = json.load(cvat_file)
            cvat_file.close()

    def add_label(self):
        for obj in self.origin['objects']:
            obj_num = self.origin['objects'].index(obj)

            shape = obj['shape']
            color = obj['color']
            size = obj['size']
            material = obj['material']

            new_label = shape+'_'+color+'_'+size+'_'+material

            for category in categories: 
                if new_label == category['name']:         
                    self.origin['objects'][obj_num]['category_id'] = category['id']
                    self.origin['objects'][obj_num]['category'] = category['name']
                else:
                    pass
                
    def save(self):
       
        output = json.dumps(self.origin)
        output_file = open(self.output_path, 'w', encoding = 'UTF-8')
        output_file.write(output)
        output_file.close()    


if __name__ == "__main__":
    
    start = 5200
    end = 5499

    black_list = [5251,5258,5300,5375,5393,5398,5427,5457]
    for i in range(start,int(end+1)):
        
        if i in black_list:
            pass

        else:
            
            opencv_path = './opencv_data/CATER_new_00{}.json'.format(str(i))
            rlo = Relabel_opencv(input_path=opencv_path, output_path=opencv_path)
            rlo.reorder()
            rlo.relabel()
            rlo.save()
            print(2*'>>>>>>>>>'+str(i)+' relabel finish')

            origin_path = './json_data/CATER_new_00{}.json'.format(str(i))
            alj = Addlabel_json(input_path=origin_path, output_path=origin_path)
            alj.add_label()
            alj.save()
            print(2*'<<<<<<<<<'+str(i)+' addlabel finish')

    print('success!!!')