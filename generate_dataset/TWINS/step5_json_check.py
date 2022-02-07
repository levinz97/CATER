import json

class Data:

    def __init__(self, input_path:str, json_path:str, output_path:str):
        
        self.input_path = input_path
        self.json_path = json_path
        self.output_path= output_path

        with open(self.input_path, 'r', encoding = 'UTF-8') as cvat_file:
            self.opencv = json.load(cvat_file)
            cvat_file.close()
    
    def get_frame_number(self, frame_order: int):
        
        image_id = self.opencv['images'][frame_order]['id']
        file_name = self.opencv['images'][frame_order]['file_name']
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

            for antt in self.opencv['annotations']:

                num = self.opencv['annotations'].index(antt)
                if antt['image_id'] == image_id:
                
                    for obj in objects:
                        if obj['shape'] == self.opencv['annotations'][num]['attributes']['shape']:
                            if obj['color'] == self.opencv['annotations'][num]['attributes']['color']:
                                if obj['material'] == self.opencv['annotations'][num]['attributes']['material']:
                                    if obj['size'] == self.opencv['annotations'][num]['attributes']['size']:
                                        self.opencv['annotations'][num]['attributes']['coordination_X'] = obj['locations'][str(frame_number)][0]
                                        self.opencv['annotations'][num]['attributes']['coordination_Y'] = obj['locations'][str(frame_number)][1]
                                        self.opencv['annotations'][num]['attributes']['coordination_Z'] = obj['locations'][str(frame_number)][2]
                                    else:
                                        continue
                                else:
                                    continue
                            else:
                                continue
                        else:
                            continue

    def save(self):
        
        output = json.dumps(self.opencv)
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
            input_path = output_path = './opencv_data/CATER_new_00{}.json'.format(str(i))
            json_path = './json_data/CATER_new_00{}.json'.format(str(i))
            data = Data(input_path, json_path, output_path)
            data.check()
            data.save()
            print(2*'>>>>>>>>>'+json_path+'check finish')

    print('success!!!')