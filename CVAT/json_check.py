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