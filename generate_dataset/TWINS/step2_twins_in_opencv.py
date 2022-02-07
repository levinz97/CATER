import json

class Twins_in_opencv:

    def __init__(self, input_path:str):

        with open(input_path, 'r', encoding = 'UTF-8') as cvat_file:
            self.opencv = json.load(cvat_file)
            cvat_file.close()

        self.filename = self.opencv['images'][0]['file_name'][0:16]+'.png'
        self.twins = {}
    
    def get_image_id(self, frame_order: int):
        
        image_id = self.opencv['images'][frame_order]['id']
        return image_id
    
    def get_label(self, dictionary: dict, num: int):
        shape = dictionary['annotations'][num]['attributes']['shape']
        color = dictionary['annotations'][num]['attributes']['color']
        material = dictionary['annotations'][num]['attributes']['material']            
        size = dictionary['annotations'][num]['attributes']['size']
        label = shape+'_'+color+'_'+size+'_'+material
        return label

    def add_twin(self, image_id: int, label: str):
        twins_image_id = int(image_id)-1
        twins_file_name = self.opencv['images'][twins_image_id]['file_name']
        twins_label = label
        
        if twins_label in self.twins:
            self.twins[twins_label].append(twins_file_name)
        else:
            self.twins[twins_label] = []
            self.twins[twins_label].append(twins_file_name)

    def check_twins(self):

        for frame_order in range(7):
            
            image_id = self.get_image_id(frame_order)
            label_list = []

            for antt in self.opencv['annotations']:
                num = self.opencv['annotations'].index(antt)
                
                if antt['image_id'] == image_id:
                    label = self.get_label(self.opencv, num)
                    
                    if label in label_list:
                        self.add_twin(image_id, label)
                    else:
                        label_list.append(label)
                else:
                    pass


if __name__ == "__main__":

    twins_dict = {}
    
    start = 5200
    end = 5499
    black_list = [5251,5258,5300,5375,5393,5398,5427,5457]
    
    for i in range(start,int(end+1)):
        
        if i in black_list:
            pass

        else:
            input_path = './opencv_data/CATER_new_00{}.json'.format(i)
            tio = Twins_in_opencv(input_path)
            tio.check_twins()

            if tio.twins:
                twins_dict['CATER_new_00{}'.format(i)] = tio.twins
                
    output = json.dumps(twins_dict)
    with open('./twins_in_opencv.json', 'w', encoding = 'UTF-8') as output_file:
        output_file.write(output)
        output_file.close()
    print('success!!!')