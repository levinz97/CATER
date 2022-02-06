from step4_show_Bbox import Twins, Bbox
import json

class Decision:

    def __init__(self, file: int, frame: str, bbox_list: list):

        self.file = file
        self.frame = frame
        self.bbox_list = bbox_list

        self.decisions = []  # [{'id':123, 'locations': [1,1,1]}]

    def open_file(self):
        twins_in_json_path = r'D:/文档/12_Vorlesungen_Stuttgart/21WS_Practical_Course_ML_CV_for_HCI/TWIN/twins_in_json.json'
        with open(twins_in_json_path, 'r', encoding = 'UTF-8') as twins_in_json_file:
            twins_in_json = json.load(twins_in_json_file)
            twins_in_json_file.close()
        self.list = twins_in_json['CATER_new_00{}'.format(str(file))]
    
    def get_decision(self):
        
        for bbox in self.bbox_list:
            for dic in self.list:
                try:
                    label = bbox['label']
                    locations = dic[label][str(int(self.frame))]
                    print(2*'------')
                    print(label+'   '+str(locations))
                    object_id = int(input('这个坐标属于哪个物体？'))
                    if object_id:
                        self.decisions.append({'id':object_id, 'label': label, 'locations': locations})
                    else:
                        pass
                except KeyError:
                    pass
    def main(self):
        self.open_file()
        self.get_decision()

class Change_location:

    def __init__(self, file: int, frame: str, decisions: list):

        self.file = file
        self.frame = frame
        self.decisions = decisions
        self.opencv_path = r'D://文档//12_Vorlesungen_Stuttgart//21WS_Practical_Course_ML_CV_for_HCI//TWIN/opencv_data/CATER_new_00{}.json'.format(str(file))
 
    def open_file(self):
        with open(self.opencv_path, 'r', encoding = 'UTF-8') as opencv_file:
            self.opencv = json.load(opencv_file)
            opencv_file.close()

    def change_location(self, object_id: int, locations: int):
        annotations_length = len(self.opencv['annotations'])
        
        for i in range(annotations_length):
            if self.opencv['annotations'][i]['id'] == object_id:
                self.opencv['annotations'][i]['attributes']['coordination_X'] = locations[0]
                self.opencv['annotations'][i]['attributes']['coordination_Y'] = locations[1]
                self.opencv['annotations'][i]['attributes']['coordination_Z'] = locations[2]

    def change_all_locations(self):
        
        for decision in self.decisions:
            object_id = decision['id']
            locations = decision['locations']
            self.change_location(object_id, locations) 
            print(object_id)

    def save_file(self):
        output = json.dumps(self.opencv)
        with open(self.opencv_path, 'w', encoding = 'UTF-8') as opencv_file:
            opencv_file.write(output)
            opencv_file.close()
    
    def main(self):
        self.open_file()
        self.change_all_locations()
        self.save_file()

if __name__ == "__main__":
    
    file = 5488
    for frame in ['300','250','200','150','100','050','000']:
        
        print('')
        print('++++++++++++++++++++++++++++ '+str(file)+'_'+frame+' ++++++++++++++++++++++++++++')
        print('')
        t = Twins(file, frame)
        t.get_twins_bbox()

        d = Decision(file, frame, t.bbox_list)
        d.main()

        cl = Change_location(file, frame, d.decisions)
        cl.main()