import json

class Dataset:
    
    def __init__(self, all_objects_list: list, object_id: int, frame: int):

        self.all_objects_list = all_objects_list
        self.object_id = object_id
        self.frame = frame

        self.attributes_list = ['instance', 
                  'shape', 
                  'color', 
                  'size', 
                  'sized',                               
                  'material']
        self.attributes_value_dict = {}
        self.locations_value_dict = {}

    def get_attributes(self):
        for attribute in self.attributes_list:
            self.attributes_value_dict[attribute] = self.all_objects_list[self.object_id][attribute]

    def get_locations(self):
        locations = self.all_objects_list[self.object_id]['locations'][str(self.frame)]
        self.locations_value_dict['coordination_X'] = locations[0]
        self.locations_value_dict['coordination_Y'] = locations[1]
        self.locations_value_dict['coordination_Z'] = locations[2]

class CVAT:

    def __init__(self, attributes_value_dict: dict, locations_value_dict: dict, object_id: int):

        self.attributes_value_dict = attributes_value_dict
        self.locations_value_dict = locations_value_dict
        self.object_id = object_id
        
        self.cvat_object_label = {}
        self.label_color = ['#fa3253',
                            '#f59331',
                            '#fafa37',
                            '#83e070',
                            '#2a7dd1',
                            '#33ddff',
                            '#b83df5',
                            '#cc9933',
                            '#aaf0d1',
                            '#fa7dbb',
                            '#ff6a4d',
                            '#34d1b7']

    def initialization(self):
        shape = self.attributes_value_dict['shape']
        color = self.attributes_value_dict['color']
        self.cvat_object_label['name'] = str(shape+'_'+color)
        self.cvat_object_label['color'] = self.label_color[self.object_id]
        self.cvat_object_label['attributes'] = []

    def set_attributes(self):
        for attribute in self.attributes_value_dict:
            attribute_dict = {'name': attribute, 
                              'input_type': 'text', 
                              'mutable': False,
                              'values':[str(self.attributes_value_dict[attribute])]}
            self.cvat_object_label['attributes'].append(attribute_dict)
    
    def set_locations(self):
        for location in self.locations_value_dict:
            location_dict = {'name': location, 
                             'input_type': 'text', 
                             'mutable': False,
                             'values':[str(self.locations_value_dict[location])]}
            self.cvat_object_label['attributes'].append(location_dict)

class Label:

    def __init__(self, all_objects_list: list, object_id: int, frame: int):

        self.all_objects_list = all_objects_list
        self.object_id = object_id
        self.frame = frame

        self.attributes_value_dict = {}
        self.locations_value_dict = {}
        self.cvat_object_label = {}

    def download_dataset(self):
        dataset = Dataset(self.all_objects_list, self.object_id, self.frame)
        dataset.get_attributes()
        self.attributes_value_dict = dataset.attributes_value_dict
        dataset.get_locations()
        self.locations_value_dict = dataset.locations_value_dict

    def upload_CVAT(self):
        cvat = CVAT(self.attributes_value_dict, self.locations_value_dict, self.object_id)
        cvat.initialization()
        cvat.set_attributes()
        cvat.set_locations()
        self.cvat_object_label = cvat.cvat_object_label


if __name__=='__main__':
    cvat_array = []
    input_path = './json_input/CATER_new_005780.json'
    output_path = './json_output/CATER_new_005780.json'
    frame = 0

    with open(input_path, 'r', encoding = 'UTF-8') as input_file:
        dictionary = json.load(input_file)
        input_file.close()
    all_objects_list = dictionary['objects']

    object_amount = len(all_objects_list)
    for object_id in range(object_amount):
        label = Label(all_objects_list,object_id,frame)
        label.download_dataset()
        label.upload_CVAT()
        cvat_array.append(label.cvat_object_label)

    print(cvat_array)
    cvat_json = json.dumps(cvat_array)
    with open(output_path, 'w', encoding = 'UTF-8') as output_file:
        output_file.write(cvat_json)
        output_file.close()