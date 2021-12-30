import json

class Dataset:

    def __init__(self, shape_values: list, color_values: list, size_values: list, material_values: list, label_color:list):
        self.shape_values = shape_values
        self.color_values = color_values
        self.size_values = size_values
        self.material_values = material_values
        self.label_color =label_color
        self.object_number = 0
        self.name_list = []
        self.attributes_value_dict = {}
        self.attributes_list = ['shape','color','size','material']
        for attribute in self.attributes_list:
            self.attributes_value_dict[attribute] = []

    def get_attributes(self):
        for shape in self.shape_values:
            for color in self.color_values:
                for size in self.size_values:
                    for material in self.material_values:
                        name = shape + '_' + color + '_' + size + '_' + material
                        self.name_list.append(name)
                        self.attributes_value_dict['shape'].append(shape)
                        self.attributes_value_dict['color'].append(color)
                        self.attributes_value_dict['size'].append(size)
                        self.attributes_value_dict['material'].append(material)
        self.object_number = len(self.name_list)

    def set_attributes(self, object_id, attribute):

        attribute_dict = {'name': attribute,
                          'input_type': 'text',
                          'mutable': False,
                          'values': [self.attributes_value_dict[attribute][object_id]]}
        return attribute_dict

    def set_coordinates(self, axis):
        location_dict = {"name": "coordination_{}".format(axis),
                         "input_type": "number",
                         "mutable": True,
                         "values": ["-5", "5", "0.0000000000000001"]}
        return location_dict

class Label:

    def __init__(self, shape_values: list, color_values: list, size_values: list, material_values: list, label_color:list):
        self.shape_values = shape_values
        self.color_values = color_values
        self.size_values = size_values
        self.material_values = material_values
        self.label_color =label_color
        self.cvat_array = []

    def set_label(self):
        dataset = Dataset(self.shape_values, self.color_values, self.size_values, self.material_values, self.label_color)
        dataset.get_attributes()
        for object_id in range(dataset.object_number):
            self.cvat_object_label = {}
            self.cvat_object_label['name'] = dataset.name_list[object_id]
            self.cvat_object_label['color'] = dataset.label_color[object_id % len(dataset.label_color)]
            self.cvat_object_label['attributes'] = []
            for attribute in dataset.attributes_list:
                self.cvat_object_label['attributes'].append(dataset.set_attributes(object_id, attribute))
            for axis in ['X', 'Y', 'Z']:
                self.cvat_object_label['attributes'].append(dataset.set_coordinates(axis))
            self.cvat_array.append(self.cvat_object_label)


if __name__ == '__main__':
    shape_values = ['spl', 'cone', 'cube', 'sphere', 'cylinder']
    color_values = ['yellow', 'gold', 'green', 'red', 'brown', 'purple', 'blue', 'cyan', 'gray']
    size_values = ['large', 'small', 'medium']
    material_values = ['metal', 'rubber']
    label_color = ['#fa3253',
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
    new_label = Label(shape_values, color_values, size_values, material_values, label_color)
    new_label.set_label()
    cvat_json = json.dumps(new_label.cvat_array)
    with open('./label_new_setting1.json', 'w', encoding='UTF-8') as output_file:
        output_file.write(cvat_json)
        output_file.close()

