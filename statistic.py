import json


class Statistic:
    def __init__(self):
        material_list = ['r', 'm']
        color_list = ['red', 'blue', 'cyan', 'green', 'gold', 'purple', 'yellow', 'gray', 'brown']
        self.size_list = ['s', 'm', 'l']
        self.shape_list = ['cub', 'con', 'spl', 'sph', 'cyl']
        self.label_dict = {}
        self.name_list = []
        for i in color_list:
            for j in material_list:
                name = i + '_' + j
                self.label_dict[name] = {}
                self.label_dict[name]['hsv'] = []
                self.label_dict[name]['rgb'] = []
                self.label_dict[name]['area'] = []
                self.label_dict[name]['length'] = []
                self.label_dict[name]['center'] = []
                self.label_dict[name]['size'] = []
                self.label_dict[name]['shape'] = []
                self.name_list.append(name)

    def data_log(self, avg_hsv, avg_rgb, area, length, center):
        label_input = input('input the color_material: ')
        if label_input in self.name_list:
            size = input('input the size: ')
            if size in self.size_list:
                shape = input('input the shape: ')
                if shape in self.shape_list:
                    self.label_dict[label_input]['hsv'].append(avg_hsv.tolist())
                    self.label_dict[label_input]['rgb'].append(avg_rgb.tolist())
                    self.label_dict[label_input]['area'].append(area)
                    self.label_dict[label_input]['length'].append(length)
                    self.label_dict[label_input]['center'].append(center)
                    self.label_dict[label_input]['size'].append(size)
                    self.label_dict[label_input]['shape'].append(shape)

    def data_store(self, name):
        datalist = []
        datalist.append(self.label_dict)
        data_json = json.dumps(datalist)
        with open('./dataset/data{}'.format(name), 'w', encoding='UTF-8') as output_file:
            output_file.write(data_json)
            output_file.close()
        return self.label_dict