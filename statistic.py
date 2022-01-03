import json


class Statistic:
    def __init__(self):
        material_list = ['r','m']
        color_list = ['red', 'blue', 'cyan', 'green', 'gold', 'purple', 'yellow', 'gray', 'brown']
        self.size_list = ['s', 'b', 'l']
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
                self.name_list.append(name)

    def data_log(self, avg_hsv, avg_rgb, area, length, center):
        label_input = input('input the color_material: ')
        if label_input in self.name_list:
            size = input('input the size: ')
            if size in self.size_list:
                self.label_dict[label_input]['hsv'].append(avg_hsv.tolist())
                self.label_dict[label_input]['rgb'].append(avg_rgb.tolist())
                self.label_dict[label_input]['area'].append(area)
                self.label_dict[label_input]['length'].append(length)
                self.label_dict[label_input]['center'].append(center)
                self.label_dict[label_input]['size'].append(size)

    def data_store(self):
        datalist = []
        datalist.append(self.label_dict)
        color_json = json.dumps(datalist)
        with open('./data', 'w', encoding='UTF-8') as output_file:
            output_file.write(color_json)
            output_file.close()