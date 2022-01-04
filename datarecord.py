import json

input_path = './data_2'
with open(input_path, 'r', encoding='UTF-8') as input_file:
    dictionary = json.load(input_file)
    input_file.close()
number = len(dictionary)
#print(dictionary)
print(number)

material_list = ['r', 'm']
color_list = ['red', 'blue', 'cyan', 'green', 'gold', 'purple', 'yellow', 'gray', 'brown']
size_list = ['s', 'm', 'l']
shape_list = ['cub', 'con', 'spl', 'sph', 'cyl']
attribute_list = ['hsv', 'rgb', 'area', 'length', 'center', 'size', 'shape']
label_dict = {}
name_list = []
for i in color_list:
    for j in material_list:
        name = i + '_' + j
        label_dict[name] = {}
        label_dict[name]['hsv'] = []
        label_dict[name]['rgb'] = []
        label_dict[name]['area'] = []
        label_dict[name]['length'] = []
        label_dict[name]['center'] = []
        label_dict[name]['size'] = []
        label_dict[name]['shape'] = []
        name_list.append(name)

for i in dictionary:
    for name in name_list:
        for attribute in attribute_list:
            label_dict[name][attribute].extend(i[name][attribute])

print(label_dict)

k = 0
for key, value in label_dict.items():
    k = k + len(value['hsv'])

print(k)

list1 = []
list1.append(label_dict)
data_json1 = json.dumps(list1)
with open('./data_final', 'w', encoding='UTF-8') as output_file:
    output_file.write(data_json1)
    output_file.close()
