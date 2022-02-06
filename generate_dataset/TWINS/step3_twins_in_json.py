import json

class Twins_in_json:

    def __init__(self, json_path: str):
                   
        with open(json_path, 'r', encoding = 'UTF-8') as json_file:
            json_dict = json.load(json_file)
            json_file.close()
        self.objects = json_dict['objects']
        self.twins = []

    def get_coordination(self, obj:dict):
        coordination = {}
        for frame_num in [0, 50, 100, 150, 200, 250, 300]:
            coordination[str(frame_num)] = obj['locations'][str(frame_num)]
        return coordination

    def collect_json(self):
        
        label_list = []
        twins_label = []

        for obj in self.objects:
            label = obj['category']
            if label in label_list:
                twins_label.append(label)
            else:
                label_list.append(label)

        for obj in self.objects:
            label = obj['category']
            if label in twins_label:
                coordination = self.get_coordination(obj)
                self.twins.append({label: coordination})

if __name__ == "__main__":

    start = 5200
    end = 5499

    black_list = [5251,5258,5300,5375,5393,5398,5427,5457]
    twins_dict = {}
    
    for i in range(start,int(end+1)):
        
        if i in black_list:
            pass

        else:
            json_path = './json_data/CATER_new_00{}.json'.format(i)
            tij = Twins_in_json(json_path)
            tij.collect_json()
            
            if tij.twins:
                twins_dict['CATER_new_00{}'.format(i)] = tij.twins
                print(2*'<<<<<<<<<'+str(i)+' twins in json check finish')
                
    output = json.dumps(twins_dict)
    with open('./twins_in_json.json', 'w', encoding = 'UTF-8') as output_file:
        output_file.write(output)
        output_file.close()
    print('success!!!')