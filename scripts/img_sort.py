import os
import shutil

class img_sort:

    def __init__(self, img_path):
        self.img_path = img_path
        self.filename = self.img_path + '_' + 'sort'
        os.makedirs(self.filename, exist_ok=True)

    def sort(self):
        video_list = []
        self.img_list = os.listdir(self.img_path)
        for img in self.img_list:
            video_name = self.filename + '/' + 'CATER_new_' + img.split('_')[2]
            if not video_name in video_list:
                video_list.append(video_name)
                os.makedirs(video_name, exist_ok=True)
                shutil.copy(self.img_path + '/' + img, video_name)
            else:
                shutil.copy(self.img_path + '/' + img, video_name)

if __name__ == '__main__':
    img_path = r'D:\Das dritte Semester\HCI\raw_data_from_005200_to_005699\005600-005699'
    step = 50
    img_set = img_sort(img_path)
    img_set.sort()





