import cv2
import os

class video_convert_img:

    def __init__(self, video_path, step):
        self.video_path = video_path
        self.step = step

    def initialization(self):
        self.folder_name = './img'
        os.makedirs(self.folder_name, exist_ok=True)
        self.video_list = os.listdir(self.video_path)


    def get_img(self, video):
        capture = cv2.VideoCapture(self.video_path + '/' + video)
        frame_id = -1
        while (capture.isOpened()):
            frame_id += 1
            ret, frame = capture.read()
            img_path = self.folder_name+'/'
            if ret:
                if frame_id % self.step == 0:
                    cv2.imwrite(img_path + video.split('.')[0] + '_' + str(frame_id) + '.png', frame)
                cv2.waitKey(1)
            else:
                break
        capture.release()
        print('save_success')
        print(video)


if __name__ == '__main__':
    video_path = './video'
    step = 50
    video_set = video_convert_img(video_path,step)
    video_set.initialization()
    for video in video_set.video_list:
        video_set.get_img(video)


