import unittest
from prepare_data import PrepareData
import cv2
import numpy as np

class Test_prepare_data(unittest.TestCase):
    
    def test_get_contours(self):
        pd = PrepareData(need_visualization=False)
        num_detected_contour = []
        for i in range(0,31):
            filename = 'frame{}.png'.format(str(i*10))
            print('>>>>>>>>open file: '+filename)
            img = cv2.imread(filename)
            img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            # img = pd.presegmentImg(img, method='grabcut')
            # contours = pd.getContoursFromSegmentedImg(img)
            contours,_ = pd.getContoursWithBbox(img)
            num = len(contours)
            num_detected_contour.append(num)
            self.assertGreater(len(contours), 1, msg=f"fails at frame {i}")
        print(f"minimum detected contour is {np.min(num_detected_contour)}")
        print(f"average detected contour is {np.mean(num_detected_contour)}")



if __name__ == '__main__':
    unittest.main()
    
    