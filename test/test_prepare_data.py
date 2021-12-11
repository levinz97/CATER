import unittest
from prepare_data import PrepareData
import cv2


class Test_prepare_data(unittest.TestCase):
    
    def test_get_contours(self):
        pd = PrepareData()
        for i in range(0,31):
            filename = 'frame{}.png'.format(str(i*10))
            print('>>>>>>>>open file: '+filename)
            img = cv2.imread(filename)
            img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            img = pd.presegmentImg(img, method='grabcut',display_result=False)
            contours = pd.getContours(img,display_result=False)
            self.assertGreater(len(contours), -1)



if __name__ == '__main__':
    unittest.main()
    
    