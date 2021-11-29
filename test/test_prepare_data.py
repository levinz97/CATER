import unittest
from prepare_data import prepare_data
import cv2


class Test_prepare_data(unittest.TestCase):
    
    def test_get_contours(self):
        pd = prepare_data()
        for i in range(0,31):
            filename = 'frame{}.png'.format(str(i*10))
            print('>>>>>>>>open file: '+filename)
            img = cv2.imread(filename)
            img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            img = pd.presegment_img(img, method='grabcut',display_result=False)
            contours = pd.get_contours(img,display_result=False)
            self.assertGreater(len(contours), 0)



if __name__ == '__main__':
    unittest.main()
    
    