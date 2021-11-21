import cv2
print(cv2.__version__)
import numpy as np


class prepare_data:
    def __init__(self) -> None:
        pass
    
    def save_image(self):
        vidcap = cv2.VideoCapture("./raw_data/all_action_camera_move/videos/CATER_new_005748.avi")
        success, image = vidcap.read()
        count = 0
        while success:
            if count % 50 == 0:
                cv2.imwrite(f'frame{count}.png', image) # save as PNG file
            success,image = vidcap.read()
            print('Read a new frame ', success)
            count += 1
            chr = cv2.waitKey(10)
            if chr == 'q':
                break

    def preprocess_img(self, img):
        cv2.imshow(" ", img)
        cv2.waitKey()
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        imgHSV[:,:,1] = 255
        return imgHSV

    def process_img(self, img,type='BGR', method = 'kmeans'):    
        cv2.imshow("origin", img)
        cv2.waitKey()
        res_img = []
        # method 0: convert to HSV then applying different threshold
        if method == 'threshold':
            # imgHLS = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
            imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            # cv2.imshow("HLS",imgHLS)
            # cv2.waitKey()
            cv2.imshow("HSV",imgHSV)
            cv2.waitKey()
            lower_HSV = np.array([110,50,50])
            upper_HSV = np.array([130,255,255])
            mask = cv2.inRange(imgHSV, lower_HSV,upper_HSV)
            res_img = cv2.bitwise_and(imgHSV,imgHSV,mask=mask)
            cv2.imshow("after masking", res_img)
            cv2.waitKey()


        # methed 1: using watershed 
        # makers = cv2.watershed(imgHLS, markers)
        
        # method 2: using kmeans
        if method == 'kmeans':
            channels = 3 if len(img.shape) > 2 else 1
            # assert channels == 1 , "number of channels"
            if type == 'HSV':
                img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            twoDimg = img.reshape((-1, channels))
            twoDimg = np.float32(twoDimg)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10000, 1.0)
            K = 8 # num of clusters
            attempts = 100
            ret, label, center = cv2.kmeans(twoDimg,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
            center = np.uint8(center)
            label = label.flatten()
            res_img = center[label]
            res_img = res_img.reshape((img.shape))
            cv2.imshow("res_img", res_img)
            cv2.waitKey()
            # display each cluster
            for i in range(K):
                masked_img = np.copy(img)
                masked_img = masked_img.reshape((-1,channels))
                masked_img[label != i] = np.zeros(channels)
                masked_img = masked_img.reshape((img.shape))
                cv2.imshow(f'cluster{i}', masked_img)
                cv2.waitKey()
        
        # method 3: GrabCut
        if method == 'grabcut':
            mask = np.zeros(img.shape[:2], np.uint8)
            bgd = np.zeros((1,65),np.float64)
            fgd = np.zeros((1,65),np.float64)
            rect = (40,40,240,150)
            cv2.grabCut(img,mask,rect,bgd,fgd,50,cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask==2) | (mask==0),0,1).astype('uint8')
            res_img = img * mask2[:,:,np.newaxis]
            cv2.imshow("res",res_img)
            cv2.waitKey()

        return res_img

if __name__ == "__main__":
    # if input("save image from videos?\n") == 'y' :
    #     save_image()
    img = cv2.imread("frame0.png")
    pd = prepare_data()
    img = pd.process_img(img, method='grabcut')
    # img = process_img(img,type='HSV')