import cv2
print(cv2.__version__)
import numpy as np
import matplotlib.pyplot as plt
from time import time

# replace opencv waitKey() to avoid error due to pyqt5 
def dispImg(str,img, kill_window=True):
    plt.figure()
    plt.imshow(img)
    plt.title(str)
    plt.show(block=False)
    plt.waitforbuttonpress(0)
    if kill_window:
        plt.close('all')
    # cv2.imshow(str,img)
    # cv2.waitKey()

class PrepareData:
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
        
        
        return img

    def presegmentImg(self, img,type='BGR', method = 'grabcut', display_result = True):    
        if display_result:
            dispImg("origin", img, kill_window=False )
        res_img = []
        # method 0: convert to HSV then applying different threshold
        if method == 'threshold':
            if type == 'HSV':
                imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
                dispImg("HSV",imgHSV)
                lower_HSV = np.array([110,50,50],dtype=np.uint8)
                upper_HSV = np.array([130,255,255],dtype=np.uint8)
                mask = cv2.inRange(imgHSV,lower_HSV,upper_HSV)
                res_img = cv2.bitwise_and(imgHSV, imgHSV, mask=mask)
            else:
                lower_RGB = np.array([80,30,0],dtype=np.uint8)
                upper_RGB = np.array([160,50,100],dtype=np.uint8)
                mask = cv2.inRange(img,lower_RGB,upper_RGB)
                res_img = cv2.bitwise_and(img,img,mask=mask)
            
            dispImg("after_masking",res_img)


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
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,1000, 1.0)
            K = 7  # num of clusters
            attempts = 100
            ret, label, center = cv2.kmeans(twoDimg,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
            center = np.uint8(center)
            label = label.flatten()
            res_img = center[label]
            res_img = res_img.reshape((img.shape))
            if display_result:
                dispImg("res_img",res_img,kill_window=False)

            # display each cluster
            for i in range(K):
                masked_img = np.copy(img)
                masked_img = masked_img.reshape((-1,channels))
                masked_img[label != i] = np.zeros(channels)
                masked_img = masked_img.reshape((img.shape))
                if display_result:
                    dispImg(f'cluster{i}',masked_img, kill_window=False)
        
        # method 3: GrabCut
        if method == 'grabcut':
            # mask = cv2.Canny(img,100,200)
            mask = np.zeros(img.shape[:2], np.uint8)
            bgd = np.zeros((1,65),np.float64)
            fgd = np.zeros((1,65),np.float64)
            rect = (30,40,240,150)
            cv2.grabCut(img,mask,rect,bgd,fgd,5,cv2.GC_INIT_WITH_RECT)
            cv2.grabCut(img,mask,rect,bgd,fgd,20,cv2.GC_INIT_WITH_MASK)
            mask2 = np.where((mask==2) | (mask==0),0,1).astype('uint8') # mask to set all bgd and possible bgd to 0.
            res_img = img * mask2[:,:,np.newaxis]
            if display_result:
                dispImg("res0", res_img,kill_window=False)
            # img = cv2.GaussianBlur(img,(5,5),0)
            # first erosion then dilation to remove some bright holes after segmentation
            tmp_img = np.copy(res_img)
            kernel = np.ones((3,3),np.uint8)
            # res_img = cv2.erode(res_img,kernel,iterations=3)
            # res_img = cv2.dilate(res_img,kernel,iterations=1)
            res_img = cv2.morphologyEx(res_img,cv2.MORPH_OPEN,kernel)
            mask_tmp_img = np.where(tmp_img != 0, 255, 0).astype('uint8')
            mask_res_img = np.where(res_img != 0, 255, 0).astype('uint8')
            res_eval = cv2.bitwise_xor(mask_res_img, mask_tmp_img)
            if display_result:
                dispImg("difference after opening", res_eval, kill_window=False)
            # res_img = cv2.ximgproc.anisotropicDiffusion(res_img,0.1,100,100)
        return res_img

    def getContours(self, img, display_result = True):
        disp_contour_val = False
        MIN_ARC_LEN_THRESH = 20
        MIN_AREA_THRESH = 20

        # if area of regions above threshold, need scecond run of GrabCut on it
        MAX_AREA_THRESH = 4000
        refine_area_idx = []
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(imgray, 0, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = list(contours) # convert to list to enable del operation
        print(f'number of contours is {len(contours)}')
        cnt = 0
        bbox_list = []
        while True:
            if (cnt >= len(contours)):
                break

            item = contours[cnt]
            area = cv2.contourArea(item)
            arc_len = cv2.arcLength(item,closed=True)
            print(f'{cnt}. contour, area: {area:4.0f}, length: {arc_len:8.4f}',end=' ')

            # if detected arc_length too small, discard it.
            if (arc_len < MIN_ARC_LEN_THRESH and area < MIN_AREA_THRESH) or area > MAX_AREA_THRESH:
                del contours[cnt]
                print(f' wrong segmentation, contour deleted, now total contours = {len(contours)}')
                continue

            screen = np.zeros(img.shape[0:-1])
            # get the bounding box
            bbox = cv2.boundingRect(item)
            bbox_list.append(bbox)
            # get the shape
            shape = cv2.drawContours(screen,contours,cnt,255,cv2.FILLED)
            shape_mask = np.array(shape,dtype=np.uint8)
            crop_shape= cv2.bitwise_and(img,img, mask=shape_mask) # crop the shape of object from img
            hsv_crop_shape = cv2.cvtColor(crop_shape,cv2.COLOR_BGR2HSV)
            
            
            if display_result:
                x,y,w,h = bbox
                cv2.rectangle(crop_shape, (x,y),(x+w,y+h),255,thickness=1)
                dispImg(f"{cnt}. cropped img",crop_shape,kill_window=False)
                # disp_img("hsv_img",hsv_crop_shape,kill_window=False )
            
            Moments = cv2.moments(item)
            center = [int(Moments['m10']/Moments['m00']), int(Moments['m01']/Moments['m00'])]
            print(f'center of contour is {center}')
            avg_hsv = np.sum(np.sum(hsv_crop_shape,axis=0),axis=0) / area
            avg_rgb = np.sum(np.sum(crop_shape,axis=0),axis=0) / area
            print(f'avg_hsv = {avg_hsv}, avg_rgb = {avg_rgb}')
            
            # display the contours
            if display_result:
                screen = np.zeros(img.shape[0:-1])
                boundary = cv2.drawContours(screen,contours,cnt,255,1)
                boundary = np.array(boundary,np.int32)
                dispImg(f'{cnt}',boundary,kill_window=False)
            if disp_contour_val:
                for i in item:
                    x,y = i[0]
                    print(x, end=',')
                    print(y, end=',')
                print('\n')
            cnt += 1
        if not display_result: 
            screen = np.zeros(img.shape[0:-1])
            all_shapes = cv2.drawContours(screen,contours,-1,255,cv2.FILLED) # disp shape: cv2.FILLED, disp contour: 1
            shape_mask = np.array(all_shapes,dtype=np.uint8)
            crop_shape= cv2.bitwise_and(img,img, mask=shape_mask) # crop the shape of object from img
            for i in bbox_list:
                x,y,w,h = i
                cv2.rectangle(crop_shape, (x,y),(x+w,y+h),255,thickness=1)
            dispImg("cropped img",crop_shape,kill_window=True)

        print(f'number of valid contours is {len(contours)}')

        

        return contours


def selectiveSearch(img):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.switchToSelectiveSearchQuality()
    ss.setBaseImage(img)
    # ss.switchToSelectiveSearchFast()
    rects = ss.process()
    print(f'total number of region proposal: {len(rects)}')
    numShowRects = 100
    screen = img.copy()
    for i, (x,y,w,h) in enumerate(rects):
        if i < numShowRects:
            if (w*h < 400 or w*h > 10000):
                print(f"{w}, {h}")
                continue
            cv2.rectangle(screen, (x,y),(x+w, y+h),(0,255,0), thickness=1, lineType= cv2.LINE_AA)
        else:
            break
    dispImg("selective search", screen,kill_window=False)
    
        

def main():

    # if input("save image from videos?\n") == 'y' :
    #     save_image()
    start = time()
    need_visuliztion = False 
    for i in range(0,10):
        filename = 'frame{}.png'
        # filename = 'test.png'
        print('>>>>>>>>open file: '+filename.format(str(i*10)))
        img = cv2.imread(filename.format(str(i*10)))
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        # edge_ = cv2.Canny(img,100,200)
        # disp_img("edge",edge_,False)
        # img = cv2.ximgproc.anisotropicDiffusion(img,0.1,100,10)
        # selectiveSearch(img)
        pd = PrepareData()
        
        img = pd.presegmentImg(img, method='grabcut',display_result=need_visuliztion)
        # img = pd.process_img(img,method='threshold')
        # # img = process_img(img,type='HSV')

        pd.getContours(img, display_result=need_visuliztion)

    end = time()
    print(f'total time = {end-start}')

if __name__ == "__main__":
    main()