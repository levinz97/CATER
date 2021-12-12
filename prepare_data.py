import cv2
print(cv2.__version__)
import numpy as np
from time import time
from non_maximum_suppression import non_max_suppression
from utils import dispImg
import os

class PrepareData:
    def __init__(self, need_visualization=True):
        self.display_process = need_visualization
        self.display_result =  True
        self.display_selectiveSearch = True
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

    def getContoursWithBbox(self, raw_img):
        r'wrap function to preform presegmetImg then iteratively refined with grabCut, input: raw image -> contours, bbox'
        img = raw_img.copy()
        img = self.presegmentImg(img, method='grabcut')
        contours, refine_area_list, bbox_list = self.getContoursFromSegmentedImg(img)
        # refine the wrong segmented region with iterative grabCut
        max_iterative_cnt = 10
        cnt = 0 # count iterative time
        while len(refine_area_list) > 0:
            cnt += 1
            print(">>"*20, f'{cnt+1} run of grabcut')
            if cnt > max_iterative_cnt:
                break
            tmp_refine_list = []
            nms_bbox = []
            for idx, bbox in enumerate(refine_area_list):
                _img = self._grabCut(raw_img, bbox)
                if cnt % 7 == 0:
                    _,_,_w,_h = bbox
                    rect_from_ss = self.selectiveSearch(_img, _w*_h)
                    print(f'before nms there are {len(rect_from_ss)} bbox')
                    nms_bbox = non_max_suppression(rect_from_ss, 0.1)
                    print(nms_bbox)
                    if len(nms_bbox) > 0:
                        # expand nms_bbox a little
                        for i in range(2):
                            nms_bbox[:,i] -= nms_bbox[:,i]//15
                        for i in range(2,4):
                            nms_bbox[:,i] += nms_bbox[:,i]//4
                        print(f'after nms there are {len(nms_bbox)} bbox')
                        if self.display_selectiveSearch:
                            _raw_img = raw_img.copy() 
                            _raw_img = self._drawBboxOnImg(_raw_img, nms_bbox)
                            dispImg("after nms",_raw_img, kill_window=False)
                _contours, _refine_area_list, _bbox_list = self.getContoursFromSegmentedImg(_img)
                # update new values calculated from refined area
                contours += _contours
                bbox_list += _bbox_list
                tmp_refine_list += _refine_area_list
                tmp_refine_list += list(nms_bbox)
            refine_area_list = tmp_refine_list
        if self.display_result:          
            self._dispAllContours(raw_img, contours, bbox_list)

        print("\n","<<"*50,f"Final Number of detected contours: {len(contours)}",">>"*10)
        
        return contours, bbox_list

    def presegmentImg(self, img,type='BGR', method = 'grabcut'): 
        'try to segment image with or different conventional / unsupervised approach'   
        if self.display_process:
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
            if self.display_process:
                dispImg("res_img",res_img,kill_window=False)


            # display each cluster
            for i in range(K):
                masked_img = np.copy(img)
                masked_img = masked_img.reshape((-1,channels))
                masked_img[label != i] = np.zeros(channels)
                masked_img = masked_img.reshape((img.shape))
                if self.display_process:
                    dispImg(f'cluster{i}',masked_img, kill_window=False)

        
        # method 3: GrabCut
        if method == 'grabcut':
            vorground_rect = (30,40,240,150)
            res_img = self._grabCut(img, vorground_rect)
        return res_img

    def _grabCut(self, img, rect) -> np.array: 
        'return the segmented image use cv2.grabCut'
        if self.display_process:
            screen = img.copy()
            _x,_y,_w,_h = rect
            cv2.rectangle(screen, (_x,_y),(_x+_w,_y+_h),255,thickness=1)
            dispImg("in_grabCut", screen)
        mask = cv2.Canny(img,100,200)
        _, mask = cv2.threshold(mask, 10, 1, 0)
        # dispImg("canny",mask,kill_window=False)
        # mask = np.zeros(img.shape[:2], np.uint8)
        bgd = np.zeros((1,65),np.float64)
        fgd = np.zeros((1,65),np.float64)

        cv2.grabCut(img,mask,rect,bgd,fgd,5,cv2.GC_INIT_WITH_RECT)
        if self.display_process:
            dispImg("new_mask",mask, kill_window=False)

        cv2.grabCut(img,mask,None,bgd,fgd,15,cv2.GC_INIT_WITH_MASK)
        mask2 = np.where((mask==2) | (mask==0),0,1).astype('uint8') # mask to set all bgd and possible bgd to 0.
        res_img = img * mask2[:,:,np.newaxis]
        if self.display_process:
            dispImg("res0", res_img,kill_window=False)
        # img = cv2.GaussianBlur(img,(5,5),0)
        # first erosion then dilation to remove some bright holes after segmentation
        tmp_img = np.copy(res_img)
        kernel = np.ones((2,2),np.uint8)
        # res_img = cv2.erode(res_img,kernel,iterations=3)
        # res_img = cv2.dilate(res_img,kernel,iterations=1)
        res_img = cv2.morphologyEx(res_img,cv2.MORPH_OPEN,kernel)
        mask_tmp_img = np.where(tmp_img != 0, 255, 0).astype('uint8')
        mask_res_img = np.where(res_img != 0, 255, 0).astype('uint8')
        res_eval = cv2.bitwise_xor(mask_res_img, mask_tmp_img)
        if self.display_process:
            dispImg("difference after opening", res_eval, kill_window=False)
        # res_img = cv2.ximgproc.anisotropicDiffusion(res_img,0.1,100,100)

        return res_img

    def getContoursFromSegmentedImg(self, img):
        disp_contour_val = False
        MIN_ARC_LEN_THRESH = 40
        MIN_AREA_THRESH = 40

        # if area of regions above threshold, need scecond run of GrabCut on it
        MAX_AREA_THRESH = 5000
        SOFT_AREA_THRESH = 10000
        refine_area_list = []
        # convert to single channel, required by cv2.findContours()
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(imgray, 0, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = list(contours) # convert to list to enable del operation
        print(f'original number of contours is {len(contours)}')
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
            if arc_len < MIN_ARC_LEN_THRESH and area < MIN_AREA_THRESH:
                del contours[cnt]
                print(f'wrong segmentation, too small, contour deleted, now total contours = {len(contours)}')
                continue

            screen = np.zeros(img.shape[0:-1])
            # get the bounding box
            bbox = cv2.boundingRect(item)
       
            # get the shape
            shape = cv2.drawContours(screen,contours,cnt,255,cv2.FILLED)
            shape_mask = np.array(shape,dtype=np.uint8)
            crop_shape= cv2.bitwise_and(img,img, mask=shape_mask) # crop the shape of object from img
            hsv_crop_shape = cv2.cvtColor(crop_shape,cv2.COLOR_BGR2HSV)
            avg_hsv = np.sum(np.sum(hsv_crop_shape,axis=0),axis=0) / area
            avg_rgb = np.sum(np.sum(crop_shape,axis=0),axis=0) / area
            print(f'avg_hsv = {avg_hsv}, avg_rgb = {avg_rgb}')
            # if the area too large, refine it with another grabCut
            hue = avg_hsv[0]
            if area > MAX_AREA_THRESH or ( SOFT_AREA_THRESH < area and not self._isInColorRange(hue)):
                refine_area_list.append(bbox)
                del contours[cnt]
                print(f'wrong segmentation, too large, now total contours = {len(contours)}')
                continue
            else:
                bbox_list.append(bbox)

            if self.display_process:
                _x,_y,_w,_h = bbox
                cv2.rectangle(crop_shape, (_x,_y),(_x+_w,_y+_h),255,thickness=1)
                dispImg(f"{cnt}. cropped img",crop_shape,kill_window=False)
                # disp_img("hsv_img",hsv_crop_shape,kill_window=False )
            
            Moments = cv2.moments(item)
            center = [int(Moments['m10']/Moments['m00']), int(Moments['m01']/Moments['m00'])]
            print(f'center of contour is {center}')
  
            
            # display the contours
            if self.display_process:
                screen = np.zeros(img.shape[0:-1])
                boundary = cv2.drawContours(screen,contours,cnt,255,1)
                boundary = np.array(boundary,np.int32)
                dispImg(f'{cnt}',boundary,kill_window=False)
            if disp_contour_val:
                for i in item:
                    contour_x,contour_y = i[0]
                    print(f"{contour_x}, {contour_y}", end=',')
                print('\n')
            cnt += 1
        if self.display_process:
            self._dispAllContours(img, contours, bbox_list)

        return contours, refine_area_list, bbox_list

    def _dispAllContours(self, img, contours, bbox_list):
        screen = np.zeros(img.shape[0:-1])
        all_shapes = cv2.drawContours(screen,contours,-1,255,cv2.FILLED) # disp shape: cv2.FILLED, disp contour: 1
        shape_mask = np.array(all_shapes,dtype=np.uint8)
        crop_shape = cv2.bitwise_and(img,img, mask=shape_mask) # crop the shape of object from img
        crop_shape = self._drawBboxOnImg(crop_shape, bbox_list)

        dispImg("cropped img",crop_shape,kill_window=True)
        print(f'number of valid contours is {len(contours)}')

    def _drawBboxOnImg(self, img, bbox_list):
        for _, (_x,_y,_w,_h) in enumerate(bbox_list):
            cv2.rectangle(img, (_x,_y),(_x+_w,_y+_h),255,thickness=1)
        return img
    
    def _isInColorRange(self, hue: float):
        "check the detected object is single object or mixed objects by color"
        red = (122, 130)
        brown = (106.4, 115.1)
        yellow = (100,106.1)
        colors = [red, brown, yellow]
        num_of_colors = len(colors)
        isInRange = lambda range, hue : range[0] < hue < range[1]
        
        for i in range(num_of_colors):
            if isInRange(colors[i], hue):
                return True

        return False    


    def selectiveSearch(self, img, img_size=2000):
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(img)
        ss.switchToSelectiveSearchQuality()
        # ss.switchToSelectiveSearchFast()
        rects = np.array(ss.process())
        print(f'total number of region proposal: {len(rects)}')
        numShowRects = 100
        screen = img.copy()
        picked = []
        for i, (x,y,w,h) in enumerate(rects):
            if i < numShowRects:
                if (w*h < 0.1*img_size or w*h > 0.9*img_size):
                    print(f"{w}, {h}")
                    continue
                else:
                    picked.append(i)
                if self.display_selectiveSearch:
                    cv2.rectangle(screen, (x,y),(x+w, y+h),(0,255,0), thickness=1, lineType= cv2.LINE_AA)
            else:
                break
        if self.display_selectiveSearch:
            dispImg("selective search", screen,kill_window=False)
        return rects[picked]
        
        

def main():

    # if input("save image from videos?\n") == 'y' :
    #     save_image()
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)
    dirname = os.path.join('.','raw_data','first_frame', 'all_actions_first_frame')
    start = time()
    need_visualization = False
    for i in range(0,31):
        filename = 'frame{}.png'
        filenum = str(i)
        while len(filenum) < 6:
            filenum = '0'+ filenum
        filename = "/CATER_new_{}.png".format(filenum)
        filename = dirname + filename
        if not os.path.isfile(filename):
            continue
        # filename = 'test.png'
        print('\n',5*'>>>>>>>>','open file: '+filename.format(str(i*10)))
        img = cv2.imread(filename.format(str(i*10)))
        raw_img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

        # img = cv2.ximgproc.anisotropicDiffusion(img,0.1,100,10)
        # selectiveSearch(img)

        pd = PrepareData(need_visualization)
        contours, bbox_list = pd.getContoursWithBbox(raw_img)
        # assert(len(contours)==len(bbox_list),"size not compatible")
        
    end = time()
    print(f'total time = {end-start}')

if __name__ == "__main__":
    main()

