import matplotlib.pyplot as plt



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