import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
import cv2
import numpy as np

from numpy.core.fromnumeric import resize

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
 
def getRectFromUserSelect(img)->list:
    """
    equivalent to cv2.selectROI,
    get rectangle from user select, note that only last selection will be saved!
    exit with key 'q'
    """
    ##not using opencv due to pyqt conflict
    # rect = cv2.selectROI("region", img, showCrosshair=True)
    # print(rect)
    # return rect

    rects = []
    def onselect(eclick, erelease):
        x1,y1 = eclick.xdata, eclick.ydata
        x2,y2 = erelease.xdata, erelease.ydata
        w = x2 - x1
        h = y2 - y1
        rects.append((int(x1),int(y1),int(w),int(h)))
        print(f"starting x = {eclick.xdata:3.1f}, y={eclick.ydata:3.1f}", end=' ')
        print(f"ending x = {erelease.xdata:3.1f}, y={erelease.ydata:3.1f}")
    def press(event):
        global continue_to_select
        if event.key == 'n':
            continue_to_select = False
            plt.close('all')
        if event.key == 'c':
            print("clear all the values")
            rects.clear()
        if event.key == 'r':
            print("remove last value")
            if len(rects) > 0:
                rects.pop(-1)
            else:
                print("no rectangle to remove, list is empty")
        if event.key == 'p':
            print(rects)
    props = dict(facecolor='blue', alpha=0.5)

    fig, ax = plt.subplots()
    # ax.plot(img)
    # ax.plot([1, 2, 3], [10, 50, 100])
    ax.imshow(img)
    fig.canvas.mpl_connect('key_press_event', press)
    rect = mwidgets.RectangleSelector(ax, onselect, interactive=True,rectprops = props)
    plt.show()
    # print(rect)

    print(f"final selection: {rects}")
    return rects

if __name__ == "__main__":
    img = cv2.imread('test.png')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # dispImg("raw img", img)
    getRectFromUserSelect(img)