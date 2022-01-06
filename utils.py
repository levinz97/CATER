import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
import cv2
import numpy as np
import matplotlib
import json

# replace opencv waitKey() to avoid error due to pyqt5 
def dispImg(str,img, kill_window=True, on_press=None, move_dist=[400, 200]):
    fig = plt.figure()
    move_figure(fig, *move_dist)
    plt.imshow(img)
    plt.title(str)
    if on_press is None:
        plt.show(block=False)
        while True:
            if plt.waitforbuttonpress(0):
                break
    else:
        fig.canvas.mpl_connect('key_press_event', on_press)
        plt.show()
    if kill_window:
        plt.close('all')
    # cv2.imshow(str,img)
    # cv2.waitKey()

def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)

def getRectFromUserSelect(img)->list:
    """
    equivalent to cv2.selectROI,
    get rectangle from user select, note that wrong selection is fatal for later grabcut!
    exit with key 'q'
    """
    ##not using opencv due to pyqt conflict
    # rect = cv2.selectROI("region", img, showCrosshair=True)
    # print(rect)
    # return rect
    MIN_AREA_THRESH = 60
    rects = []
    rect_tmp = []
    def onselect(eclick, erelease):
        x1,y1 = eclick.xdata, eclick.ydata
        x2,y2 = erelease.xdata, erelease.ydata
        w = x2 - x1
        h = y2 - y1
        if w*h > MIN_AREA_THRESH:
            rect_tmp.append((int(x1),int(y1),int(w),int(h)))
            print(f"starting x = {eclick.xdata:3.1f}, y={eclick.ydata:3.1f}", end=' ')
            print(f"ending x = {erelease.xdata:3.1f}, y={erelease.ydata:3.1f}")
        else:
            print("possible wrong selection, too small")
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
        if event.key == ' ' or event.key == 'enter':
            if len(rect_tmp) > 0:
                rects.append(rect_tmp.pop(-1))
                print(f"save the last selected rectangle{rects[-1]}")
        if event.key == 'v':
            print(rects)
    props = dict(facecolor='blue', alpha=0.5)

    fig, ax = plt.subplots()
    # ax.plot(img)
    # ax.plot([1, 2, 3], [10, 50, 100])
    move_figure(fig, 1000, 200)
    ax.imshow(img)
    ax.set_title("please select the region with mouse!")
    fig.canvas.mpl_connect('key_press_event', press)
    rect = mwidgets.RectangleSelector(ax, onselect, interactive=True,rectprops = props)
    plt.show()
    # print(rect)

    print(f"final selection: {rects}")
    return rects

class StatusLogger:
    def __init__(self, file='status.json'):
        self.file = file
    def reset_logger(self):
        with open(self.file, 'r+', encoding='utf-8') as f:
            data = json.load(f)
            data['finished_secdir'] = []
            data['finished_thirdir'] = []
            data['current_dir'] = ''
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()
           

    def update_status(self, current_dir:str = None, finished_secdir= None, finished_thirdir = None):
        with open(self.file, 'r+', encoding='utf-8') as f:
            data = json.load(f)
            if current_dir is not None:
                data['current_dir'] = current_dir
            if finished_secdir is not None:
                if isinstance(finished_secdir, str):
                    data['finished_secdir'].append(finished_secdir)
                elif isinstance(finished_secdir, list):
                    data['finished_secdir'] += finished_secdir
            if finished_thirdir is not None:
                if isinstance(finished_thirdir, str):
                    data['finished_thirdir'].append(finished_thirdir)
                elif isinstance(finished_thirdir, list):   
                    data['finished_thirdir'] += finished_thirdir
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()

    def get_status(self):
        with open(self.file, 'r+', encoding='utf-8') as f:
            data = json.load(f)
            cur_d, finished_secdir, finished_thirdir = data['current_dir'], data['finished_secdir'],data['finished_thirdir']
        return cur_d, finished_secdir, finished_thirdir 

if __name__ == "__main__":
    # img = cv2.imread('test.png')
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # dispImg("raw img", img,kill_window=False)
    # getRectFromUserSelect(img)
    file = 'status.json'
    sl = StatusLogger()
    sl.reset_logger()
    sl.update_status(' ', [])
    sl.update_status('test_dir')
    sl.update_status(finished_secdir='000')
    cur_d, finished_secdir,finished_thirdir = sl.get_status()
    if '000' in finished_secdir:
        print("finished")
    sl.reset_logger()
