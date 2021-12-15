import numpy as np

def non_max_suppression(bbox_list: np.array, overlap_thresh):
    if len(bbox_list) == 0:
        return []
    # convert to float if is int type
    if bbox_list.dtype.kind == 'i':
        bbox_list = bbox_list.astype("float")
    
    # store the index of picked bbox
    picked = []
    # all upper left points
    x1 = bbox_list[:,0]
    y1 = bbox_list[:,1]
    # all width and height
    w = bbox_list[:,2]
    h = bbox_list[:,3]
    # all bottom right points
    x2 = x1 + w 
    y2 = y1 + h
    
    area = w * h
    idxs = np.argsort(y2)

    while len(idxs) > 0:

        last = len(idxs) - 1
        i = idxs[last]
        picked.append(i)

        xx1 = np.maximum(x1[i],x1[idxs[:last]])
        yy1 = np.maximum(y1[i],y1[idxs[:last]])
        xx2 = np.minimum(x2[i],x2[idxs[:last]])
        yy2 = np.minimum(y2[i],y2[idxs[:last]])

        _w = np.maximum(0, xx2 - xx1 + 1)
        _h = np.maximum(0, yy2 - yy1 + 1)

        overlap = _w * _h / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last],np.where(overlap > overlap_thresh)[0])))

    return bbox_list[picked].astype("int")