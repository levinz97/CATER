from detectron2.config import CfgNode

def add_cater_config(cfg: CfgNode):
    """
    add config for cater roi head
    """
    _C = cfg

    _C.MODEL.SEPARATE_ATTR_ON                          = False
    _C.MODEL.ROI_HEADS.NAME                            = 'CaterROIHeads'
    
    _C.MODEL.ROI_COORDINATE_HEAD = CfgNode()
    _C.MODEL.ROI_COORDINATE_HEAD.NAME = 'coordinateHead'
    _C.MODEL.ROI_COORDINATE_HEAD.USE_BACKBONE_FEATURES =  False
    _C.MODEL.ROI_COORDINATE_HEAD.IN_FEATURES           = []
    _C.MODEL.ROI_COORDINATE_HEAD.IMG_SIZE              = (240, 320)
    _C.MODEL.ROI_COORDINATE_HEAD.POOLER_RESOLUTION     = 14
    _C.MODEL.ROI_COORDINATE_HEAD.POOLER_SAMPLING_RATIO = 2
    _C.MODEL.ROI_COORDINATE_HEAD.POOLER_TYPE           = 'ROIAlignV2'

    _C.MODEL.ROI_COORDINATE_HEAD.CONV_HEAD_KERNEL_SIZE = 3
    _C.MODEL.ROI_COORDINATE_HEAD.NUM_STACKED_CONVS     = 3