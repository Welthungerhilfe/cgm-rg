from yacs.config import CfgNode


_C = CfgNode()

_C.GPUS = (0,)
_C.WORKERS = 4

# Cudnn related params
_C.CUDNN = CfgNode()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CfgNode()
_C.MODEL.NAME = 'pose_hrnet'
_C.MODEL.NUM_JOINTS = 17
_C.MODEL.TAG_PER_JOINT = True
_C.MODEL.IMAGE_SIZE = [256, 256]  # width * height, ex: 192 * 256
_C.MODEL.EXTRA = CfgNode(new_allowed=True)

_C.BOX_MODEL = CfgNode()
_C.BOX_MODEL.THRESHOLD = 0.8


# testing
_C.TEST = CfgNode()

# size of images for each device
# Test Model Epoch
_C.TEST.POST_PROCESS = False


# nms
_C.TEST.MODEL_FILE = ''


def update_config(cfg, args):

    cfg.defrost()
    cfg.merge_from_file(args)

    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
