from yacs.config import CfgNode as CN

_C = CN()
_C.SEED = 42

_C.dist_url = 'env://'
_C.world_size = 1

# _C.MU = 10
_C.lam = 1
_C.beta = 100

_C.DATASET = CN()
_C.DATASET.ROOT = [
    '/home/jc3/yzy/datasets/fastMRI_brain_DICOM_mat',
    '/home/jc3/Data/BraTS2019',
    '/home/jc3/Data',
    '/home/jc3/Data']

_C.DATASET.CLIENTS = ['fastMRI', 'BraTS' ,'JiangSu', 'lianying']
_C.DATASET.PATTERN = ['T1+T2', 'T1+T2', 'T1+T2', 'T1+T2']
_C.DATASET.SAMPLE_RATE = [0.2, 0.2, 1, 1]
_C.DATASET.CHALLENGE = 'singlecoil'  # the task of ours, singlecoil or multicoil

_C.TRANSFORMS = CN()
_C.TRANSFORMS.DICOM_MASK_DIR = '/home/jc3/yzy/datasets/dicom_split_masks/'
_C.TRANSFORMS.MASK_FILE = ["2D-Radial-4x_192.mat", "1D-Cartesian_5x_240.mat",
                           "2D-Radial-4x_192.mat", "2D-Random-6x_192.mat"]
_C.TRANSFORMS.MASK_DIR = '/home/jc3/Data/maskyzy'
_C.TRANSFORMS.MASK_SPEC = ['2D-Radial-4X', '2D-Random-6X']



_C.FL = CN()
_C.FL.CLIENTS_NUM = 4
_C.FL.MODEL_NAME = 'unet_poor3'
_C.FL.SHARE_WAY = 'only_encoder'  # 'whole_archi' or 'only_encoder' or 'except_last'
_C.FL.DATAMIX = False # 'pd'
_C.FL.SHOW_SIZE = True

# model config
_C.MODEL = CN()
_C.MODEL.INPUT_DIM = 1   # the channel of input
_C.MODEL.OUTPUT_DIM = 1   # the channel of output

_C.MULTI = CN()
_C.MULTI.MODE = 'base'       # 'concat' means concat image direct, 'base' means single

# the solver config

_C.SOLVER = CN()
_C.SOLVER.DEVICE = 'cuda'
_C.SOLVER.DEVICE_IDS = [0, 1]  # if [] use cpu, else gpu
_C.SOLVER.LR = [1e-4,1e-4, 1e-4, 1e-4]
_C.SOLVER.WEIGHT_DECAY = 1e-4
_C.SOLVER.LR_DROP = 40
_C.SOLVER.BATCH_SIZE = 8
_C.SOLVER.NUM_WORKERS = 1
_C.SOLVER.PRINT_FREQ = 10
_C.SOLVER.LR_GAMMA = 0.1

# the others config
_C.RESUME = 'data3_only_encoder_ours/jiangsu/unet_jiangsu'  # model resume path
_C.OUTPUTDIR = './data3_only_encoder_ours/lianying/'  # the model output dir
_C.LOGDIR = './logs'
_C.TEST_OUTPUTDIR = 'data3_output_result/jiangsu/unet_1'

#the train configs
_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 50  # the train epochs
_C.TRAIN.SMALL_EPOCHS = 10