from yacs.config import CfgNode as CN

# config definition
_C = CN()

_C.SEED = 42

_C.dist_url = 'env://'
_C.world_size = 1

# dataset config
_C.DATASET = CN()
_C.DATASET.ROOT = '/home/jc3/Data'  # the root of dataset
# _C.DATASET.CLIENTS =     ['fastMRI', 'JiangSu', 'lianying', 'IXI']  # ['pd', 'pdfs', 'JiangSu', 'lianying', 'IXI']  ['IXI_in_5']
# _C.DATASET.PATTERN =     ['pd+pdfs', 'T1+T2', 'T1+T2', 'T2']  # 'pd'  'pdfs'  'pd+pdfs'  'T1'  'T2'  'T1+T2'
_C.DATASET.CLIENTS =     ['fastMRI', 'JiangSu', 'lianying', 'IXI']  # ['pd', 'pdfs', 'JiangSu', 'lianying', 'IXI']  ['IXI_in_5']  ['BraTS2019_in_5']
_C.DATASET.PATTERN =     ['pdfs', 'T2', 'T2', 'T2'] # ['flair', 't1ce', 't2', 't1']  # 'pd'  'pdfs'  'pd+pdfs'  'T1'  'T2'  'T1+T2'  ['flair+T1ce+T2+T1']
_C.DATASET.SAMPLE_RATE = [1, 1, 1, 0.2]

_C.DATASET.CHALLENGE = 'singlecoil'  # the task of ours, singlecoil or multicoil
_C.DATASET.MODE = 'train'  # train or test

# Federated learning configs
_C.FL = CN()
_C.FL.CLIENTS_NUM = 2
_C.FL.MODEL_NAME = 'v3_FedMRI_fastMRI2C_encoder_1Ca3_2'
_C.FL.SHARE_WAY = 'only_encoder'  # 'whole_archi' or 'only_encoder' or 'except_last'
_C.FL.DATAMIX = False # 'pd'
_C.FL.SHOW_SIZE = True

_C.TRANSFORMS = CN()
_C.TRANSFORMS.MASKTYPE = 'random'  # "random" or "equispaced"
_C.TRANSFORMS.CENTER_FRACTIONS = [0.08]
_C.TRANSFORMS.ACCELERATIONS = [4]
_C.TRANSFORMS.MASK_DIR = '/home/jc3/Data/maskyzy'
_C.TRANSFORMS.MASK_SPEC = '1D-Cartesian-3X'
# _C.TRANSFORMS.MASK = ''


# model config
_C.MODEL = CN()
_C.MODEL.INPUT_DIM = 1   # the channel of input
_C.MODEL.OUTPUT_DIM = 1   # the channel of output
_C.MODEL.PATCH_SIZE = 16

_C.MULTI = CN()
_C.MULTI.MODE = 'base'       # 'concat' means concat image direct, 'base' means single

# the solver config

_C.SOLVER = CN()
_C.SOLVER.DEVICE = 'cuda'
_C.SOLVER.DEVICE_IDS = [0, 1]  # if [] use cpu, else gpu
_C.SOLVER.LR = 1e-4
_C.SOLVER.WEIGHT_DECAY = 1e-4
_C.SOLVER.LR_DROP = 80
_C.SOLVER.BATCH_SIZE = 8
_C.SOLVER.NUM_WORKERS = 1
_C.SOLVER.PRINT_FREQ = 10
_C.SOLVER.LR_GAMMA = 0.1

# the others config
_C.RESUME = ''  # model resume path
_C.OUTPUTDIR = './checkpints'  # the model output dir
_C.LOGDIR = './logs'

#the train configs
_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 50  # the train epochs
