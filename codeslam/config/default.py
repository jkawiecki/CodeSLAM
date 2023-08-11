from yacs.config import CfgNode as CN

cfg = CN()

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
cfg.INPUT = CN()
cfg.INPUT.IMAGE_SIZE = [192, 256]                                    # HxW
cfg.INPUT.IMAGE_CHANNELS = 3 #1
cfg.INPUT.PIXEL_MEAN = [111.33145032, 106.68657402,  99.29848617]
cfg.INPUT.PIXEL_STD = [65.31386607, 64.87560227, 65.97477225]
cfg.INPUT.RGB2GRAY = False

# -----------------------------------------------------------------------------
# OUTPUT
# -----------------------------------------------------------------------------
cfg.OUTPUT = CN()
cfg.OUTPUT.IMAGE_SIZE = [192, 256]                                   # HxW
cfg.OUTPUT.CHANNELS = 1
cfg.OUTPUT.PIXEL_MEAN = 3728.1855890148872
cfg.OUTPUT.PIXEL_STD = 2435.2189247928313
cfg.OUTPUT.PIXEL_MAX = 65535


# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
cfg.MODEL = CN()
cfg.MODEL.NAME = "codeslam"
cfg.MODEL.DEVICE = "cpu"            # used during eval I guess?
cfg.MODEL.NUM_OUTPUT_SCALES = 1     # No more pyramids
cfg.MODEL.PRETRAINED = False        # Whole model is pretrained
cfg.MODEL.PRETRAINED_PATH = ""      # Path to pretrained model



### UNET ###
cfg.MODEL.UNET = CN()
cfg.MODEL.UNET.USE_SPARSE = False
cfg.MODEL.UNET.ENCODER = CN()
cfg.MODEL.UNET.ENCODER.PRETRAINED = False

### CVAE ###
cfg.MODEL.CVAE = CN()

cfg.MODEL.CVAE.LATENT = CN()
cfg.MODEL.CVAE.LATENT.DIMENSIONS = 32               # Dimension of mean vector
cfg.MODEL.CVAE.LATENT.INPUT_DIM = [6, 8]            # Size of feature map (HxW) before latent code

# Not sure what this does quite yet
cfg.MODEL.CVAE.CONDITION = CN()
cfg.MODEL.CVAE.CONDITION.DIMENSIONS = 1


# -----------------------------------------------------------------------------
# TRAINER
# -----------------------------------------------------------------------------
cfg.TRAINER = CN()
cfg.TRAINER.EPOCHS = 4                 # Number of epochs
cfg.TRAINER.MAX_ITER = 1000           # Set EPOCHS=1 to train for MAX_ITER number of iterations instead
cfg.TRAINER.BATCH_SIZE = 16
cfg.TRAINER.SAVE_STEP = -1              # Save model periodically during training (-1 to not save)
cfg.TRAINER.LOG_STEP = 10               # Log metrics periodically during training (-1 to not log)
cfg.TRAINER.EVAL_STEP = -1              # Evaluate on validation set during training (-1 to not evaluate)
cfg.TRAINER.KL_MILESTONE = 0            # Epoch to activate KL loss in loss function

cfg.TRAINER.OPTIMIZER = CN()
cfg.TRAINER.OPTIMIZER.TYPE = "adam"
cfg.TRAINER.OPTIMIZER.LR = 3e-4
cfg.TRAINER.OPTIMIZER.WEIGHT_DECAY = 0.0
cfg.TRAINER.OPTIMIZER.MOMENTUM = 0.9
cfg.TRAINER.OPTIMIZER.MILESTONES = []       # List of milestones (epochs) to decrease learning rate
cfg.TRAINER.OPTIMIZER.GAMMA = 1.0           # Factor to decrease learning rate with at each milestone
cfg.TRAINER.OPTIMIZER.WARMUP_PERIOD = 0.0   # Which step to end learning rate warmup



# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------
cfg.DATASETS = CN()
cfg.DATASETS.TRAIN = ()                 # listed in path catalog
cfg.DATASETS.TEST = ()                  # listed in path catalog
cfg.DATASETS.DATASET_DIR = "datasets"

# -----------------------------------------------------------------------------
# DATALOADER
# -----------------------------------------------------------------------------
cfg.DATA_LOADER = CN()
# Number of data loading threads
cfg.DATA_LOADER.NUM_WORKERS = 4
cfg.DATA_LOADER.PIN_MEMORY = True


# -----------------------------------------------------------------------------
# OUTPUT DIRECTORY
# -----------------------------------------------------------------------------
cfg.OUTPUT_DIR = "outputs"
cfg.OUTPUT_DIR_MODEL = "outputs/model"
cfg.PRETRAINED_WEIGHTS = ""

# ---------------------------------------------------------------------------- #
# TEST CONFIGURATION
# ---------------------------------------------------------------------------- #
cfg.TEST = CN()
cfg.TEST.BATCH_SIZE = 1

# -----------------------------------------------------------------------------
# DEMO CONFIGURATION
# -----------------------------------------------------------------------------
cfg.DEMO_RGB_PATH = ""
cfg.DEMO_DEPTH_PATH = ""