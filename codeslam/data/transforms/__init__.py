from codeslam.data.transforms.transforms import *

def build_transforms(cfg, is_train=True):
    if is_train:
        transform = [
            Resize(cfg.INPUT.IMAGE_SIZE),
            Normalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD),
            RGBToGrayscale() if cfg.INPUT.RGB2GRAY else Identity(),
            Proximity(cfg.OUTPUT.PIXEL_MEAN),
            ToTensor(),
        ]
    else:
        transform = [
            Resize(cfg.INPUT.IMAGE_SIZE),
            Normalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD),
            RGBToGrayscale() if cfg.INPUT.RGB2GRAY else Identity(),
            Proximity(cfg.OUTPUT.PIXEL_MEAN),
            ToTensor()
        ]
        
    transform = Compose(transform)
    return transform
