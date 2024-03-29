import logging

from codeslam.models import make_model
from codeslam.utils.parser import get_parser
from codeslam.config.default import cfg
from codeslam.trainer.inference import do_evaluation
from codeslam.utils.checkpointer import CheckPointer
from codeslam.utils.logger import setup_logger
from codeslam.utils import torch_utils


def evaluation(cfg):
    logger = logging.getLogger("CodeSLAM.inference")

    model = make_model(cfg)
    model = torch_utils.to_cuda(model)

    ckpt = cfg.PRETRAINED_WEIGHTS if len(cfg.PRETRAINED_WEIGHTS) > 0 else None
    if ckpt is None:
        raise RuntimeError("Specify file with model weights in config")

    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR, logger=logger)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()
    print('Loaded weights from {}'.format(weight_file))

    do_evaluation(cfg, model)


def main():
    args = get_parser().parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    logger = setup_logger("CodeSLAM", cfg.OUTPUT_DIR)
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    evaluation(cfg)


if __name__ == '__main__':
    main()