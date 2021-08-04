import logging
from pathlib import Path

REPO_DIR = Path(__file__).parents[1].absolute()

def setup_custom_logger(name):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    fh = logging.FileHandler(REPO_DIR / 'log' / 'rg_with_api.log')
    fh.setFormatter(formatter)
    # fh.setLevel(logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(fh)

    return logger

