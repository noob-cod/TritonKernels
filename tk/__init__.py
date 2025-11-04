import os
import logging


__version__ = "0.1.0"


log_format = "%(asctime)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Create a logger for the package
logger = logging.getLogger(f"{__name__}Logger")
logger.setLevel(logging.INFO)

if os.getenv("TKDEBUG"):
    logger.setLevel(logging.DEBUG)

if os.getenv("TKSAVELOG"):
    # Create a formatter for log handler
    formatter = logging.Formatter(log_format)

    # Create a rotating file handler
    rotate_handler = logging.handlers.RotatingFileHandler(
        "triton_kernels_rotating.log", maxBytes=5 * 1024 * 1024, backupCount=2
    )
    rotate_handler.setLevel(logging.DEBUG)
    rotate_handler.setFormatter(formatter)

    logger.addHandler(rotate_handler)

from . import utils
