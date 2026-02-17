import logging
import os


def setup_logging(log_file: str = "train.log", level=logging.INFO):
    logging.basicConfig(
        level=level,
        force=True,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="w")
        ]
    )

    # Optional: log device info right after setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"PyTorch {torch.__version__} | Device: {device}")
    if device.type == "cuda":
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"Initial VRAM: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    else:
        logging.warning("No GPU detected. Training will be slow.")