import logging

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s %(processName)-10s %(name)s - %(levelname)s: %(message)s")

logger = logging.getLogger("dataset.py")

logger.info(f'element 0')

if __name__ == "__main__":
    pass