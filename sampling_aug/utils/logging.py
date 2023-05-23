import logging

if 'logger' not in locals():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
