import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)
