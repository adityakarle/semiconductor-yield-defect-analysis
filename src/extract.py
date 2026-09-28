"""Extract stage: pull the real UCI SECOM sensor and label files.

Downloads are cached locally under data/raw/ — re-running the pipeline
without --refresh skips files that already exist.
"""
import logging

import requests

from src.config import RAW_DIR, SOURCES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def download_file(url: str, dest, refresh: bool = False) -> None:
    if dest.exists() and not refresh:
        logger.info("Cached, skipping download: %s", dest.name)
        return

    logger.info("Downloading %s ...", dest.name)
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    dest.write_bytes(response.content)
    logger.info("Saved %s (%.1f MB)", dest.name, len(response.content) / 1_000_000)


def extract_all(refresh: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, source in SOURCES.items():
        download_file(source["url"], source["file"], refresh=refresh)


if __name__ == "__main__":
    extract_all()
