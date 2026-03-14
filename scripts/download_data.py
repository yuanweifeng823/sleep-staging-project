#!/usr/bin/env python
"""Script to download Sleep-EDFx dataset"""

import os
import sys
from pathlib import Path

# ensure src package can be imported when running script directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import tarfile
from tqdm import tqdm
from src.utils.logger import setup_logger
from src.utils.paths import paths

logger = setup_logger(__name__)

# Sleep-EDFx expanded dataset URLs
BASE_URL = "https://physionet.org/files/sleep-edfx/1.0.0/"
FILES = [
    "SC-subjects.xls",
    "ST-subjects.xls",
    "sleep-cassette/SC4001E0-PSG.edf",
    "sleep-cassette/SC4001EC-Hypnogram.edf",
    # Add more files as needed
]


def download_file(url: str, save_path: Path):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(save_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, 
                  desc=save_path.name) as pbar:
            for data in response.iter_content(chunk_size=1024):
                f.write(data)
                pbar.update(len(data))


def main():
    """Main download function"""
    data_dir = paths.data_raw
    data_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading Sleep-EDFx dataset to {data_dir}")
    
    for file_path in FILES:
        url = BASE_URL + file_path
        save_path = data_dir / file_path.split('/')[-1]
        
        if save_path.exists():
            logger.info(f"File {save_path} already exists, skipping...")
            continue
        
        logger.info(f"Downloading {file_path}...")
        download_file(url, save_path)
    
    logger.info("Download complete!")


if __name__ == "__main__":
    main()