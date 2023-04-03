import os
import logging

logger = logging.getLogger(__name__)


def load_data(dir_path, test):
    """
    Load a corpus from directory and concat into a single object for processing.
    """
    paths = os.listdir(dir_path)
    files = []
    for path in paths:
        with open(dir_path + path, "r") as file:
            files.append(file.read().lower())

    logger.info("Concatenating dataset into a single string for easy handling.")
    dataset = ""
    length_dataset = len(files) if not test else 3
    for file in files[:length_dataset]:
        dataset += file

    return dataset
