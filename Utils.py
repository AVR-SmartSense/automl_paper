""" Helper script to load image (x) and target (y) dataset """

# Import libraries
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path


# Functions to load dataset
def load_image(img_file):
    """
    Loads image file, resize to 128 x 128 and return as numpy array
    :param img_file: path to image file in .TIFF format
    :return: numpy array (128, 128, 3)
    """
    img = Image.open(img_file)
    img = img.resize((128, 128))
    img = np.array(img)
    return img


def load_dataset(data_dir, task='classification'):
    """
    Loads x and y dataset for classification or regression
    :param data_dir: path to data folder containing images and CSV file
    :param task: 'classification' or 'regression', default is 'classification'
    :return: x and y dataset
    """
    # loads x data, images sorted according to plot number
    img_list = [load_image(file) for file in sorted(Path(data_dir).glob('*.tiff'),
                                                    key=lambda x: int(x.stem.split('_')[1]))]
    x = np.stack(img_list)
    # loads y data
    csv_file = Path(data_dir) / 'Lodging_data.csv'
    df = pd.read_csv(csv_file)
    if task == 'regression':
        df = df[df['Lodging'] == 'Yes']  # select rows for only lodged plots
        y = df['Score']
        x = [x[i] for i in df.index]  # slice x for only lodged plots
        x = np.array(x)
    else:
        y = df['Lodging']
    return x, y