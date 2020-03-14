import os
import pickle
from PIL import Image
from io import BytesIO
import numpy as np


def get_data():
    with open(os.path.join('data', 'data.sav'), 'rb') as file:
        data = pickle.load(file)
    with open(os.path.join('data', 'values.sav'), 'rb') as file:
        values = pickle.load(file)
    with open(os.path.join('data', 'true_values.sav'), 'rb') as file:
        true_values = pickle.load(file)

    return data, values, true_values

def _translate_image_to_numpy(jpg_bytes_array):
    pic = Image.open(BytesIO(jpg_bytes_array)).convert('L')
    pic.thubnail((128, 128), Image.ANTIALIAS)
    return np.array(np.array(pic))

