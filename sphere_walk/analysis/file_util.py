import numpy as np
import glob
from natsort import natsorted

def get_all_files(path, ext='parquet'):
    return natsorted(glob.glob(f'{path}/*.{ext}'))

