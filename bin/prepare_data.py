import os
import argparse

from astropy.io import fits
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import zarr

TRAIN_PCT = 0.6
VALID_PCT = 0.2
TEST_PCT = 0.2

SCALE_POWER = 0.25
LOW_PCT = 25
HIGH_PCT = 99.99

THMAP_SIZE = 256

def load_single_set(paths:list(str), directory:str)->(np.ndarray, np.ndarray):
    """Loads all channels and thematic for a single time

    Args:
        paths (list[str]): filenames to be loaded in order of 
            94, 131, 171, 195, 284, 304, thematic map
        directory (str): path that contains the data

    Returns:
        tuple[np.ndarray, np.ndarray]: 
            first element is a cube of the SUVI observations and second is the thematic map
            shapes are (6, 1280, 1280) and (256, 256)
    """    
    cube = []
    for path in paths[:-1]:
        with fits.open(os.path.join(directory, path)) as hdul:
            cube.append(hdul[1].data)
    with fits.open(os.path.join(directory, paths[-1])) as hdul:
            thmap = cv2.resize(hdul[0].data, 
                               dsize=(THMAP_SIZE, THMAP_SIZE), 
                               interpolation=cv2.INTER_NEAREST)
    return np.array(cube), thmap
    

def load_all_sets(index: pd.DataFrame, 
                  data_path: str, 
                  start: int, 
                  end: int) -> (np.ndarray, np.ndarray):
    """Loads all sets and thematic maps defined in an index dataframe 
        that are between start and end indices

    Args:
        index (pd.DataFrame): dataframe that links the different channel files together
        data_path (str): directory that contains the data
        start (int): end index
        end (int): end index

    Returns:
        tuple[np.ndarray, np.ndarray]: 
            first element is a cube of the SUVI observations and second is the thematic map
            shapes are (N, 6, 1280, 1280) and (N, 256, 256) where N is the image set count
    """    
    sub_index = index.loc[start:end]
    x, y = [], []
    
    for _, paths in tqdm(sub_index.iterrows(), total=len(sub_index)):
        try:
            cube, thmap = load_single_set(paths, data_path)
            if cube.shape == (6, 1280, 1280) and thmap.shape == (THMAP_SIZE, THMAP_SIZE):
                x.append(cube)
                y.append(thmap)
        except ValueError:
            pass
         
    return np.array(x), np.array(y)

def get_thresholds(cube: np.ndarray, 
                   low_pct: float, 
                   high_pct: float, 
                   channel: int) -> (float, float):
    """Given low and high percentiles, determines the corresponding values 
        in an image cube for a given channel

    Args:
        cube (np.ndarray): cube of observations returned from `load_all_sets`
        low_pct (float): low percentile
        high_pct (float): high percentile
        channel (_type_): index of the channel

    Returns:
        tuple[float, float]:
            low percentile and high percentile value respectively
    """    
    return (np.nanpercentile(cube[:, channel, :, :], low_pct), 
            np.nanpercentile(cube[:, channel, :, :], high_pct))
    
def scale_data(cube: np.ndarray, lows: [float], highs: [float]) -> np.ndarray:
    """Scales data to the 0.25 power, linearizes between 0 and 255, and clips

    Args:
        cube (np.ndarray): cube of observations returned from `load_all_sets`
        lows (list[floats]): low percentile values for each channel
        highs (list[floats]): high percentile values for each channel

    Returns:
        np.ndarray: 
            scaled data cube
    """    
    cube = np.sign(cube) * np.power(np.abs(cube), SCALE_POWER)
        
    for i, (low, high) in enumerate(zip(lows, highs)):
        cube[:, i, :, :] = np.clip((cube[:, i, :, :] - low) / (high - low) * 255, 
                                   0, 255)
    return cube.astype(np.uint8)

if __name__ == "__main__":
    # parse inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("index_path", help="path to the file index")
    parser.add_argument("data_path", help="path to data parent folder")
    args = parser.parse_args()
    
    index = pd.read_csv(args.index_path, index_col=0)
    
    # setup the start and ends
    n = len(index)
    train_start = 0
    train_end = int(round(n * TRAIN_PCT))
    
    valid_start = train_end
    valid_end = int(round(n * (TRAIN_PCT + VALID_PCT)))
    
    test_start = valid_end
    test_end = n
        
    # load the data and scale it for each kind
    train_x, train_y = load_all_sets(index, args.data_path, train_start, train_end)
    cube = np.sign(train_x) * np.power(np.abs(train_x), SCALE_POWER)
    
    lows, highs = [], []
    for channel in range(6):
        low, high = get_thresholds(cube, LOW_PCT, HIGH_PCT, channel)
        lows.append(low)
        highs.append(high)
        
    train_x = scale_data(train_x, lows, highs)
    
    valid_x, valid_y = load_all_sets(index, args.data_path, valid_start, valid_end)
    valid_x = scale_data(valid_x, lows, highs)
    
    test_x, test_y = load_all_sets(index, args.data_path, test_start, test_end)
    test_x = scale_data(test_x, lows, highs)

    # save everything!     
    zarr.save("/d0/mhughes/thmap_suvi_train_x.zarr", train_x)
    zarr.save("/d0/mhughes/thmap_suvi_train_y.zarr", train_y)

    zarr.save("/d0/mhughes/thmap_suvi_valid_x.zarr", valid_x)
    zarr.save("/d0/mhughes/thmap_suvi_valid_y.zarr", valid_y)
    
    zarr.save("/d0/mhughes/thmap_suvi_test_x.zarr", test_x)
    zarr.save("/d0/mhughes/thmap_suvi_test_y.zarr", test_y)

