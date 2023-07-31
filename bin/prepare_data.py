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

def load_single_set(paths, directory):
    cube = []
    for path in paths[:-1]:
        with fits.open(os.path.join(directory, path)) as hdul:
            cube.append(hdul[1].data)
    with fits.open(os.path.join(directory, paths[-1])) as hdul:
            thmap = cv2.resize(hdul[0].data, 
                               dsize=(THMAP_SIZE, THMAP_SIZE), 
                               interpolation=cv2.INTER_NEAREST)
    return np.array(cube), thmap
    

def load_cube(index, data_path, start, end):
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

def get_thresholds(cube, low_pct, high_pct, channel):
    return (np.nanpercentile(cube[:, channel, :, :], low_pct), 
            np.nanpercentile(cube[:, channel, :, :], high_pct))
    
def scale_data(cube, lows, highs):
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
    
    valid_start = train_end + 1
    valid_end = int(round(n * (TRAIN_PCT + VALID_PCT)))
    
    test_start = valid_end + 1
    test_end = n
        
    # load the data and scale it for each kind
    train_x, train_y = load_cube(index, args.data_path, train_start, train_end)
    cube = np.sign(train_x) * np.power(np.abs(train_x), SCALE_POWER)
    
    lows, highs = [], []
    for channel in range(6):
        low, high = get_thresholds(cube, LOW_PCT, HIGH_PCT, channel)
        lows.append(low)
        highs.append(high)
        
    train_x = scale_data(train_x, lows, highs)
    
    valid_x, valid_y = load_cube(index, args.data_path, valid_start, valid_end)
    valid_x = scale_data(valid_x, lows, highs)
    
    test_x, text_y = load_cube(index, args.data_path, test_start, test_end)
    test_x = scale_data(test_x, lows, highs)

    # save everything!     
    zarr.save("/d0/mhughes/thmap_suvi_train_x.zarr", train_x)
    zarr.save("/d0/mhughes/thmap_suvi_train_y.zarr", train_y)

    zarr.save("/d0/mhughes/thmap_suvi_valid_x.zarr", train_x)
    zarr.save("/d0/mhughes/thmap_suvi_valid_y.zarr", train_y)
    
    zarr.save("/d0/mhughes/thmap_suvi_test_x.zarr", train_x)
    zarr.save("/d0/mhughes/thmap_suvi_test_y.zarr", train_y)

