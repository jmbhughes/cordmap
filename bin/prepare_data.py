import os
import argparse
from typing import List, Tuple

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

EUV_CHANNEL_ORDER = [94, 131, 171, 195, 284, 304]

def load_single_set(paths: List[str], directory: str) -> (np.ndarray, np.ndarray):
    """Loads all channels and thematic for a single time

    Args:
        paths (list[str]): filenames to be loaded in order of 
            94, 131, 171, 195, 284, 304, thematic map
        directory (str): path that contains the data

    Returns:
        tuple[np.ndarray, np.ndarray, dict]: 
            first element is a cube of the SUVI observations, shape (6, 1280, 1280)
            second element is the thematic map, shape (256, 256)
            third element is meta data for both the cube and thematic map 
    """    
    meta = {}
    cube = []
    # open the SUVI EUV composites
    for channel, path in zip(EUV_CHANNEL_ORDER, paths[:-1]):
        with fits.open(os.path.join(directory, path)) as hdul:
            head = hdul[1].header
            cube.append(hdul[1].data)
            meta[f"ci_{channel}_file_name"] = path
            meta[f"ci_{channel}_min"] = head.get('IMG_MIN', -1)
            meta[f"ci_{channel}_mean"] = head.get('IMG_MEAN', -1)
            meta[f"ci_{channel}_total_irradiance"] = head.get('IMGTII', -1)
            meta[f"ci_{channel}_total_radiance"] = head.get("IMGTIR", -1)
            meta[f"ci_{channel}_stdev"] = head.get("IMG_SDEV", -1)
            meta[f"ci_{channel}_source_count"] = head.get("NUM_IMGS", -1)
    
    # open the thematic map
    thmap_path = paths[-1]
    with fits.open(os.path.join(directory, thmap_path)) as hdul:
        head = hdul[0].header
        thmap = cv2.resize(hdul[0].data, 
                            dsize=(THMAP_SIZE, THMAP_SIZE), 
                            interpolation=cv2.INTER_NEAREST)
        meta["thmap_file_name"] = thmap_path
        meta["thmap_num_inputs"] = head.get("NUM_IMGS", -1)
        meta["thmap_algorithm"] = head.get("ALGORTHM", "unspecified")
        meta["thmap_model"] = head.get("TRAINING", "unspecified")
        
    return np.array(cube), thmap, meta
    
def data_is_good(cube, thmap, meta):
    shapes_good = cube.shape == (6, 1280, 1280) and thmap.shape == (THMAP_SIZE, THMAP_SIZE)
    thmap_uses_six_imgs = meta['thmap_num_inputs'] == 6
    
    composites_good = meta["ci_94_source_count"] == 6
    composites_good &= meta["ci_131_source_count"] == 3
    composites_good &= meta["ci_171_source_count"] == 2
    composites_good &= meta["ci_195_source_count"] == 7
    composites_good &= meta["ci_284_source_count"] == 2
    composites_good &= meta["ci_304_source_count"] == 4
    
    return shapes_good and thmap_uses_six_imgs and composites_good


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
            first element is a cube of the SUVI observations shape=(N, 6, 1280, 1280)
            second is the thematic map shape=(N, 256, 256)
            third is a dataframe of metadata
    """    
    sub_index = index.loc[start:end]
    x, y, all_meta = [], [], []
    
    for _, paths in tqdm(sub_index.iterrows(), total=len(sub_index)):
        try:
            cube, thmap, meta = load_single_set(paths, data_path)
            if data_is_good(cube, thmap, meta):
                x.append(cube)
                y.append(thmap)
                all_meta.append(meta)
        except ValueError:
            pass
         
    return np.array(x), np.array(y), pd.DataFrame(all_meta)

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
    parser.add_argument("output_path", help="where to save the zarr files")
    args = parser.parse_args()
    
    if not os.path.isdir(args.output_path):
        raise RuntimeError("Specified output path does not exist.")
    
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
    train_x, train_y, train_meta = load_all_sets(index, args.data_path, 
                                                 train_start, train_end)
    cube = np.sign(train_x) * np.power(np.abs(train_x), SCALE_POWER)
    
    lows, highs = [], []
    for channel in range(6):
        low, high = get_thresholds(cube, LOW_PCT, HIGH_PCT, channel)
        lows.append(low)
        highs.append(high)
        
    train_x = scale_data(train_x, lows, highs)
    
    valid_x, valid_y, valid_meta = load_all_sets(index, args.data_path, 
                                                 valid_start, valid_end)
    valid_x = scale_data(valid_x, lows, highs)
    
    test_x, test_y, test_meta = load_all_sets(index, args.data_path, 
                                              test_start, test_end)
    test_x = scale_data(test_x, lows, highs)

    # save everything!     
    with open(os.path.join(args.output_path, "thmap_suvi_lows_highs.txt"), "w") as f:
        low_str = f"{lows[0]} {lows[1]} {lows[2]} {lows[3]} {lows[4]} {lows[5]}"
        high_str = f"{highs[0]} {highs[1]} {highs[2]} {highs[3]} {highs[4]} {highs[5]}"
        f.write(low_str + "\n" + high_str)
    
    zarr.save(os.path.join(args.output_path, "thmap_suvi_train_x.zarr"), train_x)
    zarr.save(os.path.join(args.output_path, "thmap_suvi_train_y.zarr"), train_y)
    train_meta.to_csv(os.path.join(args.output_path, "thmap_suvi_train.csv"))

    zarr.save(os.path.join(args.output_path, "thmap_suvi_valid_x.zarr"), valid_x)
    zarr.save(os.path.join(args.output_path, "thmap_suvi_valid_y.zarr"), valid_y)
    valid_meta.to_csv(os.path.join(args.output_path, "thmap_suvi_valid.csv"))
    
    zarr.save(os.path.join(args.output_path, "thmap_suvi_test_x.zarr"), test_x)
    zarr.save(os.path.join(args.output_path, "thmap_suvi_test_y.zarr"), test_y)
    test_meta.to_csv(os.path.join(args.output_path, "thmap_suvi_test.csv"))
