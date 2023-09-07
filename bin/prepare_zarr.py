import zarr
from numcodecs import Blosc
from astropy.io import fits
from glob import glob
import pandas as pd
from datetime import datetime, timedelta
import os
import numpy as np
from tqdm import tqdm
import dask
from multiprocessing.pool import Pool

KEYS_TO_KEEP = [
'degraded',
'date-obs',
'date-beg',
'date-end',
'eclipse',
'ctype1', 'cunit1', 'crval1', 'cdelt1', 'crpix1',  # dimension 1 properties
'ctype2', 'cunit2', 'crval2', 'cdelt2', 'crpix2',  # dimension 2 propertie
'pc1_1', 'pc1_2', 'pc2_1', 'pc2_2', # Detector rotation matrix
'diam_sun',
'dsun_obs',
'solar_b0',
'lonpole',
'crota',
'num_imgs',
'instrume'
]

if __name__ == "__main__":
    filenames = sorted(glob("/d0/mhughes/thmap_suvi_hourly/*.fits"))
    cadence = timedelta(hours=1)
    cadence_epsilon = timedelta(minutes=5)
    cadence_window = (cadence - cadence_epsilon, cadence + cadence_epsilon)

    DATETIME_STR_FORMAT =  "s%Y%m%dT%H%M%SZ"

    product_filenames = dict()
    product_datetimes = dict()
    for kind in ["ci094", "ci131", "ci171", "ci195", "ci284", "ci304", "thmap"]:
        product_filenames[kind] = [fn for fn in filenames if f"suvi-l2-{kind}" in fn]
        product_datetimes[kind] = np.array([datetime.strptime(os.path.basename(fn).split("_")[3], DATETIME_STR_FORMAT) 
                                            for fn in product_filenames[kind]])
    #product_filenames = {k: set(v) for k, v in product_filenames.items()}
    product_datetimes_set = {k: set(v) for k, v in product_datetimes.items()}
    
    groups = []
    for fn, dt in zip(product_filenames['ci094'], product_datetimes['ci094']):
        all_exist = all(dt in product_datetimes_set[other_kind] for other_kind in ["ci131", "ci171", "ci195", "ci284", "ci304", "thmap"])
        if all_exist:
            group = {'dt': dt, "ci094": fn}
            for other_product_kind in ["ci131", "ci171", "ci195", "ci284", "ci304", "thmap"]:
                for version in ["v1-0-1.fits", "v1-0-2.fits", "v1-0-3.fits"]:
                    candidate_filename = fn.replace("ci094", other_product_kind)[:-11] + version
                    if os.path.isfile(candidate_filename):
                        group[other_product_kind] = candidate_filename
            if len(group) == 8:
                groups.append(group)
                
    df = pd.DataFrame(groups)
    df.to_csv("/d0/mhughes/thmap_suvi_hourly.csv")
    
    time_chunk_size = 3
    channel_chunk_size = 7

    store = zarr.DirectoryStore("/d0/mhughes/thmap_suvi_hourly.zarr")
    compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
    root = zarr.group(store=store, overwrite=True)
    data_group = root.create_dataset("data", 
                                shape=(len(df), 7, 1280, 1280), 
                                chunks=(time_chunk_size, channel_chunk_size, None, None), 
                                dtype='f4',
                                compressor=compressor)

    saved_meta = {k: [] for k in KEYS_TO_KEEP}
    
    def load_stack(row):
        for channel, kind in enumerate(["ci094", "ci131", "ci171", "ci195", 
                                        "ci284", "ci304", "thmap"]):
            try:
                with fits.open(row[1][kind]) as hdul:
                    if kind == "thmap":
                        img_data = hdul[0].data.astype(float)
                        head = hdul[0].header
                    else:
                        img_data = hdul[1].data
                        head = hdul[1].header
            except Exception as e:
                print(e)
            else:
                if img_data is not None:
                    data_group[row[0], channel, :, :] = img_data
                    for key in KEYS_TO_KEEP:
                        saved_meta[key].append(head[key.upper()])
        return True

    with Pool(32) as p:
        for _ in tqdm(p.imap_unordered(load_stack, df.iterrows()), total=len(df)):
            pass
        p.close()
        p.join()
        
    for key in KEYS_TO_KEEP:
        data_group.attrs[key] = saved_meta[key]
            
    data_group.attrs['_ARRAY_DIMENSIONS'] = ['t_obs', 'channel', 'x', 'y']

    t_obs = root.create_dataset('t_obs', 
                                shape=(len(df)), 
                                chunks=(None), 
                                dtype='M8[ns]',
                                compressor=None) 
    t_obs[:] = list(df['dt'])
    t_obs.attrs['_ARRAY_DIMENSIONS'] = ['t_obs']

    channels = root.create_dataset('channel', 
                                shape=(7), 
                                chunks=(None), 
                                dtype=str,
                                compressor=None)
    channels[:] =['ci094', 'ci131', 'ci171', 'ci195', 'ci284', 'ci304', 'thmap']
    channels.attrs['_ARRAY_DIMENSIONS'] = ['channel']
        
    # zarr.consolidate_metadata(store)       
