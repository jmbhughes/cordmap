from typing import List, Tuple, Dict
from datetime import datetime, timedelta
import os
import argparse
import pandas as pd
from goessolarretriever import Retriever, Satellite, Product
from dateutil.parser import parse as parse_str_to_datetime

from cordmap.util import generate_spaced_times

ALL_SUVI_PRODUCTS = (Product.suvi_l2_ci094,
                     Product.suvi_l2_ci131,
                     Product.suvi_l2_ci171,
                     Product.suvi_l2_ci195,
                     Product.suvi_l2_ci284,
                     Product.suvi_l2_ci304,
                     Product.suvi_l2_thmap)



def download_single_date(dt: datetime,
                         destination: str,
                         satellite: Satellite = Satellite.GOES16,
                         products: Tuple[Product] = ALL_SUVI_PRODUCTS) -> Dict[Product, str]:
    """Downloads specified `products` from `satellite` 
        at given datetime `dt` to the `destination`

    Args:
        dt (`datetime`): observation time
        destination (str): where images will be saved
        satellite (Satellite, optional): which satellite to download. 
            Defaults to Satellite.GOES16.
        products (Tuple[Product], optional): which products to download.
            Defaults to ALL_SUVI_PRODUCTS.

    Returns:
        Dict[Product, str]: mapping of product to the string where it was saved
    """
    r = Retriever()
    filenames = {}
    for product in products:
        try:
            filename = r.retrieve_nearest(satellite, product, dt, destination)
        except RuntimeError:
            filename = None
        filenames[product] = filename
    return filenames


def download_many_dates(start: datetime, 
                        end: datetime, 
                        n: int, 
                        destination: str) -> None:
    """Downloads `n` SUVI observations between `start` and `end` 
        to `destination` with an index.csv to pair them up

    Args:
        start (datetime): first observation time requested
        end (datetime): last observation time requested
        n (int): how many observations to download
        destination (str): where to save the images
        
    Notes:
        If a date cannot be downloaded, the function proceeds. 
        May return fewer than `n` observations in this case
    """
    # Compute the dates to download
    dates = generate_spaced_times(start, end, n)

    # make the destination directory if it doesn't exist
    os.makedirs(destination, exist_ok=True)

    # download all the sets keeping track of their filenames
    results = []
    for dt in dates:
        try:
            result = download_single_date(dt, destination)
        except ValueError:
            pass
        else:
            results.append(result)

    # Keep only sets that have all their files
    valid_results = [s for s in results if all([e is not None for e in s.values()])]
    invalid_results = [s for s in results if any([e is None for e in s.values()])]
    for invalid_result in invalid_results:
        for filename in invalid_result.values():
            if filename is not None:
                os.remove(filename)
    
    # write out the index
    df = pd.DataFrame(valid_results)

    # remove all the absolute paths now that we no longer need them
    for col in df.columns:
        df[col] = df[col].map(lambda e: os.path.basename(e))

    # save the index
    df.to_csv(os.path.join(destination, "index.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("start", help="first date to retrieve data from")
    parser.add_argument("end", help="last date to retrieve data from")
    parser.add_argument("count", 
                        help="number of image sets to retrieve"
                        " uniformly between start and end",
                        type=int)
    parser.add_argument("destination", help="where to store the images")
    args = parser.parse_args()

    start = parse_str_to_datetime(args.start)
    end = parse_str_to_datetime(args.end)
    if args.count <= 0:
        raise RuntimeError("Count must be an integer >0")

    download_many_dates(start, end, args.count, args.destination)



