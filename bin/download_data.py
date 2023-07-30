from typing import List, Tuple, Dict
from datetime import datetime, timedelta
import os
import argparse
import pandas as pd
from goessolarretriever import Retriever, Satellite, Product
from dateutil.parser import parse as parse_str_to_datetime


def generate_spaced_times(start: datetime, end: datetime, count: int) -> List[datetime]:
    step = (end - start) / count
    return [start + i * step for i in range(count)]


def download_single_date(dt: datetime,
                         destination: str,
                         satellite: Satellite = Satellite.GOES16,
                         products: Tuple[Product] = (Product.suvi_l2_ci094,
                                                     Product.suvi_l2_ci131,
                                                     Product.suvi_l2_ci171,
                                                     Product.suvi_l2_ci195,
                                                     Product.suvi_l2_ci284,
                                                     Product.suvi_l2_ci304,
                                                     Product.suvi_l2_thmap)) -> Dict[Product, str]:
    r = Retriever()
    filenames = {}
    for product in products:
        try:
            filename = r.retrieve_nearest(satellite, product, dt, destination)
        except RuntimeError:
            filename = None
        filenames[product] = filename
    return filenames


def download_many_dates(start: datetime, end: datetime, count: int, destination: str) -> None:
    # Compute the dates to download
    dates = generate_spaced_times(start, end, count)

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

    df.to_csv(os.path.join(destination, "index.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("start", help="first date to retrieve data from")
    parser.add_argument("end", help="last date to retrieve data from")
    parser.add_argument("count", help="number of image sets to retrieve between start and end, uniformly spaced",
                        type=int)
    parser.add_argument("destination", help="where to store the images")
    args = parser.parse_args()

    start = parse_str_to_datetime(args.start)
    end = parse_str_to_datetime(args.end)
    if args.count <= 0:
        raise RuntimeError("Count must be an integer >0")

    download_many_dates(start, end, args.count, args.destination)



