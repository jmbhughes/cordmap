from datetime import datetime, timedelta
from typing import List
from dask import compute, delayed
from dask.diagnostics import ProgressBar
import humanfriendly
import urllib.request
import os
from bs4 import BeautifulSoup


def format_url(date: datetime) -> str:
    """Creates a formatted URL to the folder containing that month's synoptic maps

    Args:
        date (datetime): what date you wish to retrieve for

    Returns:
        str: a URL that can be used to access that corresponding month's maps
    """
    return f"https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-imagery/composites/full-sun-drawings/boulder/{date.strftime('%Y/%m')}/"


def fetch_page(url: str) -> List[dict]:
    """Parses a given page's observations 
        and extracts all download links for maps in that month

    Args:
        url (str): page to fetch

    Returns:
        List[Dict]: List of result dicts for that month. 
            Keys are `url`, `dt` (for dateeitme), and `size` for how big the file is
    """
    try:
        def split_entry(entry):
            filename = entry.find_all("td")[0].text
            try:
                size = humanfriendly.parse_size(entry.find_all("td")[2].text)
            except humanfriendly.InvalidSize:
                return None
            return {"url": url + filename,
                    "dt": datetime.strptime(
                        "_".join(filename.split(".jpg")[0].split("_")[-2:]), 
                        "%Y%m%d_%H%M"),
                    "size": size}

        with urllib.request.urlopen(url) as response:
            html = response.read()
        soup = BeautifulSoup(html, 'html.parser')
        entries = soup.find_all('tr')[3:-1]
        results = list(map(split_entry, entries))
        return results
    except urllib.request.HTTPError:
        return []


def download_all(page_results: List[dict], download_dir: str):
    """Downloads all maps for a given page into `download_dir`

    Args:
        page_results (List[dict]): output of `fetch_page`
        download_dir (str): where to save the maps
    """
    for line in page_results:
        if line is not None:
            urllib.request.urlretrieve(line['url'], 
                                       os.path.join(download_dir, 
                                                    line['url'].split("/")[-1]))


def generate_days_between(start: datetime, end: datetime) -> List[datetime]:
    """Generate a list of all days between `start` and `end`

    Args:
        start (datetime): the initial date
        end (datetime): the final date

    Returns:
        List[datetime]: all days between `start` and `end` inclusive
    """
    # Truncate the hours, minutes, and seconds
    sdate = datetime(start.year, start.month, start.day)
    edate = datetime(end.year, end.month, end.day)

    # compute all dates in that difference
    delta: timedelta = edate - sdate
    return [start + timedelta(days=i) for i in range(delta.days + 1)]


def fetch(dt, save_dir="/Users/jhughes/Desktop/data/synopticmaps"):
    download_all(fetch_page(format_url(dt)), save_dir)


if __name__ == "__main__":
    dates = generate_days_between(datetime(2023, 1, 1), datetime(2023, 7, 23))
    delayed_results = [delayed(fetch)(dt) for dt in dates]
    ProgressBar().register()
    _ = compute(*delayed_results)
