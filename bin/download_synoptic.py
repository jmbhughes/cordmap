from datetime import datetime, timedelta
from dask import compute, delayed
from dask.diagnostics import ProgressBar
import humanfriendly
import urllib.request
import os
from bs4 import BeautifulSoup


def format_url(date) -> str:
    return f"https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-imagery/composites/full-sun-drawings/boulder/{date.strftime('%Y/%m')}/"


def fetch_page(url: str):
    try:
        def split_entry(entry):
            filename = entry.find_all("td")[0].text
            try:
                size = humanfriendly.parse_size(entry.find_all("td")[2].text)
            except humanfriendly.InvalidSize:
                return None
            return {"url": url + filename,
                    "dt": datetime.strptime("_".join(filename.split(".jpg")[0].split("_")[-2:]), "%Y%m%d_%H%M"),
                    "size": size}

        with urllib.request.urlopen(url) as response:
            html = response.read()
        soup = BeautifulSoup(html, 'html.parser')
        entries = soup.find_all('tr')[3:-1]
        results = list(map(split_entry, entries))
        return results
    except urllib.request.HTTPError:
        return []


def download_all(page_results, download_dir):
    for line in page_results:
        if line is not None:
            download_path = os.path.join(download_dir, line['url'].split("/")[-1])
            urllib.request.urlretrieve(line['url'], os.path.join(download_dir, line['url'].split("/")[-1]))


def date_range(start, end):
    # Truncate the hours, minutes, and seconds
    sdate = datetime(start.year, start.month, start.day)
    edate = datetime(end.year, end.month, end.day)

    # compute all dates in that difference
    delta: timedelta = edate - sdate
    return [start + timedelta(days=i) for i in range(delta.days + 1)]


def fetch(dt, save_dir="/Users/jhughes/Desktop/data/synopticmaps"):
    download_all(fetch_page(format_url(dt)), save_dir)


if __name__ == "__main__":
    dates = date_range(datetime(2023, 1, 1), datetime(2023, 7, 23))
    delayed_results = [delayed(fetch)(dt) for dt in dates]
    ProgressBar().register()
    _ = compute(*delayed_results)
