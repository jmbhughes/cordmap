# cordmap
Tools for **CO**mprehensive **R**apid **D**igest maps for space weather. 

## 1. Preparing data

### a. Downloading and loading
First run the `bin/download_data.py` to download the data. 
Then, create previews by running `bin/preview_data.py`. Inspect each preview and delete ones that are bad, i.e.
they're missing a channel or the thematic map doesn't look reasonable. Finally, run `bin/select_data.py` to filter out
and keep only good image sets. This should result in a `good_index.csv` file in your `data` directory that can be used
to load the data. 

To load:
```python
from cordmap.data import SUVIImageDataset

d = SUVIImageDataset("../data/good_index.csv", "../data")
```

### b. Train/test splits


## 2. Fine-tuning SAM 


## Contact
This is a work in progress. For more information, contact Marcus Hughes at `marcus.hughes@swri.org`. 
