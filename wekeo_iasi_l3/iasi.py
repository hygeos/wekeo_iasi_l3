from pathlib import Path

from wekeo_iasi_l3 import config
from wekeo_iasi_l3.download import download_IASI_products
from wekeo_iasi_l3.reader_l2.reader import read_iasi_l2
from wekeo_iasi_l3.global_accumulator import GlobalAccumulator2D

import xarray as xr

from wekeo_iasi_l3.hygeos_core import log
from wekeo_iasi_l3.hygeos_core import env
from datetime import date

def  download_iasi_day_files(day: date) -> list[Path]:
    """Download and Compile multiple FRP files into a single xarray Dataset."""
    
    log.info(f"Downloading FRP products for date: {day}")
    
    eday = day.strftime("%Y-%m-%d")
    sday = day.strftime("%Y-%m-%d") # same day for single day query

    files = download_IASI_products(
        start_date = sday,
        end_date = eday,
    )
    
    return files

def get_gridded_iasi_l3(
    day: date,
    width: int = 3272, 
    variables: list[str]=["INTEGRATED_CO"], 
    remove_night: bool=False,
    save_result: bool = False,
    use_cache: bool = False
    ) -> xr.Dataset:
    """
    Download all the IASI Level 2 products for a given day and accumulate them into a gridded L3 dataset.
    """
    
    version = "v1"
    
    output_file = config.gridded_iasi_dir / f"iasi_l3_grid{width}_{day.strftime('%Y%m%d')}_{version}.nc"
    
    if use_cache and output_file.exists():
        log.info(f"Loading cached Iasi L3 dataset from {output_file}")
        return xr.open_dataset(output_file)
    
    files = download_iasi_day_files(day)
    
    WIDTH = width
    acc = GlobalAccumulator2D(width = WIDTH, height = int(WIDTH/2))
    ds = xr.Dataset()
    
    bad_files = []
    dataset = None
    for variable in variables:
        
        log.info(f"Processing variable: {variable}")
        
        for idx, file in enumerate(files):
            if file in bad_files:
                continue
            try:
                log.info(f"Reading file {idx + 1}/{len(files)}: {file} ")
                dataset = read_iasi_l2(file, variables=variables)
                if remove_night:
                    dataset[variable] = dataset[variable].where(abs(dataset.SZA) < 90)
                acc.add(dataset[variable].values, lat=dataset.latitude.values, lon=dataset.longitude.values)
                
            except Exception as e:
                log.error(f"Error processing file {file}: {e}")
                bad_files.append(file)
                continue
        
        if dataset is None:
            log.warning(f"No valid data found for variable {variable} on {day}. Skipping.")
            continue
        
        # append the gridded L3 to the output dataset
        ds[variable + "_MEAN"] = acc.get_mean_data_array()
        ds[variable + "_CNT"] =  acc.get_cnt_data_array()
        
        if "unit" in dataset[variable].attrs: 
            ds[variable + "_MEAN"].attrs["unit"] = dataset[variable].attrs["unit"]
            
        if "long_name" in dataset[variable].attrs: 
            ds[variable + "_MEAN"].attrs["long_name"] = dataset[variable].attrs["long_name"] + " gridded mean"

    ds.attrs["description"] = f"Gridded L3 dataset from IASI Level 2 products"
    ds.attrs["source_files"] = ", ".join(f.name for f in files)

    if save_result:
        ds.to_netcdf(output_file)
        log.info(f"Saved gridded IASI L3 dataset to {output_file}")

    return ds
