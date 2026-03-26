"""Read IASI L2 Combined Product (.nat files) using CODA library.

IASI L2 Combined Product contains:

Requires CODA library and EPS codadef file.
Download from: https://github.com/stcorp/codadef-eps/releases
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union, Optional, List
import warnings

import numpy as np
import xarray as xr

from wekeo_iasi_l3.hygeos_core import log

# Set CODA_DEFINITION before importing coda
_CODADEF_SEARCH_PATHS = [
    Path(__file__).parent / "codadefs",   # decided to embed codadef files in the package for simplicity (~250kb)
]

def _find_codadef() -> Optional[Path]:
    """Find CODA definition directory containing .codadef files."""
    # Check environment variable first
    if "CODA_DEFINITION" in os.environ:
        return Path(os.environ["CODA_DEFINITION"])
    
    for path in _CODADEF_SEARCH_PATHS:
        if path.exists():
            # Check if directory contains .codadef files
            if path.is_dir() and list(path.glob("*.codadef")):
                return path
            elif path.is_file() and path.suffix == ".codadef":
                return path.parent
    return None


def _setup_coda():
    """Setup CODA environment and import module."""
    codadef_path = _find_codadef()
    if codadef_path is None:
        raise RuntimeError(
            "CODA definition files not found. "
            "Download EPS codadef from https://github.com/stcorp/codadef-eps/releases "
            "and place in 'download/' directory or set CODA_DEFINITION environment variable."
        )
    os.environ["CODA_DEFINITION"] = str(codadef_path)
    import coda
    return coda


# Variables available in IASI L2 Combined Product
IASI_L2_VARIABLES = {
    # Geolocation
    "EARTH_LOCATION": {
        "description": "Earth location (latitude, longitude)",
        "dims": ("scanline", "footprint", "latlon"),
        "units": "degrees",
    },
    "ANGULAR_RELATION": {
        "description": "Angular relations (satellite zenith, satellite azimuth, solar zenith, solar azimuth)",
        "dims": ("scanline", "footprint", "angle"),
        "units": "degrees",
    },
    # Atmospheric profiles
    "ATMOSPHERIC_TEMPERATURE": {
        "description": "Atmospheric temperature profile",
        "dims": ("scanline", "footprint", "pressure_level"),
        "units": "K",
    },
    "ATMOSPHERIC_WATER_VAPOUR": {
        "description": "Atmospheric water vapour profile",
        "dims": ("scanline", "footprint", "pressure_level"),
        "units": "kg/kg",
    },
    "ATMOSPHERIC_OZONE": {
        "description": "Atmospheric ozone profile",
        "dims": ("scanline", "footprint", "pressure_level"),
        "units": "kg/kg",
    },
    # Surface parameters
    "SURFACE_TEMPERATURE": {
        "description": "Surface skin temperature",
        "dims": ("scanline", "footprint"),
        "units": "K",
    },
    "SURFACE_PRESSURE": {
        "description": "Surface pressure",
        "dims": ("scanline", "footprint"),
        "units": "Pa",
    },
    "SURFACE_EMISSIVITY": {
        "description": "Surface emissivity at different wavelengths",
        "dims": ("scanline", "footprint", "emissivity_wavelength"),
        "units": "1",
    },
    # Cloud parameters
    "FRACTIONAL_CLOUD_COVER": {
        "description": "Fractional cloud cover for each cloud formation",
        "dims": ("scanline", "footprint", "cloud_formation"),
        "units": "1",
    },
    "CLOUD_TOP_TEMPERATURE": {
        "description": "Cloud top temperature",
        "dims": ("scanline", "footprint", "cloud_formation"),
        "units": "K",
    },
    "CLOUD_TOP_PRESSURE": {
        "description": "Cloud top pressure",
        "dims": ("scanline", "footprint", "cloud_formation"),
        "units": "Pa",
    },
    "CLOUD_PHASE": {
        "description": "Cloud phase (0=unknown, 1=water, 2=ice, 3=mixed)",
        "dims": ("scanline", "footprint", "cloud_formation"),
        "units": "1",
    },
    "NUMBER_CLOUD_FORMATIONS": {
        "description": "Number of cloud formations detected",
        "dims": ("scanline", "footprint"),
        "units": "1",
    },
    # Integrated trace gases
    "INTEGRATED_WATER_VAPOUR": {
        "description": "Total column water vapour",
        "dims": ("scanline", "footprint"),
        "units": "kg/m^2",
    },
    "INTEGRATED_OZONE": {
        "description": "Total column ozone",
        "dims": ("scanline", "footprint"),
        "units": "kg/m^2",
    },
    "INTEGRATED_N2O": {
        "description": "Total column N2O",
        "dims": ("scanline", "footprint"),
        "units": "kg/m^2",
    },
    "INTEGRATED_CO": {
        "description": "Total column CO",
        "dims": ("scanline", "footprint"),
        "units": "kg/m^2",
    },
    "INTEGRATED_CH4": {
        "description": "Total column CH4",
        "dims": ("scanline", "footprint"),
        "units": "kg/m^2",
    },
    "INTEGRATED_CO2": {
        "description": "Total column CO2",
        "dims": ("scanline", "footprint"),
        "units": "kg/m^2",
    },
    # Quality flags
    "FLG_CLDFRM": {
        "description": "Cloud formation flag",
        "dims": ("scanline", "footprint"),
        "units": "1",
    },
    "FLG_LANSEA": {
        "description": "Land/sea flag",
        "dims": ("scanline", "footprint"),
        "units": "1",
    },
    "FLG_DAYNIT": {
        "description": "Day/night flag",
        "dims": ("scanline", "footprint"),
        "units": "1",
    },
    "FLG_ITCONV": {
        "description": "Iteration convergence flag",
        "dims": ("scanline", "footprint"),
        "units": "1",
    },
}


def explore_product(input_path: Union[str, Path]) -> None:
    """
    Print the structure of an IASI L2 product file.
    
    Args:
        input_path: Path to directory containing .nat file or path to .nat file
    """
    coda = _setup_coda()
    nat_file = _find_nat_file(input_path)
    
    with coda.Product(str(nat_file)) as product:
        print(f"File: {nat_file.name}")
        print(f"Product class: {product.product_class}")
        print(f"Product type: {product.product_type}")
        print(f"Product version: {product.version}")
        print(f"Format: {product.format}")
        print()
        
        # Root fields
        root_fields = product.field_names()
        print(f"Root fields: {root_fields}")
        print()
        
        # MDR info
        mdr_size = coda.get_size(product, "MDR")
        print(f"Number of MDR records (scanlines): {mdr_size[0]}")
        
        # MDR fields
        mdr_fields = product.field_names("MDR", 0, "MDR")
        print(f"\nMDR fields ({len(mdr_fields)} total):")
        for field in mdr_fields:
            print(f"  - {field}")


def _find_nat_file(input_path: Union[str, Path]) -> Path:
    """Find the .nat file from input path."""
    input_path = Path(input_path)
    
    if input_path.is_dir():
        nat_files = list(input_path.glob("*.nat"))
        if not nat_files:
            raise FileNotFoundError(f"No .nat file found in {input_path}")
        return nat_files[0]
    elif input_path.suffix == ".nat":
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")
        return input_path
    else:
        raise ValueError(f"Expected .nat file or directory, got: {input_path}")


def _read_variable_by_record(coda, product, var_name: str, n_scanlines: int) -> Optional[np.ndarray]:
    """
    Read a variable record by record, handling "special" type MDRs.
    
    When some MDR records are marked as "special" (empty/missing data), 
    bulk fetch with -1 index fails. This function reads each record 
    individually and fills gaps with NaN.
    
    Args:
        coda: The coda module
        product: Open CODA product handle  
        var_name: Name of the variable to read
        n_scanlines: Number of MDR records (scanlines)
        
    Returns:
        numpy array with data, or None if all records failed
    """
    records = []
    valid_count = 0
    
    for i in range(n_scanlines):
        # try:
        
        raw_data = coda.fetch(product, "MDR", i, "MDR", var_name)
        records.append(np.array(raw_data))
        valid_count += 1
        
        # except Exception:
        #     # This record is "special" (missing) - will be filled with NaN
        #     log.warning(f"Record {i} for variable '{var_name}' is missing/special. Filling with NaN.")
        #     records.append(None)
    
    if valid_count == 0:
        return None
    
    # Find a valid record to determine shape and dtype
    ref_record = next(r for r in records if r is not None)
    shape = ref_record.shape
    dtype = ref_record.dtype
    
    # Create output array filled with NaN
    out_shape = (n_scanlines,) + shape
    result = np.full(out_shape, np.nan, dtype=dtype)
    
    # Fill in valid records
    for i, rec in enumerate(records):
        if rec is not None:
            result[i] = rec
    
    return result


def read_iasi_l2(
    input_path: Union[str, Path],
    variables: Optional[List[str]] = None,
    chunked: bool = False,
) -> xr.Dataset:
    """
    Read IASI L2 Combined Product and return as xarray Dataset.
    
    Args:
        input_path: Path to directory containing .nat file or path to .nat file directly
        variables: List of variable names to read.
                  Use IASI_L2_VARIABLES.keys() to see all available variables.
        chunked: If True, return dask arrays (for lazy loading). Not yet implemented.
    
    Returns:
        xarray.Dataset with IASI L2 data
        
    Example:
        >>> ds = read_iasi_l2("path/to/IASI_SND_02_M01_.../")
        >>> print(ds)
        >>> # Access temperature profile
        >>> temp = ds.ATMOSPHERIC_TEMPERATURE
        >>> # Access surface temperature 
        >>> sfc_temp = ds.SURFACE_TEMPERATURE
        >>> # Get coordinates
        >>> lat = ds.latitude
        >>> lon = ds.longitude
    """
    coda = _setup_coda()
    
    if chunked:
        raise NotImplementedError("Chunked/lazy loading not yet implemented")
    
    nat_file = _find_nat_file(input_path)
    
    # Validate variable names
    invalid_vars = set(variables) - set(IASI_L2_VARIABLES.keys())
    if invalid_vars:
        raise ValueError(
            f"Unknown variables: {invalid_vars}. "
            f"Available: {list(IASI_L2_VARIABLES.keys())}"
        )
    
    with coda.Product(str(nat_file)) as product:
        # Get dimensions
        n_scanlines = coda.get_size(product, "MDR")[0]
        
        # Get coordinate data from GIADR
        pressure_levels = np.array(coda.fetch(product, "GIADR", "PRESSURE_LEVELS_TEMP"))
        emissivity_wavelengths = np.array(coda.fetch(product, "GIADR", "SURFACE_EMISSIVITY_WAVELENGTHS"))
        
        # Read all MDR data
        data_vars = {}
        
        always_read_variables = ["ANGULAR_RELATION", "EARTH_LOCATION"]
        variables = set(variables + always_read_variables)  # Ensure we always read these for coordinates
        
        for var_name in variables:
            var_info = IASI_L2_VARIABLES[var_name]
            
            # try:
            
            # First try to fetch all scanlines at once using -1 index (faster)
            raw_data = coda.fetch(product, "MDR", -1, "MDR", var_name)
            data = np.array(raw_data)
            
            # Handle object arrays (happens when CODA returns nested structures)
            if data.dtype == object:
                # Stack into proper array
                data = np.stack([np.array(d) for d in data])
            
            data_vars[var_name] = (var_info["dims"], data)
                
            # except Exception as e:
            #     log.error(f"Failed to read variable '{var_name}' with bulk fetch: {e}")
            #     # If bulk fetch fails (due to "special" type MDRs), read record by record
            #     if "special" in str(e).lower():
            #         try:
            #             data = _read_variable_by_record(coda, product, var_name, n_scanlines)
            #             if data is not None:
            #                 data_vars[var_name] = (var_info["dims"], data)
            #             else:
            #                 warnings.warn(f"Could not read any records for {var_name}")
            #         except Exception as e2:
            #             warnings.warn(f"Could not read {var_name} even record-by-record: {e2}")
            #             continue
            #     else:
            #         warnings.warn(f"Could not read {var_name}: {e}")
            #         continue
        
        # Extract lat/lon from EARTH_LOCATION if available
        coords = {}
        if "EARTH_LOCATION" in data_vars:
            earth_loc = data_vars["EARTH_LOCATION"][1]
            coords["latitude"] = (("scanline", "footprint"), earth_loc[:, :, 0])
            coords["longitude"] = (("scanline", "footprint"), earth_loc[:, :, 1])
        
        # Add coordinate arrays
        coords["pressure_level"] = ("pressure_level", pressure_levels)
        coords["emissivity_wavelength"] = ("emissivity_wavelength", emissivity_wavelengths)
        coords["scanline"] = ("scanline", np.arange(n_scanlines))
        coords["footprint"] = ("footprint", np.arange(120))  # IASI has 120 footprints per scanline
        
        coords["cloud_formation"] = ("cloud_formation", np.arange(3))  # Up to 3 cloud layers
        # 'long_name': 'Angular relations (satellite zenith, satellite azimuth, solar zenith, solar azimuth)', 
        
        coords["angle"] = ("angle", ["VZA", "VAA", "SZA", "SAA"])
        
        # convert to VZA, VAA, SZA, SAA if ANGULAR_RELATION is available
        if "ANGULAR_RELATION" in data_vars:
            angles = data_vars["ANGULAR_RELATION"][1]
            coords["VZA"] = (("scanline", "footprint"), angles[:, :, 0])
            coords["VAA"] = (("scanline", "footprint"), angles[:, :, 1])
            coords["SZA"] = (("scanline", "footprint"), angles[:, :, 2])
            coords["SAA"] = (("scanline", "footprint"), angles[:, :, 3])
        
        # Create dataset
        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        
        # Add attributes
        ds.attrs["source_file"] = str(nat_file.name)
        ds.attrs["product_class"] = product.product_class
        ds.attrs["product_type"] = product.product_type
        ds.attrs["product_version"] = product.version
        ds.attrs["title"] = "IASI L2 Combined Product"
        ds.attrs["institution"] = "EUMETSAT"
        ds.attrs["instrument"] = "IASI"
        
        # Add variable attributes
        for var_name, var_info in IASI_L2_VARIABLES.items():
            if var_name in ds:
                ds[var_name].attrs["long_name"] = var_info["description"]
                ds[var_name].attrs["units"] = var_info["units"]
            
                # filter fill values only for non-coordinate variables
                # WARNING: kind of a hack but works
                if var_name not in always_read_variables:
                    fill_val = ds[var_name].max()
                    ds[var_name] = ds[var_name].where(ds[var_name] < fill_val * 0.99, np.nan) # NOTE: Unclean HACK for fill values
                    
        
        # Add coordinate attributes
        if "latitude" in ds.coords:
            ds.coords["latitude"].attrs["units"] = "degrees_north"
            ds.coords["latitude"].attrs["long_name"] = "Latitude"
            
        if "longitude" in ds.coords:
            ds.coords["longitude"].attrs["units"] = "degrees_east"
            ds.coords["longitude"].attrs["long_name"] = "Longitude"
            
        if "pressure_level" in ds.coords:
            ds.coords["pressure_level"].attrs["units"] = "Pa"
            ds.coords["pressure_level"].attrs["long_name"] = "Pressure level"
            ds.coords["pressure_level"].attrs["positive"] = "down"
        
        return ds
