"""Read IASI L2 .nat files and convert to xarray datasets.

IASI (Infrared Atmospheric Sounding Interferometer) L2 Product contains:
- Temperature Profiles
- Humidity Profiles
- Surface Temperature
- Surface Emissivity
- Fractional Cloud Cover
- Cloud Top Temperature
- Cloud Top Pressure
- Cloud Phase
- Total Column Ozone
- Columnar ozone amounts in thick layers
- Total column N2O, CO, CH4, CO2
"""
from pathlib import Path
from typing import List, Union
import warnings

import numpy as np
import xarray as xr

import coda


# Common IASI L2 variables (actual names may vary by product version)
# Use explore_iasi_structure() to see all available fields in your specific file
IASI_COMMON_VARIABLES = {
    # Geolocation
    "GGeoSondLoc": "Geolocation (latitude, longitude)",
    "GGeoSondAnglesMETOP": "Viewing angles (satellite zenith, azimuth, solar angles)",
    "GEPSLocInd": "Location indicators",
    
    # Surface parameters
    "GEPS_SP": "Surface pressure",
    "GEPS_TSurfAir": "Surface skin temperature",
    "GEPS_Emiss": "Surface emissivity",
    
    # Cloud parameters
    "GEPS_CldFrac": "Fractional cloud cover",
    "GEPS_CldTopPres": "Cloud top pressure",
    "GEPS_CldTopTemp": "Cloud top temperature",
    "GEPS_CldPhase": "Cloud phase",
    
    # Atmospheric profiles
    "GEPSProfTemp": "Temperature profile",
    "GEPSProfHum": "Humidity profile",
    
    # Trace gases - Total columns
    "GEPS_ColumnO3": "Total column ozone",
    "GEPS_ColumnCO": "Total column CO",
    "GEPS_ColumnCH4": "Total column CH4",
    "GEPS_ColumnCO2": "Total column CO2",
    "GEPS_ColumnN2O": "Total column N2O",
}


def explore_iasi_structure(nat_file: Path, max_depth: int = 3):
    """
    Explore the structure of an IASI .nat file.
    
    Args:
        nat_file: Path to the .nat file
        max_depth: Maximum depth to explore in the structure
    
    Returns:
        None (prints structure)
    """
    product = coda.open(str(nat_file))
    
    print(f"Product class: {coda.get_product_class(product)}")
    print(f"Product type: {coda.get_product_type(product)}")
    print(f"Product format: {coda.get_product_format(product)}")
    print("\nStructure:")
    
    cursor = coda.Cursor()
    coda.cursor_set_product(cursor, product)
    
    def explore(cursor, indent=0, depth=0):
        if depth >= max_depth:
            return
            
        type_class = coda.cursor_get_type_class(cursor)
        
        if type_class == coda.coda_record_class:
            num_fields = coda.cursor_get_num_elements(cursor)
            for i in range(num_fields):
                coda.cursor_goto_record_field_by_index(cursor, i)
                field_name = coda.cursor_get_record_field_available_name(cursor, i)
                
                try:
                    field_type = coda.cursor_get_type_class(cursor)
                    type_name = {
                        coda.coda_record_class: "record",
                        coda.coda_array_class: "array",
                        coda.coda_integer_class: "int",
                        coda.coda_real_class: "float",
                        coda.coda_text_class: "text",
                        coda.coda_raw_class: "raw",
                    }.get(field_type, "unknown")
                    
                    if field_type == coda.coda_array_class:
                        dims = coda.cursor_get_array_dim(cursor)
                        print("  " * indent + f"- {field_name} ({type_name}, shape={dims})")
                    else:
                        print("  " * indent + f"- {field_name} ({type_name})")
                    
                    explore(cursor, indent + 1, depth + 1)
                except Exception as e:
                    print("  " * indent + f"- {field_name} (error: {e})")
                
                coda.cursor_goto_parent(cursor)
    
    explore(cursor)
    coda.close(product)


def read_iasi_variable(product, path: List[str], index: int = -1):
    """
    Read a variable from the IASI product.
    
    Args:
        product: Open coda product
        path: List of field names forming the path to the variable
        index: Index for array fields (-1 for all)
    
    Returns:
        numpy array with the data
    """
    try:
        if index == -1:
            # Fetch all records
            fetch_path = path
        else:
            # Fetch specific record
            fetch_path = path[:1] + [index] + path[1:]
        
        data = coda.fetch(product, *fetch_path)
        return np.array(data)
    except Exception as e:
        warnings.warn(f"Could not read {'/'.join(path)}: {e}")
        return None


def read_IASI_L2_product(
    input_path: Union[str, Path],
    variables: List[str] = None,
) -> xr.Dataset:
    """
    Read IASI L2 .nat file and return an xarray Dataset.
    
    Args:
        input_path: Path to directory containing .nat file or direct path to .nat file
        variables: List of variable names to read. If None, reads common variables.
    
    Returns:
        xarray.Dataset with IASI L2 data
    
    Example:
        >>> ds = read_IASI_L2_product(
        ...     "test_input/IASI_SND_02_M01_20240816222952Z_20240817001456Z_N_O_20240816233022Z/",
        ...     variables=["GGeoSondLoc", "GEPS_CldFrac"]
        ... )
    """
    input_path = Path(input_path)
    
    # Find .nat file
    if input_path.is_dir():
        nat_files = list(input_path.glob("*.nat"))
        if not nat_files:
            raise FileNotFoundError(f"No .nat file found in {input_path}")
        nat_file = nat_files[0]
    else:
        nat_file = input_path
    
    if not nat_file.exists():
        raise FileNotFoundError(f"File not found: {nat_file}")
    
    # Default variables to read
    # Common IASI L2 variable names (actual names may vary, use explore_iasi_structure to check)
    if variables is None:
        variables = [
            "GGeoSondLoc",           # Geolocation (lat/lon)
            "GGeoSondAnglesMETOP",   # Viewing angles (satellite zenith, azimuth, etc.)
            "GEPS_CldFrac",          # Fractional cloud cover
            "GEPS_SP",               # Surface pressure
            "GEPS_TSurfAir",         # Surface skin temperature
            "GEPSLocInd",            # Location indicators
            # Temperature & Humidity profiles:
            # "GEPSProfTemp",        # Temperature profile
            # "GEPSProfHum",         # Humidity profile
            # Trace gases:
            # "GEPS_ColumnO3",       # Total column ozone
            # "GEPS_ColumnCO",       # Total column CO
            # "GEPS_ColumnCH4",      # Total column CH4
            # "GEPS_ColumnCO2",      # Total column CO2
            # "GEPS_ColumnN2O",      # Total column N2O
            # Cloud parameters:
            # "GEPS_CldTopPres",     # Cloud top pressure
            # "GEPS_CldTopTemp",     # Cloud top temperature
            # "GEPS_CldPhase",       # Cloud phase
        ]
    
    # Open product
    product = coda.open(str(nat_file))
    
    try:
        # The main data is typically in MDR (Measurement Data Record)
        # Structure is usually: MDR/<record_index>/field_name
        
        data_vars = {}
        coords = {}
        
        # Read each variable
        for var_name in variables:
            data = read_iasi_variable(product, ["MDR", -1, var_name])
            if data is not None:
                # Handle different dimensionalities
                if data.ndim == 1:
                    # 1D array: one value per scanline
                    data_vars[var_name] = (["scanline"], data)
                elif data.ndim == 2:
                    # 2D array: scanline x footprint (typically 4 footprints per scanline)
                    if data.shape[1] <= 4:
                        data_vars[var_name] = (["scanline", "footprint"], data)
                    else:
                        # Could be scanline x channels or other
                        data_vars[var_name] = (["scanline", f"dim_{data.shape[1]}"], data)
                elif data.ndim == 3:
                    # 3D array
                    data_vars[var_name] = (
                        ["scanline", f"dim_{data.shape[1]}", f"dim_{data.shape[2]}"],
                        data
                    )
                else:
                    # Higher dimensions
                    dims = [f"dim_{i}" for i in range(data.ndim)]
                    dims[0] = "scanline"
                    data_vars[var_name] = (tuple(dims), data)
        
        # Try to extract latitude/longitude if GGeoSondLoc was read
        if "GGeoSondLoc" in data_vars:
            geo_data = data_vars["GGeoSondLoc"][1]
            if geo_data.shape[1] >= 2:
                # Typically: [scanline, 2] where 2 is [lat, lon]
                coords["latitude"] = (["scanline"], geo_data[:, 0])
                coords["longitude"] = (["scanline"], geo_data[:, 1])
        
        # Create dataset
        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        
        # Add attributes
        ds.attrs["source_file"] = str(nat_file.name)
        ds.attrs["product_class"] = coda.get_product_class(product)
        ds.attrs["product_type"] = coda.get_product_type(product)
        
        return ds
        
    finally:
        coda.close(product)
