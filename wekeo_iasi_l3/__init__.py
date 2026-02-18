"""WEkEO IASI L3 data download package."""
from wekeo_iasi_l3.download import get_IASI_products
from wekeo_iasi_l3.reader import (
    read_iasi_l2,
    read_iasi_l2_subset,
    explore_product,
    open_iasi,
    IASI_L2_VARIABLES,
    DEFAULT_VARIABLES,
)

# Backwards compatibility alias
read_IASI_L2_product = read_iasi_l2
explore_iasi_structure = explore_product
IASI_COMMON_VARIABLES = IASI_L2_VARIABLES

__all__ = [
    'get_IASI_products', 
    'read_iasi_l2',
    'read_iasi_l2_subset',
    'read_IASI_L2_product',  # backwards compat
    'explore_product',
    'explore_iasi_structure',  # backwards compat
    'open_iasi',
    'IASI_L2_VARIABLES',
    'IASI_COMMON_VARIABLES',  # backwards compat
    'DEFAULT_VARIABLES',
]
