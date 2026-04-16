"""Top-level package initialization for wekeo_iasi_l3.

This package requires the STCorp CODA Python API (the one exposing
``coda.Product`` / ``coda.fetch``). A different package named ``coda`` exists
on PyPI and is not compatible with this project.
"""


def _validate_coda_installation() -> None:
    """Fail fast if CODA is missing or if the wrong ``coda`` package is installed."""
    try:
        import coda
    except Exception as exc:
        raise ImportError(
            "Failed to import the required STCorp CODA Python API. "
            "Install CODA from conda-forge (package name: coda), not the unrelated PyPI 'coda' package."
        ) from exc

    required_symbols = ("Product", "fetch", "get_size")
    missing_symbols = [name for name in required_symbols if not hasattr(coda, name)]
    if missing_symbols:
        missing = ", ".join(missing_symbols)
        coda_file = getattr(coda, "__file__", "<unknown>")
        raise ImportError(
            "Detected an incompatible 'coda' module at "
            f"{coda_file}. Missing expected STCorp CODA API symbols: {missing}. "
            "Install CODA from conda-forge (package name: coda) and remove the unrelated PyPI 'coda' package."
        )
    
    else:
        print("Successfully validated the presence of the STCorp CODA Python API.")


_validate_coda_installation()

