"""Download and extract IASI L3 data from WEkEO."""
import zipfile
from datetime import datetime
from pathlib import Path

from hda import Client, Configuration

from wekeo_iasi_l3 import config, env
from wekeo_iasi_l3.hygeos_core import log


def unzip(archive: Path, to: Path | None):
    """
    Extract a zip archive to a specified directory. Keep the same name as the archive
    without the .zip extension.
    
    Args:
        archive: Path to the zip file to extract
        to: Path to the directory where contents will be extracted.
            If None, extracts to the same directory as the archive.
    
    Returns:
        None
    """
    if to is None:
        to = archive.parent  # Extract to same directory as archive
    
    archive = Path(archive)
    to = Path(to)
    target = to / archive.stem
    
    try:
        with zipfile.ZipFile(archive, 'r') as zip_ref:
            zip_ref.extractall(target)
    except Exception as e:
        log.error(f"Failed to extract {archive} to {target}", exc_info=True)
        return


def download(query, archive_dir: Path, extract_dir: Path | None = None, rm_archive: bool = False, recursive_try: int = 0, max_recursive_try: int = 3):
    """
    Download files from query results, skipping files that already exist locally.
    
    Args:
        query: SearchResults object containing items to download
        archive_dir: Path where archive files will be downloaded
        extract_dir: Path where archives files will be extracted
        rm_archive: bool, whether to remove archive after extraction (default: False)
        recursive_try: Current recursion attempt counter
        max_recursive_try: Maximum number of recursive download attempts
    
    Returns:
        list: List of extracted paths
    """
    
    if extract_dir is None:
        extract_dir = archive_dir
    
    missing = []  # archives to download
    extract = []  # archives to extract after download
    results = []  # all extracted paths
    
    for item in query.results:
        
        archive_path = archive_dir / f"{item['id']}.zip"
        extract_path = extract_dir / item['id']
        results.append(extract_path)
        
        if extract_path.exists() == True:
            continue  # already extracted, skip
            
        elif archive_path.exists() == True:  # archive exists locally but not extracted
            extract.append(archive_path)  # queue archive for extraction after download
            
        else:
            missing.append(item)  # archive missing, queue for download
            extract.append(archive_path)  # queue archive for extraction after download
            
    # Download missing archives
    if missing:  # query only if missing files
        log.info(f"Downloading {len(missing)} missing files...")
        query.results = missing
        query.download(download_dir=archive_dir)
    else:
        log.info("All files already present locally, skipping download.")
    
    error_not_downloaded = []
    
    # Extract downloaded archives
    if extract:
        for archive in extract:
            if not archive.exists():
                error_not_downloaded.append(archive)
                continue
            
            unzip(archive, to=extract_dir)
            if rm_archive:
                archive.unlink()  # remove archive after extraction

    if error_not_downloaded:
        if recursive_try >= max_recursive_try:
            raise RuntimeError(f"Error: Maximum recursive download attempts ({max_recursive_try}) reached. Some files could not be downloaded.")
        log.info(f"Warning: {len(error_not_downloaded)} archives were not downloaded and could not be extracted:")
        log.info("Recursively try again to download missing files.")
        results = download(query, archive_dir, extract_dir, rm_archive, recursive_try + 1, max_recursive_try)

    return results


def format_query(
    start_date: str,
    end_date: str,
    publication: str | None = None,
):
    """
    Format a query for IASI L3 data.
    
    Args:
        start_date: Start date in format "YYYY-MM-DD"
        end_date: End date in format "YYYY-MM-DD"
        publication: Publication date in ISO format (optional)
    
    Returns:
        dict: Formatted query for the HDA client
    """
    
    # convert from string to datetime
    start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")

    json_query = {
        "dataset_id": "EO:EUM:DAT:METOP:IASSND02",
        "startdate": start_date_dt.strftime("%Y-%m-%dT00:00:00.000Z"),
        "enddate": end_date_dt.strftime("%Y-%m-%dT23:59:59.999Z"),
        "sat": "Metop-B",
        "itemsPerPage": 200,
        "startIndex": 0
    }
    
    if publication is not None:
        json_query["publication"] = publication
    
    return json_query



def download_IASI_products(
    start_date: str,
    end_date: str,
    publication: str | None = None,
):
    """
    Query and download METOP IASI L3 products from WEkEO.
    
    Args:
        start_date: Start date in format "YYYY-MM-DD"
        end_date: End date in format "YYYY-MM-DD"
        publication: Publication date in ISO format (optional)
    
    Returns:
        List[Path]: List of paths to the extracted IASI product directories
    """

    log.info(f"Querying IASI L3 products from {start_date} to {end_date} with publication date {publication}")
    hda_client = Client()

    json_query = format_query(start_date, end_date, publication)
    query = hda_client.search(json_query)
    
    results = download(query, archive_dir=config.iasi_download_dir, rm_archive=False)
    return results
