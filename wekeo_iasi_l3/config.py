from wekeo_iasi_l3.hygeos_core import env
output_dir = env.getdir("OUTPUT_DIR")
iasi_download_dir = env.getdir("DIR_ANCILLARY") / "IASI_L2"

if not iasi_download_dir.exists():
    raise FileNotFoundError(f"IASI download directory {iasi_download_dir} does not exist. Please create it or check your environment configuration.")

if not output_dir.exists():
    raise FileNotFoundError(f"Output directory {output_dir} does not exist. Please create it or check your environment configuration.")


gridded_iasi_dir = output_dir / "gridded_iasi"
gridded_iasi_dir.mkdir(parents=False, exist_ok=True)