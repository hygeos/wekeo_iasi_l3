from wekeo_iasi_l3.hygeos_core import env
output_dir = env.getdir("OUTPUT_DIR")
ancillary_dir = env.getdir("DIR_ANCILLARY") 

if not ancillary_dir.exists():
    raise FileNotFoundError(f"Ancillary directory {ancillary_dir} does not exist. Please check your environment configuration.")

iasi_download_dir = ancillary_dir / "IASI_L2"
iasi_download_dir.mkdir(parents=False, exist_ok=True)

if not output_dir.exists():
    raise FileNotFoundError(f"Output directory {output_dir} does not exist. Please create it or check your environment configuration.")


gridded_iasi_dir = output_dir / "gridded_iasi"
gridded_iasi_dir.mkdir(parents=False, exist_ok=True)