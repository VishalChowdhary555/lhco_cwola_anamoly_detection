from pathlib import Path
import requests
from tqdm.auto import tqdm

from src.config import ZENODO_API_URL, TARGET_FILENAME
from src.utils import ensure_dir


def fetch_zenodo_metadata(api_url=ZENODO_API_URL):
    response = requests.get(api_url, timeout=60)
    response.raise_for_status()
    return response.json()


def extract_files_from_metadata(metadata):
    files = []
    for item in metadata.get("files", []):
        filename = item.get("key", "unknown_file")
        size = item.get("size", None)
        links = item.get("links", {})
        direct_url = (
            links.get("self")
            or links.get("download")
            or item.get("self")
            or item.get("download")
        )
        files.append({
            "filename": filename,
            "size": size,
            "url": direct_url
        })
    return files


def get_target_file_info(filename=TARGET_FILENAME):
    metadata = fetch_zenodo_metadata()
    files = extract_files_from_metadata(metadata)

    for f in files:
        if f["filename"] == filename:
            return f

    raise ValueError(f"File '{filename}' not found in Zenodo record.")


def download_file(url, output_path, expected_size=None, chunk_size=1024 * 1024):
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()

        total = expected_size
        if total is None:
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                total = int(content_length)

        with open(output_path, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=output_path.name
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    return output_path


def download_lhco_dataset(output_path, filename=TARGET_FILENAME):
    info = get_target_file_info(filename)
    return download_file(info["url"], output_path, expected_size=info["size"])
