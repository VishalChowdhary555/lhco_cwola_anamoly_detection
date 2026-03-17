from pathlib import Path
import pandas as pd
import h5py


def inspect_hdf5_structure(h5_path):
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        items = []

        def visitor(name, obj):
            if isinstance(obj, h5py.Group):
                items.append({"type": "group", "name": name})
            elif isinstance(obj, h5py.Dataset):
                items.append({
                    "type": "dataset",
                    "name": name,
                    "shape": obj.shape,
                    "dtype": str(obj.dtype)
                })

        f.visititems(visitor)
    return items


def list_hdf_keys(h5_path):
    store = pd.HDFStore(str(h5_path), mode="r")
    keys = store.keys()
    store.close()
    return keys


def load_hdf_preview(h5_path, key="df", nrows=5):
    return pd.read_hdf(h5_path, key=key, stop=nrows)


def load_full_dataframe(h5_path, key="df"):
    return pd.read_hdf(h5_path, key=key)
