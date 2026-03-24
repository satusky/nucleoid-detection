"""File loading utilities for TIFF and HDF5 image data."""

import io
import tempfile
import numpy as np
import streamlit as st
import tifffile
import h5py


def load_tiff(uploaded):
    """Read a TIFF from an UploadedFile object."""
    return tifffile.imread(io.BytesIO(uploaded.read()))


def _collect_h5_datasets(group, prefix=""):
    """Recursively collect dataset paths from an HDF5 group."""
    paths = []
    for key in group:
        item = group[key]
        full = f"{prefix}/{key}" if prefix else key
        if isinstance(item, h5py.Dataset):
            paths.append(full)
        elif isinstance(item, h5py.Group):
            paths.extend(_collect_h5_datasets(item, full))
    return paths


def load_h5_datasets(uploaded):
    """Open an uploaded HDF5 file and return (h5py.File, dataset_paths).

    Because h5py needs a real file handle, we write to a temp file.
    The caller must close the returned File object.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    tmp.write(uploaded.getvalue())
    tmp.flush()
    f = h5py.File(tmp.name, "r")
    paths = _collect_h5_datasets(f)
    return f, paths, tmp.name


def image_upload_widget(label, key_prefix, accept_multiple=False):
    """Unified upload widget that accepts TIFF or HDF5 image stacks.

    Returns a list of 2-D numpy arrays (one per channel / selected frame).
    For TIFF: each file is one channel.
    For HDF5: user picks a dataset; if 3-D+, a frame slider appears.

    Parameters
    ----------
    label : str
        Label shown on the uploader.
    key_prefix : str
        Unique prefix for widget keys.
    accept_multiple : bool
        Allow multiple file uploads (TIFF only; HDF5 always single).

    Returns
    -------
    images : list[ndarray] or None
        List of 2-D images, or None if nothing uploaded yet.
    names : list[str] or None
        Display names for each image.
    """
    uploaded = st.file_uploader(
        label,
        type=["tif", "tiff", "h5", "hdf5"],
        accept_multiple_files=accept_multiple,
        key=f"{key_prefix}_upload",
    )

    if not uploaded:
        return None, None

    # Normalize to list
    files = uploaded if isinstance(uploaded, list) else [uploaded]

    # Check if any file is HDF5
    h5_files = [f for f in files if f.name.lower().endswith((".h5", ".hdf5"))]
    tiff_files = [f for f in files if f not in h5_files]

    images = []
    names = []

    # Load TIFFs
    for f in tiff_files:
        img = tifffile.imread(io.BytesIO(f.getvalue()))
        if img.ndim == 2:
            images.append(img)
            names.append(f.name)
        elif img.ndim == 3:
            # TIFF stack — pick a frame or sum frames
            sum_frames = st.checkbox(
                f"Sum frames ({f.name})", key=f"{key_prefix}_tiff_sum_{f.name}",
            )
            if sum_frames:
                range_col1, range_col2 = st.columns(2)
                with range_col1:
                    sum_start = st.number_input(
                        f"Start frame ({f.name})", value=0,
                        min_value=0, max_value=img.shape[0] - 1,
                        key=f"{key_prefix}_tiff_sum_start_{f.name}",
                    )
                with range_col2:
                    sum_end = st.number_input(
                        f"End frame ({f.name})", value=img.shape[0] - 1,
                        min_value=0, max_value=img.shape[0] - 1,
                        key=f"{key_prefix}_tiff_sum_end_{f.name}",
                    )
                sum_start, sum_end = int(min(sum_start, sum_end)), int(max(sum_start, sum_end))
                images.append(img[sum_start:sum_end + 1].sum(axis=0))
                names.append(f"{f.name} [sum {sum_start}-{sum_end}]")
            else:
                frame_idx = st.slider(
                    f"Frame index ({f.name})",
                    0, img.shape[0] - 1, 0,
                    key=f"{key_prefix}_tiff_frame_{f.name}",
                )
                frame_idx = st.number_input(
                    f"Frame index (manual, {f.name})",
                    value=frame_idx, min_value=0, max_value=img.shape[0] - 1,
                    key=f"{key_prefix}_tiff_frame_input_{f.name}",
                )
                images.append(img[frame_idx])
                names.append(f"{f.name} [frame {frame_idx}]")
        else:
            st.warning(f"Unsupported TIFF shape {img.shape} in {f.name}, skipping.")

    # Load HDF5 files
    for f in h5_files:
        cache_key = f"{key_prefix}_h5cache_{f.name}"
        h5_bytes = f.getvalue()
        h5_hash = hash(h5_bytes)

        # Cache the opened file to avoid re-writing on every rerun
        prev = st.session_state.get(f"{cache_key}_hash")
        if prev != h5_hash:
            tmp = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
            tmp.write(h5_bytes)
            tmp.flush()
            h5f = h5py.File(tmp.name, "r")
            ds_paths = _collect_h5_datasets(h5f)
            st.session_state[f"{cache_key}_hash"] = h5_hash
            st.session_state[f"{cache_key}_file"] = h5f
            st.session_state[f"{cache_key}_paths"] = ds_paths
            st.session_state[f"{cache_key}_tmp"] = tmp.name

        h5f = st.session_state[f"{cache_key}_file"]
        ds_paths = st.session_state[f"{cache_key}_paths"]

        if not ds_paths:
            st.warning(f"No datasets found in {f.name}.")
            continue

        ds_name = st.selectbox(
            f"Dataset ({f.name})", ds_paths,
            key=f"{key_prefix}_h5ds_{f.name}",
        )
        ds = h5f[ds_name]
        st.caption(f"Shape: {ds.shape}, dtype: {ds.dtype}")

        if ds.ndim == 2:
            images.append(np.array(ds))
            names.append(f"{f.name}:/{ds_name}")
        elif ds.ndim >= 3:
            sum_frames = st.checkbox(
                f"Sum frames ({f.name}:/{ds_name})",
                key=f"{key_prefix}_h5sum_{f.name}_{ds_name}",
            )
            if sum_frames:
                range_col1, range_col2 = st.columns(2)
                with range_col1:
                    sum_start = st.number_input(
                        f"Start frame ({f.name}:/{ds_name})", value=0,
                        min_value=0, max_value=ds.shape[0] - 1,
                        key=f"{key_prefix}_h5sum_start_{f.name}_{ds_name}",
                    )
                with range_col2:
                    sum_end = st.number_input(
                        f"End frame ({f.name}:/{ds_name})",
                        value=ds.shape[0] - 1,
                        min_value=0, max_value=ds.shape[0] - 1,
                        key=f"{key_prefix}_h5sum_end_{f.name}_{ds_name}",
                    )
                sum_start, sum_end = int(min(sum_start, sum_end)), int(max(sum_start, sum_end))
                frame = np.array(ds[sum_start:sum_end + 1]).sum(axis=0)
                while frame.ndim > 2:
                    frame = frame[0]
                images.append(frame)
                names.append(f"{f.name}:/{ds_name} [sum {sum_start}-{sum_end}]")
            else:
                frame_idx = st.slider(
                    f"Frame index ({f.name}:/{ds_name})",
                    0, ds.shape[0] - 1, 0,
                    key=f"{key_prefix}_h5frame_{f.name}_{ds_name}",
                )
                frame_idx = st.number_input(
                    f"Frame index (manual)",
                    value=frame_idx, min_value=0, max_value=ds.shape[0] - 1,
                    key=f"{key_prefix}_h5frame_input_{f.name}_{ds_name}",
                )
                frame = np.array(ds[frame_idx])
                while frame.ndim > 2:
                    frame = frame[0]
                images.append(frame)
                names.append(f"{f.name}:/{ds_name} [frame {frame_idx}]")
        else:
            st.warning(f"Dataset {ds_name} is 1-D, skipping.")

    if not images:
        return None, None
    return images, names
