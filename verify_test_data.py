"""Validate the short Open Ephys ZIP without extracting or modifying the data.
Run: python verify_test_data.py path/to/2026-0822_473nm_optotest_cosine.zip
"""
import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np


def check_archive(archive):
    archive = Path(archive)
    if not archive.is_file():
        raise FileNotFoundError(f"ZIP not found: {archive}")
    reports = []
    with zipfile.ZipFile(archive) as z:
        files = z.namelist()
        structure_paths = [p for p in files if p.endswith('/structure.oebin')]
        if len(structure_paths) != 1:
            raise ValueError(f"Expected exactly one structure.oebin, found {len(structure_paths)}")
        structure_path = structure_paths[0]
        root = structure_path.rsplit('/', 1)[0]
        metadata = json.loads(z.read(structure_path))
        if not metadata.get('continuous'):
            raise ValueError('Open Ephys metadata has no continuous streams')
        for stream in metadata['continuous']:
            folder = stream['folder_name'].strip('/')
            path = f"{root}/continuous/{folder}"
            data_file = f"{path}/continuous.dat"
            timestamps_file = f"{path}/timestamps.npy"
            samples_file = f"{path}/sample_numbers.npy"
            for name in (data_file,timestamps_file,samples_file):
                if name not in files:
                    raise FileNotFoundError(f"Missing required input: {name}")
            nc = int(stream['num_channels']); fs = float(stream['sample_rate'])
            bytes_per_frame = nc * 2  # Open Ephys binary int16 samples
            data_size = z.getinfo(data_file).file_size
            if data_size % bytes_per_frame:
                raise ValueError(f"{folder}: binary byte length is not a multiple of {bytes_per_frame}")
            frames = data_size // bytes_per_frame
            timestamps = np.load(io.BytesIO(z.read(timestamps_file)),allow_pickle=False)
            numbers = np.load(io.BytesIO(z.read(samples_file)),allow_pickle=False)
            if len(timestamps) != frames or len(numbers) != frames:
                raise ValueError(f"{folder}: binary frame count does not match timestamps/sample_numbers")
            if frames < 2 or not np.isfinite(timestamps).all():
                raise ValueError(f"{folder}: too few frames or invalid timestamps")
            diffs = np.diff(numbers.astype(np.int64))
            nonsequential = int(np.count_nonzero(diffs != 1))
            backward = int(np.count_nonzero(np.diff(timestamps) < 0))
            reports.append(dict(stream=folder,channels=nc,sampling_rate_hz=fs,frames=frames,
                                duration_s=frames/fs,nonsequential_sample_steps=nonsequential,
                                backward_timestamp_steps=backward))
        if not any('Neuropix' in r['stream'] or 'OneBox' in r['stream'] for r in reports):
            raise ValueError('No Neuropixels stream found')
        if not any('/events/' in p and p.endswith('/timestamps.npy') for p in files):
            raise ValueError('No TTL event timestamp arrays found')
    return reports


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive',help='Original Open Ephys short-recording ZIP')
    args=p.parse_args()
    for r in check_archive(args.archive):
        print(f"{r['stream']}: {r['channels']} channels, {r['sampling_rate_hz']:g} Hz, "
              f"{r['frames']:,} frames, {r['duration_s']:.3f} s; "
              f"sample discontinuities={r['nonsequential_sample_steps']}, "
              f"backward timestamps={r['backward_timestamp_steps']}")
    print('Metadata, binary lengths, original sample numbers and timestamps verified.')

if __name__=='__main__':
    main()
