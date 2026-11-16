import io
import json
import zipfile
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_test_data import check_archive


def make_sample_zip(path, *, truncate=False):
    r='session/Record Node 1/experiment1/recording1'
    samples=np.arange(100,105,dtype=np.int64)
    times=np.arange(5,dtype=np.float64)/30000
    def array_bytes(a):
        b=io.BytesIO();np.save(b,a);return b.getvalue()
    with zipfile.ZipFile(path,'w') as z:
        z.writestr(f'{r}/structure.oebin',json.dumps({'continuous':[
          {'folder_name':'Neuropix-PXI-X.ProbeA/','num_channels':2,'sample_rate':30000}]}))
        data=(np.arange(10,dtype='<i2')).tobytes()
        z.writestr(f'{r}/continuous/Neuropix-PXI-X.ProbeA/continuous.dat',data[:-4] if truncate else data)
        z.writestr(f'{r}/continuous/Neuropix-PXI-X.ProbeA/timestamps.npy',array_bytes(times))
        z.writestr(f'{r}/continuous/Neuropix-PXI-X.ProbeA/sample_numbers.npy',array_bytes(samples))
        z.writestr(f'{r}/events/Neuropix-PXI-X.ProbeA/TTL/timestamps.npy',array_bytes(times[:2]))


def test_valid_short_recording(tmp_path):
    path=tmp_path/'test.zip';make_sample_zip(path)
    rows=check_archive(path)
    assert len(rows)==1
    assert rows[0]['frames']==5
    assert rows[0]['nonsequential_sample_steps']==0


def test_missing_zip(tmp_path):
    with pytest.raises(FileNotFoundError):
        check_archive(tmp_path/'missing.zip')


def test_mismatched_frame_counts(tmp_path):
    path=tmp_path/'truncated.zip';make_sample_zip(path,truncate=True)
    with pytest.raises(ValueError,match='does not match'):
        check_archive(path)
