#!/usr/bin/env python
import sys
from typing import Any

try:
    import h5py
except Exception as e:
    print('IMPORT_ERROR:', repr(e))
    sys.exit(2)

def main(path: str) -> None:
    try:
        f = h5py.File(path, 'r')
    except Exception as e:
        print('OPEN_ERROR:', repr(e))
        sys.exit(3)
    print('FILE:', path)
    try:
        print('ATTRS:', {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()})
    except Exception:
        print('ATTRS: <unavailable>')

    summaries = []

    def walk(name: str, obj: Any):
        if isinstance(obj, h5py.Dataset):
            try:
                shape = obj.shape
                dtype = str(obj.dtype)
                dattrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in obj.attrs.items()}
            except Exception:
                shape = 'UNKNOWN'
                dtype = 'UNKNOWN'
                dattrs = {}
            summaries.append((name, shape, dtype, dattrs))

    f.visititems(walk)

    print('\nTOP_LEVEL_KEYS:')
    for k in f.keys():
        obj = f[k]
        kind = 'Group' if isinstance(obj, h5py.Group) else 'Dataset'
        print(f'- {k} ({kind})')

    print('\nDATASETS (up to 100):')
    for i, (name, shape, dtype, dattrs) in enumerate(summaries[:100]):
        print(f'[{i:02d}] {name} | shape={shape} | dtype={dtype} | attrs={dattrs}')

    print(f'\nTOTAL_DATASETS: {len(summaries)}')
    f.close()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: inspect_hdf5.py <path_to_hdf5>')
        sys.exit(1)
    main(sys.argv[1])
