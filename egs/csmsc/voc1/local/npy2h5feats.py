import sys
sys.path.insert(0, '../../../')
import os
from parallel_wavegan.utils import write_hdf5
import numpy as np

inputnpy=sys.argv[1]

if os.path.isdir(inputnpy):
    for filebase in os.listdir(inputnpy):
        if not filebase.endswith('.npy'):
            continue
        dirname=inputnpy
        filename=filebase.split('.')[0]

        print(os.path.join(dirname, filebase))
        mel=np.load(os.path.join(dirname, filebase))
        write_hdf5(os.path.join(dirname, f"{filename}.h5"), "feats", mel.astype(np.float32))
else:
    dirname=os.path.dirname(inputnpy)
    filename=os.path.basename(inputnpy).split('.')[0]

    mel=np.load(inputnpy)
    write_hdf5(os.path.join(dirname, f"{filename}.h5"), "feats", mel.astype(np.float32))
