import datajoint as dj

dj.config["enable_python_native_blobs"] = True
dj.config["nnfabrik.schema_name"]= 'nnfabrik_core'
schema = dj.schema('nnfabrik_core')
from nnfabrik.main import Dataset

# first step: copy data from shared sinz folder to scratch
# Either use qcopy, tar xf /file/on/qb/volume or rclone

# prepare content of scratch folder /scratch/data, etc. if needed.

print("there are many entries in our dataset table: ", len(Dataset()))