import os
os.system('sudo ln /dev/null /dev/raw1394')
os.environ.update({"PS_VERBOSE": "2"})
import mxnet as mx