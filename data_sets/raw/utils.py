import os
import hashlib
import wget

def valid_data(file_name, file_md5_hash):

    if not os.path.isfile(file_name):
        return False
    calculated_hash = hashlib.md5(open(file_name,'rb').read()).hexdigest()
    return calculated_hash == file_md5_hash

def download(url, out_file_name):
  wget.download(
    url,
    out=out_file_name,
    bar=wget.bar_thermometer)
