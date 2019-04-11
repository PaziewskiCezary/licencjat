import os
import hashlib
import wget
import scipy.io
import sys

def valid_data(file_name, file_md5_hash):

    if not os.path.isfile(file_name):
        return False
    calculated_hash = hashlib.md5(open(file_name,'rb').read()).hexdigest()
    return calculated_hash == file_md5_hash

def download(url, out_file_name):
    wget.download(
        url,
        out=out_file_name,
        bar=wget.bar_thermometer
        )

def get_data_set(dataset, save_path):
    '''Takes Dataset namedtuple and downloads dataset'''
    if not os.path.isfile:
        os.mkdir(save_path)

    destination_path = os.path.join(save_path, dataset.file)
    if not valid_data(destination_path, dataset.hash) :
        print(f'Downloading {dataset.file} data')
        download(dataset.url, destination_path)
        print(f'"{dataset.file}" data downloaded!')

    return dataset.name, destination_path

def mkdir(path):
    os.makedirs(path, exist_ok=True)


def load_matlab_file(path):
    return scipy.io.loadmat(path)


def in_notebook():
    """
    Returns ``True`` if the module is running in IPython kernel,
    ``False`` if in IPython shell or other Python shell.
    """
    return 'ipykernel' in sys.modules
