from .utils import *

from .config import sets_path
from .datasets import sets as datasets

import os
def get_data(dataset):
    '''Takes Dataset namedtuple and downloads dataset'''
    if not valid_data(dataset.file, dataset.hash) :
      print(f'Downloading {dataset.file} data')
      download(dataset.url, dataset.file)
      print(f'"{dataset.file}" data downloaded!')

old_path = os.path.curdir
old_path = os.path.abspath(old_path)
dest_dir = os.path.join(os.path.curdir, 'sets_path')
print(dest_dir)
if not os.path.isdir(dest_dir):
    os.mkdir(dest_dir)
os.chdir(dest_dir)

sets = {}
for set_ in datasets:
    get_data(set_)
    path = os.path.join(os.path.curdir, sets_path)
    path = os.path.abspath(path)
    sets[set_.name] = path

os.chdir(old_path)