from collections import namedtuple

Dataset = namedtuple('Dataset', 'name file hash url')

OASBUD = [
    'OASBUD',
    'OASBUD.mat',
    'e2b770a6ee2f06ebe480ed0962252100',
    'https://zenodo.org/record/545928/files/OASBUD.mat?download=1'
    ]
OASBUD = Dataset(*OASBUD)

sets =  [
    value
    for (key,value) in locals().items()
    if isinstance(value, Dataset)
    ]
