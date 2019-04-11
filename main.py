from utils import get_data_set, mkdir, load_matlab_file, in_notebook
from datasets import sets
import os

from models import unet, data_gen_small, Adam, dice_coef

from usg import reconstruct

if in_notebook():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

## GETING DATASETS

# home_dir = os.getenv("HOME")
# sets_path = os.path.join(home_dir, 'licencjat/dane')
# mkdir(sets_path)

# datas = []
# for set_ in sets:
#     datas.append(get_data_set(set_, sets_path))

# for name, data_path in datas:
#     path = os.path.join(sets_path, name)
#     mkdir(path)

#     if name == 'OASBUD':
#         data = load_matlab_file(data_path)['data'][0]

#     reconstruct(data, path)


from sklearn.model_selection import train_test_split

data_dir = '/home/supciorr/licencjat/dane/OASBUD/photos/'
mask_dir = '/home/supciorr/licencjat/dane/OASBUD/masks/'
all_images = os.listdir(data_dir)
train_images, validation_images = train_test_split(all_images, train_size=0.8, test_size=0.2)

train_gen = data_gen_small(data_dir, mask_dir, train_images, 5, [128, 128])
img, msk = next(train_gen)


model = unet(filters=64)
model.summary()

model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=[dice_coef])
model.fit_generator(train_gen, steps_per_epoch=5, epochs=10)

model.save_weights('unet_w')
train_gen = data_gen_small(data_dir, mask_dir, train_images, 5, [128, 128])
model.predict_generator(train_gen)