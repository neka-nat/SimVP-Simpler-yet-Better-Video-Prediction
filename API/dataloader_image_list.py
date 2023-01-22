import os
from io import BytesIO, IOBase
import numpy as np
from PIL import Image, ImageCms
import h5py

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


srgb_p = ImageCms.createProfile("sRGB")
lab_p  = ImageCms.createProfile("LAB")
rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")


def load_list(path, root):
    tuples = []
    for line in open(path):
        pair = line.strip().split(",")
        tuples.append([os.path.join(root, p) for p in pair])
    return tuples


def read_image(path, size, ch, c_space="RGB"):
    if isinstance(path, IOBase) or os.path.splitext(path)[-1] == ".jpg":
        img = Image.open(path)
        img_resize = img.resize(size)
        w, h = size
        if c_space == "RGB":
            if ch == 1: # gray scale mode
                img_gray = img_resize.convert("L")
                image = np.asarray(img_gray).reshape((h, w, 1)).transpose(2, 0, 1).astype(np.float32)
            elif ch == 4: # color + gray scale mode
                img_gray = img_resize.convert("L")
                color_array = np.asarray(img_resize).transpose(2, 0, 1).astype(np.float32)
                gray_array = np.asarray(img_gray).reshape((h, w, 1)).transpose(2, 0, 1).astype(np.float32)
                image = np.concatenate([color_array, gray_array], 0)
            else: # color mode
                image = np.asarray(img_resize).transpose(2, 0, 1).astype(np.float32)
        elif c_space == "LAB":
            # Convert to Lab color rspace
            img_conv = ImageCms.applyTransform(img_resize, rgb2lab)
            image = np.asarray(img_conv).transpose(2, 0, 1).astype(np.float32)
        else:
            img_conv = img_resize.convert(c_space)
            image = np.asarray(img_conv).transpose(2, 0, 1).astype(np.float32)
        image /= 255
        return image
    else:
        data = np.load(path)
        return data


class ImageListDataset(Dataset):
    def __init__(self, img_size=(160, 128), input_len=20, channels=3):
        self.img_w = img_size[0]
        self.img_h = img_size[1]
        self.input_len = input_len
        self.img_ch = channels
        self.img_paths = None
        self.mode = None
        self.c_space = "RGB"
        self.mean = 0
        self.std = 1

    def load_images(self, img_paths, c_space="RGB"):
        self.img_paths = img_paths
        self.mode = "img" if os.path.splitext(self.img_paths[0][0])[-1] == ".jpg" else "audio"
        self.c_space = c_space

    def __getitem__(self, index):
        # print("target_idx: ", index)
        # print(self.img_paths[int(index * self.input_len):int((index + 1) * self.input_len)])
        # print(self.img_paths[int((index + 1) * self.input_len)])
        assert self.img_paths is not None

        X = np.ndarray((1, self.input_len, self.img_ch, self.img_h, self.img_w), dtype=np.float32)
        X[0] = [
            read_image(path[0], (self.img_w, self.img_h), self.img_ch, self.c_space)
            for path in self.img_paths[int(index * self.input_len):int((index + 1) * self.input_len)]
        ]
        if len(self.img_paths[0]) > 1:
            y = np.array([[
                read_image(self.img_paths[int((index + 1) * self.input_len) - 1][1], (self.img_w, self.img_h), self.img_ch, self.c_space)
            ]])
        else:
            y = np.array([[
                read_image(self.img_paths[int((index + 1) * self.input_len)][0], (self.img_w, self.img_h), self.img_ch, self.c_space)
            ]])
        return X.reshape(self.input_len, self.img_ch, self.img_h, self.img_w), y.reshape(1, self.img_ch, self.img_h, self.img_w)

    def __len__(self):
        return len(self.img_paths[:-self.input_len:self.input_len])


class ImageHDF5Dataset(Dataset):
    def __init__(self, img_size=(160, 128), input_len=20, channels=3):
        self.img_w = img_size[0]
        self.img_h = img_size[1]
        self.input_len = input_len
        self.img_ch = channels
        self.hf_data = None
        self.mode = None
        self.c_space = "RGB"

    def load_hdf5(self, h5path, c_space="RGB"):
        self.hf_data = h5py.File(h5path)
        self.mode = "img" if os.path.splitext(list(self.hf_data.keys())[0])[-1] == ".jpg" else "audio"
        self.c_space = c_space

    def __getitem__(self, index):
        assert self.hf_data is not None

        keys_list = list(self.hf_data.keys())
        X = np.ndarray((1, self.input_len, self.img_ch, self.img_h, self.img_w), dtype=np.float32)
        X[0] = [read_image(BytesIO(np.array(self.hf_data[key])), (self.img_w, self.img_h), self.img_ch, self.c_space) for key in keys_list[int(index * self.input_len):int((index + 1) * self.input_len)]]
        y = np.array([[read_image(BytesIO(np.array(self.hf_data[keys_list[int((index + 1) * self.input_len)]])), (self.img_w, self.img_h), self.img_ch, self.c_space)]])
        return X.reshape(self.input_len, self.img_ch, self.img_h, self.img_w), y.reshape(1, self.img_ch, self.img_h, self.img_w)

    def __len__(self):
        return len(list(self.hf_data.keys())[:-self.input_len:self.input_len])


def load_data(
        batch_size, val_batch_size,
        data_root, num_workers):

    train_list = load_list(data_root + 'image_list/data/train_list.txt', os.path.join(data_root, 'image_list'))
    test_list = load_list(data_root + 'image_list/data/test_list.txt', os.path.join(data_root, 'image_list'))

    train_set = ImageListDataset()
    test_set = ImageListDataset()
    train_set.load_images(train_list)
    test_set.load_images(test_list)

    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    dataloader_test = torch.utils.data.DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    return dataloader_train, None, dataloader_test, 0, 1


if __name__ == "__main__":
    def load_list(path, root):
        tuples = []
        for line in open(path):
            pair = line.strip().split()
            tuples.append(os.path.join(root, pair[0]))
        return tuples

    img_paths = load_list("data/train_list.txt", ".")
    dataset = ImageListDataset()
    dataset.load_images(img_paths)
    print("data len:", len(dataset))

    from torch.utils.data import DataLoader
    data_loader = DataLoader(dataset, batch_size=20, shuffle=True)
    for data in tqdm(data_loader):
        print(data.shape)
