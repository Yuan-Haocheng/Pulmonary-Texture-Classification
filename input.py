import os
import numpy as np
import pickle
import random


def getFileNameList(filePath):
    fileNameList = os.listdir(filePath)
    fileNameList = sorted(fileNameList, key=lambda x: x[:x.find('.')])

    return fileNameList


def load_CT(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    us = []
    file_list = getFileNameList(ROOT)

    for file_name in file_list:
        file_name = os.path.join(ROOT, file_name)
        with open(file_name, 'rb') as f:
            datadict = pickle.load(f)
            X = datadict[0]
            Y = datadict[1]
            U = datadict[2]

        xs.append(X)
        ys.append(Y)
        us.append(U)
        del X, Y, U

    Xtr = np.concatenate(xs)# 使变成行向量
    Xtr = Xtr.reshape(-1, 32, 32, 1)
    Ytr = np.concatenate(ys)
    Utr = np.concatenate(us)
    Utr = Utr.reshape(-1, 32, 32, 1)

    return Xtr, Ytr, Utr


def random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2*padding, oshape[1] + 2*padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                                    nw:nw + crop_shape[1]]
    return np.array(new_batch)


def random_flip_leftright(batch):
        for i in range(len(batch)):
            if bool(random.getrandbits(1)):
                batch[i] = np.fliplr(batch[i])
        return batch


def random_flip_updown(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.flipud(batch[i])
    return batch


def data_preprocessing(x_train,x_test):
    for i in range(x_train.shape[-1]):
        x_train[:, :, :, i] = (x_train[:, :, :, i] - np.mean(x_train[:, :, :, i])) / np.std(x_train[:, :, :, i])
        x_test[:, :, :, i] = (x_test[:, :, :, i] - np.mean(x_test[:, :, :, i])) / np.std(x_test[:, :, :, i])

    return x_train, x_test


def data_augmentation(batch):
    batch = random_flip_leftright(batch)
    batch = random_flip_updown(batch)
    batch = random_crop(batch, [32, 32], 4)
    return batch


def next_batch(img, label, batch_size, step):
    img_batch = img[step * batch_size:step * batch_size + batch_size]
    img_batch = data_augmentation(img_batch)
    lab_batch = label[step * batch_size:step * batch_size + batch_size]

    return img_batch, lab_batch


def shuffle_data(imgData, labData):
    index = np.random.permutation(len(imgData))
    shuffled_image = imgData[index]
    shuffled_label = labData[index]

    return shuffled_image, shuffled_label


if __name__ == '__main__':
    X, Y, U = load_CT('../DataPKL')
    print(X.shape)
    print(Y.shape)
    print(U.shape)

