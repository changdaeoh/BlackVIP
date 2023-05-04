import os
import argparse
import numpy as np
import cv2 as cv

import shutil
import errno
from torchvision.datasets import MNIST


def mkdir_if_missing(dirname):
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def extract_and_save_image(dataset, save_dir, categories):
    if os.path.exists(save_dir):
        print('Folder "{}" already exists'.format(save_dir))
        return

    print('Extracting images to "{}" ...'.format(save_dir))
    mkdir_if_missing(save_dir)

    for i in range(len(dataset)):
        img, label = dataset[i]
        # class_dir = osp.join(save_dir, str(label).zfill(3))
        class_dir = os.path.join(save_dir, categories[label])
        mkdir_if_missing(class_dir)
        impath = os.path.join(class_dir, str(i + 1).zfill(5) + ".jpg")
        img.save(impath)


def download_and_prepare(name, root):
    print("Dataset: {}".format(name))
    print("Root: {}".format(root))

    train = MNIST(root, train=True, download=True)
    test = MNIST(root, train=False, download=True)

    train_dir = os.path.join(root, name, "train")
    test_dir = os.path.join(root, name, "test")

    categories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    extract_and_save_image(train, train_dir, categories)
    extract_and_save_image(test, test_dir, categories)


def listdir_nohidden(path, sort=False):
    items = [f for f in os.listdir(path) if not f.startswith(".")]
    if sort:
        items.sort()
    return items


def synthetic(args, split):
    np.random.seed(args.seed)
    seed = args.seed
    r_size = args.r_size
    f_size = args.f_resize

    for i in range(10):
        data_dir = f"{args.data_root}/locmnist_source/MNIST/{split}/{i}"
        categories = listdir_nohidden(data_dir)
        
        for j in range(len(categories)):
            impath = os.path.join(data_dir, categories[j])
            img = cv.imread(impath, 0)
            img = cv.resize(img, None, fx=r_size, fy=r_size)
            
            img_size = int(28 * r_size)

            a = np.random.randint(224 - img_size + 1)
            if (a == 0) or (a == int(224 - img_size)):
                b = np.random.randint(224 - img_size + 1)
            else:
                b = (int(224 - img_size)) * np.random.randint(2)

            black = np.zeros([224, 224])
            if np.random.randint(2) == 0:
                black[a:a + img_size, b:b + img_size] = img
            else:
                black[b:b + img_size, a:a + img_size] = img

            fake_class = np.random.randint(10)
            fake_dir = f"{args.data_root}/locmnist_source/MNIST/{split}/{fake_class}"
            fake_categories = listdir_nohidden(fake_dir)
            fake_idx = np.random.randint(len(fake_categories))
            fake_path = os.path.join(fake_dir, fake_categories[fake_idx])
            fake_img = cv.imread(fake_path, 0)
            fake_img = cv.resize(fake_img, None, fx=f_size, fy=f_size)

            low = int(112 - (28 * f_size / 2))
            high = int(112 + (28 * f_size / 2))
            black[low:high, low:high] = fake_img

            syn_dir = f"{args.data_root}/locmnist_r{r_size}_f{f_size}/{split}/{i}"
            if not os.path.exists(syn_dir):
                os.makedirs(syn_dir)
            cv.imwrite(os.path.join(syn_dir, categories[j]), black)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("syn")
    parser.add_argument('--seed', type=int, default=1, help='')
    parser.add_argument('--data_root', type=str, default='./data', help='')
    parser.add_argument('--r_size', type=int, default=1, help='real digit size ratio')
    parser.add_argument('--f_size', type=int, default=1, help='fake digit size ratio', choices=[1,4])
    args = parser.parse_args()
    
    src_data_dir = f'{args.data_root}/locmnist_source'
    download_and_prepare("MNIST", src_data_dir)

    synthetic(args, split='train')
    synthetic(args, split='test')

    shutil.rmtree(src_data_dir, ignore_errors=True)