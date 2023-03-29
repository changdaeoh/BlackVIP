import os.path as osp
from torchvision.datasets import SVHN

from my_dassl.utils import mkdir_if_missing

def extract_and_save_image(dataset, save_dir, categories):
    if osp.exists(save_dir):
        print('Folder "{}" already exists'.format(save_dir))
        return

    print('Extracting images to "{}" ...'.format(save_dir))
    mkdir_if_missing(save_dir)

    for i in range(len(dataset)):
        img, label = dataset[i]
        # class_dir = osp.join(save_dir, str(label).zfill(3))
        class_dir = osp.join(save_dir, categories[label])
        mkdir_if_missing(class_dir)
        impath = osp.join(class_dir, str(i + 1).zfill(5) + ".jpg")
        img.save(impath)

def download_and_prepare(name, root):
    print("Dataset: {}".format(name))
    print("Root: {}".format(root))
    
    if name == "svhn":
        train = SVHN(root, split="train", download=True)
        test = SVHN(root, split="test", download=True)
    else:
        raise ValueError
    
    train_dir = osp.join(root, name, "train")
    test_dir = osp.join(root, name, "test")

    if name == 'svhn':
        categories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    else:
        categories = train.classes

    extract_and_save_image(train, train_dir, categories)
    extract_and_save_image(test, test_dir, categories)

if __name__ == "__main__":
    download_and_prepare("svhn", 'YOURPATH')