import os
import random
import pickle

from my_dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from my_dassl.utils import mkdir_if_missing, listdir_nohidden, write_json, read_json

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class LocMNIST(DatasetBase):
    dataset_dir = 'locmnist'

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        tailpath = f'_r{cfg.DATASET.LOCMNIST.R_SIZE}_f{cfg.DATASET.LOCMNIST.F_SIZE}'
        self.dataset_dir = os.path.join(root, self.dataset_dir + tailpath)
        self.image_dir = self.dataset_dir
        self.train_image_dir = os.path.join(self.dataset_dir, 'train')
        self.test_image_dir = os.path.join(self.dataset_dir, 'test')
        self.split_path = os.path.join(self.dataset_dir, "split_LocMNIST.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, 'split_fewshot')
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val = self.read_split(self.split_path, self.train_image_dir)
            _, _, test = self.read_and_split_data(self.train_image_dir, self.test_image_dir)
        else:
            train, val, test = self.read_and_split_data(self.train_image_dir, self.test_image_dir)
            self.save_split(train, val, self.split_path, self.train_image_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")

            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}

                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)

    @ staticmethod
    def read_and_split_data(train_image_dir, test_image_dir, p_trn=0.8, ignored=[], new_cnames=None):
        categories = listdir_nohidden(train_image_dir)
        categories = [c for c in categories if c not in ignored]
        categories.sort()

        p_val = 1 - p_trn
        print(f"Splitting into {p_trn:.0%} train, {p_val:.0%} val")

        def _collate(ims, y, c):
            items = []
            for im in ims:
                item = Datum(impath=im, label=y, classname=c)
                items.append(item)
            return items

        train, val, test = [], [], []
        for label, category in enumerate(categories):
            category_dir = os.path.join(train_image_dir, category)
            images = listdir_nohidden(category_dir)
            images = [os.path.join(category_dir, im) for im in images]
            random.shuffle(images)

            test_images = listdir_nohidden(os.path.join(test_image_dir, category))
            test_images = [os.path.join(os.path.join(test_image_dir, category), im) for im in test_images]
            random.shuffle(test_images)

            n_total = len(images)
            n_train = round(n_total * p_trn)
            n_val = round(n_total * p_val)
            assert n_train > 0 and n_val > 0

            if new_cnames is not None and category in new_cnames:
                category = new_cnames[category]

            train.extend(_collate(images[:n_train], label, category))
            val.extend(_collate(images[n_train:n_train + n_val], label, category))
            test.extend(_collate(test_images, label, category))

        return train, val, test

    @staticmethod
    def read_split(filepath, path_prefix):
        def _convert(items):
            out = []
            for impath, label, classname in items:
                impath = os.path.join(path_prefix, impath)
                item = Datum(impath=impath, label=int(label), classname=classname)
                out.append(item)
            return out

        print(f"Reading split from {filepath}")
        split = read_json(filepath)

        train = _convert(split["train"])
        val = _convert(split["val"])

        return train, val

    @ staticmethod
    def save_split(train, val, filepath, path_prefix):
        def _extract(items):
            out = []

            for item in items:
                impath = item.impath
                label = item.label
                classname = item.classname
                impath = impath.replace(os.path.join(path_prefix), "")

                if impath.startswith("/"):
                    impath = impath[1:]
                out.append((impath, label, classname))

            return out

        train = _extract(train)
        val = _extract(val)
        split = {"train": train, "val": val}

        write_json(split, filepath)
        print(f"Saved split to {filepath}")
