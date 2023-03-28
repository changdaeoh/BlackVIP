import os
import pickle
from collections import defaultdict
from pathlib import Path

from my_dassl.data.datasets import (
    DATASET_REGISTRY,
    Datum, 
    DatasetBase,
)
from my_dassl.utils import mkdir_if_missing

from ..oxford_pets import OxfordPets
import datasets.colour_biased_mnist.colour_biased_mnist_original as colour_biased_mnist


@DATASET_REGISTRY.register()
class ColourBiasedMNIST(DatasetBase):

    dataset_dir = "colour_biased_mnist"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.train_rho = cfg.DATASET.COLOUR_BIASED_MNIST.TRAIN_RHO
        self.test_rho = cfg.DATASET.COLOUR_BIASED_MNIST.TEST_RHO
        self.train_n_confusing_labels = cfg.DATASET.COLOUR_BIASED_MNIST.TRAIN_N_CONFUSING_LABELS
        self.test_n_confusing_labels = cfg.DATASET.COLOUR_BIASED_MNIST.TEST_N_CONFUSING_LABELS
        self.use_test_as_val = cfg.DATASET.COLOUR_BIASED_MNIST.USE_TEST_AS_VAL
        self.randomize = cfg.DATASET.COLOUR_BIASED_MNIST.RANDOMIZE

        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.train_image_dir = os.path.join(
            self.dataset_dir, 
            f"images_rho={self.train_rho}_ncl={self.train_n_confusing_labels}_rand={self.randomize}",
            "train",
        )
        self.test_image_dir = os.path.join(
            self.dataset_dir, 
            f"images_rho={self.test_rho}_ncl={self.test_n_confusing_labels}_rand={self.randomize}",
            "test",
        )
        self.split_path = os.path.join(
            self.dataset_dir, 
            "split_mlai_colour_biased_mnist_rho={}_{}_ncl={}{}.json".format(
                    self.train_rho,
                    self.test_rho,
                    self.train_n_confusing_labels,
                    self.test_n_confusing_labels,
                ),
        )
        self.split_fewshot_dir = os.path.join(
            self.dataset_dir, 
            "split_fewshot_val={}_rho={}_{}_ncl={}{}".format(
                    self.use_test_as_val,
                    self.train_rho,
                    self.test_rho,
                    self.train_n_confusing_labels,
                    self.test_n_confusing_labels,
            ),
        )
        mkdir_if_missing(self.split_fewshot_dir)

        train_dir = Path(self.train_image_dir)
        if not train_dir.exists():
            train_and_val_raw = self._load_colour_biased_mnist(
                train=True, 
                rho=self.train_rho,
                n_confusing_labels=self.train_n_confusing_labels,
                randomize=self.randomize,
            )
            self._save_images(train_and_val_raw, train_dir)
        
        test_dir = Path(self.test_image_dir)
        if not test_dir.exists():
            test_raw = self._load_colour_biased_mnist(
                train=False,
                rho=self.test_rho,
                n_confusing_labels=self.test_n_confusing_labels,
                randomize=self.randomize,
            )
            self._save_images(test_raw, test_dir)

        test = self.read_data(self.test_image_dir) 
       
        if self.use_test_as_val:
            train = self.read_data(self.train_image_dir) 
            val = self.read_data(self.test_image_dir)
            test = self.read_data(self.test_image_dir)
        else:
            if not os.path.exists(self.split_path):
                train_and_val = self.read_data(self.train_image_dir) 
                train, val = OxfordPets.split_trainval(train_and_val)
                OxfordPets.save_split(
                    train=train, 
                    val=val, 
                    test=test, 
                    filepath=self.split_path, 
                    path_prefix=self.dataset_dir,
                )
            else:
                train, val, test_raw = OxfordPets.read_split(self.split_path, self.dataset_dir)

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
                if not self.use_test_as_val:
                    val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)
        super().__init__(train_x=train, val=val, test=test)
    
    def _load_colour_biased_mnist(
        self, 
        train: bool, 
        rho: float, 
        n_confusing_labels: int, 
        randomize: bool,
    ):
        return colour_biased_mnist.ColourBiasedMNIST(
            root=self.dataset_dir,
            train=train,
            transform=None,
            target_transform=None,
            download=True,
            data_label_correlation=rho,
            n_confusing_labels=n_confusing_labels,
            randomize=randomize,
        )
    
    def _save_images(
        self, 
        mnist_dataset: colour_biased_mnist.ColourBiasedMNIST, 
        dest_dir: str
    ):
        image_name_counter = defaultdict(int)
        for image, label, _ in mnist_dataset:
            image_dir = Path(dest_dir) / str(label)
            image_dir.mkdir(parents=True, exist_ok=True)
            image_name = f'{image_name_counter[label]}.png'
            image_path = image_dir / image_name
            image.save(image_path)
            image_name_counter[label] += 1
    
    def read_data(self, image_dir: str):
        data = []
        image_paths = Path(image_dir).rglob('*')
        for image_path in image_paths:
            if not image_path.is_file():
                continue
            datum = Datum(
                impath=str(image_path.resolve()),
                label=int(str(image_path.parent.name)),
                classname=f'{image_path.parent.name}'
            )
            data.append(datum)
        return data