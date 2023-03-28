import os
import pickle
import json
from collections import defaultdict
from typing import (
    List,
    Dict,
    Optional,
)

from my_dassl.data.datasets import (
    DATASET_REGISTRY,
    Datum, 
    DatasetBase,
)
from my_dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets


IGNORED = []
NEW_CNAMES = {}


@DATASET_REGISTRY.register()
class CLEVR(DatasetBase):

    dataset_dir = "CLEVR_v1.0"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_path = os.path.join(self.dataset_dir, "split_mlai_CLEVR.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        self.scene_dir = os.path.join(self.dataset_dir, 'scenes')
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train_val_conbimed = self.read_data(
                image_dir=self.image_dir,
                scene_path=os.path.join(self.scene_dir, 'CLEVR_train_scenes.json'),
                ignored=IGNORED,
                new_cnames=NEW_CNAMES,
            )
            train, val = OxfordPets.split_trainval(
                trainval=train_val_conbimed,
            )
            test = self.read_data(
                image_dir=self.image_dir,
                scene_path=os.path.join(self.scene_dir, 'CLEVR_val_scenes.json'),
                ignored=IGNORED,
                new_cnames=NEW_CNAMES,
            )
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

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

    def read_data(
        self,
        image_dir: str, 
        scene_path: str,
        ignored: List[str] = [], 
        new_cnames: Optional[Dict[str, str]] = None,
    ) -> List[Datum]:
        classes = self.load_paths_by_class(
            image_dir=image_dir,
            scene_path=scene_path, 
        )
        
        categories = [key for key in classes.keys() if key not in ignored]
        categories.sort()

        data = []
        for label, category in enumerate(categories):
            paths = classes[category] 

            if new_cnames is not None and category in new_cnames:
                category = new_cnames[category]
            
            data += [
                Datum(impath=path, label=label, classname=category)
                for path in paths
            ]
        
        return data

    def load_paths_by_class(
        self,
        image_dir: str, 
        scene_path: str
    ) -> Dict[str, List[str]]:
        with open(scene_path) as f:
            data = json.load(f)
            classes = defaultdict(list)
            scenes = data['scenes']
            for scene in scenes:
                num_objects = len(scene['objects'])
                file_name = scene['image_filename']
                split = scene['split']
                file_path = os.path.join(image_dir, split, file_name)
                classes[str(num_objects)].append(file_path)

        return classes