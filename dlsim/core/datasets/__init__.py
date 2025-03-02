import os
from typing import Optional, Dict

from dlsim.core.datasets.Dataset import Dataset
from dlsim.core.mappings import Linear
from dlsim.core.session_settings import SessionSettings


def create_dataset(settings: SessionSettings, participant_index: int = 0, train_dir: Optional[str] = None, test_dir: Optional[str] = None) -> Dataset:
    mapping = Linear(1, settings.target_participants)
    if settings.dataset == "cifar10":
        from dlsim.core.datasets.CIFAR10 import CIFAR10
        return CIFAR10(participant_index, 0, mapping, settings.partitioner,
                       train_dir=train_dir, test_dir=test_dir, shards=settings.target_participants, alpha=settings.alpha)
    elif settings.dataset == "celeba":
        from dlsim.core.datasets.Celeba import Celeba
        img_dir = None
        if train_dir:
            img_dir = os.path.join(train_dir, "..", "..", "data", "raw", "img_align_celeba")
        elif test_dir:
            img_dir = os.path.join(test_dir, "..", "raw", "img_align_celeba")
        return Celeba(participant_index, 0, mapping, train_dir=train_dir, test_dir=test_dir, images_dir=img_dir)
    elif settings.dataset == "femnist":
        from dlsim.core.datasets.Femnist import Femnist
        return Femnist(participant_index, 0, mapping, train_dir=train_dir, test_dir=test_dir)
    else:
        raise RuntimeError("Unknown dataset %s" % settings.dataset)
