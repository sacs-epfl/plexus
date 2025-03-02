import os

from dlsim.core.datasets import create_dataset
from dlsim.core.session_settings import SessionSettings


class ModelEvaluator:
    """
    Contains the logic to evaluate the accuracy of a given model on a test dataset.
    """

    def __init__(self, data_dir: str, settings: SessionSettings):
        if settings.dataset in ["cifar10", "mnist", "movielens", "google_speech"]:
            test_dir = data_dir
        else:
            test_dir = os.path.join(data_dir, "data", "test")
        self.dataset = create_dataset(settings, test_dir=test_dir)

    def evaluate_accuracy(self, model, device_name: str = "cpu"):
        return self.dataset.test(model, device_name)
