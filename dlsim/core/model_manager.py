import logging
import os
from typing import Dict, Optional, List

import torch
import torch.nn as nn

from dlsim.core.gradient_aggregation import GradientAggregationMethod
from dlsim.core.gradient_aggregation.fedavg import FedAvg
from dlsim.core.model_trainer import ModelTrainer
from dlsim.core.models import unserialize_model, serialize_model
from dlsim.core.session_settings import SessionSettings


class ModelManager:
    """
    This class manages the current ML model and training.
    """

    def __init__(self, model: Optional[nn.Module], settings: SessionSettings, participant_index: int):
        self.model: nn.Module = model
        self.settings: SessionSettings = settings
        self.participant_index: int = participant_index
        self.logger = logging.getLogger(self.__class__.__name__)

        dataset_base_path: str = self.settings.dataset_base_path or os.environ["HOME"]
        if self.settings.dataset in ["cifar10", "mnist", "google_speech"]:
            self.data_dir = os.path.join(dataset_base_path, "dfl-data")
        else:
            # The LEAF dataset
            self.data_dir = os.path.join(dataset_base_path, "leaf", self.settings.dataset)

        self.model_trainer: ModelTrainer = ModelTrainer(self.data_dir, self.settings, self.participant_index)

        # Keeps track of the incoming trained models as aggregator
        self.incoming_trained_models: Dict[bytes, nn.Module] = {}

    def process_incoming_trained_model(self, peer_pk: bytes, incoming_model: nn.Module):
        if peer_pk in self.incoming_trained_models:
            # We already processed this model
            return

        self.incoming_trained_models[peer_pk] = incoming_model

    def reset_incoming_trained_models(self):
        self.incoming_trained_models = {}

    def get_aggregation_method(self):
        if self.settings.gradient_aggregation == GradientAggregationMethod.FEDAVG:
            return FedAvg

    def aggregate_trained_models(self, weights: List[float] = None) -> Optional[nn.Module]:
        models = [model for model in self.incoming_trained_models.values()]
        return self.get_aggregation_method().aggregate(models, weights=weights)

    async def train(self) -> int:
        samples_trained_on = await self.model_trainer.train(self.model, device_name=self.settings.train_device_name)

        # Detach the gradients
        self.model = unserialize_model(serialize_model(self.model), self.settings.dataset, architecture=self.settings.model)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return samples_trained_on
