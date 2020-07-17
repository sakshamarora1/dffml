from __future__ import print_function, division

import os
import pathlib
import hashlib
from typing import Any, Tuple, AsyncIterator, List, Type, Dict
import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import models


from dffml.record import Record
from dffml.model.accuracy import Accuracy
from dffml.util.entrypoint import entrypoint
from dffml.base import config, field
from dffml.feature.feature import Feature, Features
from dffml.source.source import Sources, SourcesContext
from dffml.model.model import ModelContext, Model, ModelNotTrained
from .pytorch_utils import NumpyToTensor


@config
class ResNet18ModelConfig:
    predict: Feature = field("Feature name holding classification value")
    classifications: List[str] = field("Options for value of classification")
    features: Features = field("Features to train on")
    directory: pathlib.Path = field("Directory where state should be saved")
    clstype: Type = field("Data type of classifications values", default=str)
    imageSize: int = field(
        "Common size for all images to resize and crop to", default=None
    )
    useCUDA: bool = field("Utilize GPUs for processing", default=False)
    epochs: int = field(
        "Number of iterations to pass over all records in a source", default=20
    )
    trainable: bool = field(
        "Tweak pretrained model by training again", default=False
    )
    batch_size: int = field("Batch size", default=10)
    validation_split: float = field(
        "Split training data for Validation", default=0.0
    )
    # criterion: str = field("Loss function", default="CrossEntropyLoss")
    # optimizer: str = field("Optimizer used by model", default="SGD")
    # metrics: str = field("Scheduler", default="")

    def __post_init__(self):
        self.classifications = list(map(self.clstype, self.classifications))


class ResNet18ModelContext(ModelContext):
    def __init__(self, parent):
        super().__init__(parent)
        self.cids = self._mkcids(self.parent.config.classifications)
        self.classifications = self._classifications(self.cids)
        self.features = self._applicable_features()
        self.model_dir_path = self._model_dir_path()
        self._model = None
        if self.parent.config.useCUDA and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.logger.info("Using CUDA")
        else:
            self.device = torch.device("cpu")

    async def __aenter__(self):
        path = self._model_dir_path()
        if os.path.isfile(os.path.join(path, "model.pt")):
            self.logger.info(f"Using saved model from {path}")
            self._model = torch.load(os.path.join(path, "model.pt"))
        else:
            self._model = self.createModel()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        pass

    def _classifications(self, cids):
        """
        Map classifications to numeric values
        """
        classifications = {value: key for key, value in cids.items()}
        self.logger.debug(
            "classifications(%d): %r", len(classifications), classifications
        )
        return classifications

    @property
    def classification(self):
        return self.parent.config.predict.name

    def _applicable_features(self):
        return [name for name in self.parent.config.features.names()]

    @property
    def model(self):
        """
        Generates or loads a model
        """
        return self._model

    @model.setter
    def model(self, model):
        """
        Loads a model if already trained previously
        """
        self._model = model

    def createModel(self):
        """
        Generates a model
        """
        if self._model is not None:
            return self._model
        self.logger.debug(
            "Loading model with classifications(%d): %r",
            len(self.classifications),
            self.classifications,
        )

        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.require_grad = self.parent.config.trainable
        # num_features = model.classifier[-1].in_features
        # features = list(model.classifier.children())[:-1]
        # model.classifier = nn.Sequential(*features)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, len(self.classifications)),
            nn.LogSoftmax(dim=1),
        )
        # nn.Linear(model.fc.in_features, len(self.classifications))
        self._model = model.to(self.device)

        return self._model

    def _model_dir_path(self):
        if self.parent.config.directory is None:
            return None
        _to_hash = self.features + [
            self.classification,
            str(len(self.cids)),
        ]
        model = hashlib.sha384("".join(_to_hash).encode("utf-8")).hexdigest()
        if not os.path.isdir(self.parent.config.directory):
            raise NotADirectoryError(
                "%s is not a directory" % (self.parent.config.directory)
            )
        os.makedirs(
            os.path.join(self.parent.config.directory, model), exist_ok=True
        )
        return os.path.join(self.parent.config.directory, model)

    def _mkcids(self, classifications):
        """
        Create an index, possible classification mapping and sort the list of
        classifications first.
        """
        cids = dict(
            zip(range(0, len(classifications)), sorted(classifications))
        )
        self.logger.debug("cids(%d): %r", len(cids), cids)
        return cids

    async def dataset_generator(self, sources: Sources):
        self.logger.debug("Training on features: %r", self.features)
        x_cols: Dict[str, Any] = {feature: [] for feature in self.features}
        y_cols = []
        all_records = []
        all_sources = sources.with_features(
            self.features + [self.classification]
        )

        async for record in all_sources:
            if record.feature(self.classification) in self.classifications:
                all_records.append(record)
        for record in all_records:
            for feature, results in record.features(self.features).items():
                x_cols[feature].append(np.array(results))
            y_cols.append(
                self.classifications[record.feature(self.classification)]
            )
        if (len(self.features)) > 1:
            self.logger.critical(
                "Found more than one feature to train on. Only first feature will be used"
            )
        if not y_cols:
            raise ValueError("No records to train on")

        y_cols = np.array(y_cols)
        for feature in x_cols:
            x_cols[feature] = np.array(x_cols[feature])

        self.logger.info("------ Record Data ------")
        self.logger.info("x_cols:    %d", len(list(x_cols.values())[0]))
        self.logger.info("y_cols:    %d", len(y_cols))
        self.logger.info("-----------------------")

        x_cols = x_cols[self.features[0]]
        dataset = NumpyToTensor(x_cols, y_cols)

        return dataset, len(dataset)

    async def prediction_data_generator(self, data):
        dataset = NumpyToTensor([data])
        dataloader = torch.utils.data.DataLoader(dataset)
        return dataloader

    async def train(self, sources: Sources):
        dataset, size = await self.dataset_generator(sources)
        size = {
            "Training": size - int(self.parent.config.validation_split * size),
            "Validation": int(self.parent.config.validation_split * size),
        }

        if self.parent.config.validation_split:
            data = dict(
                zip(
                    ["Training", "Validation"],
                    list(
                        torch.utils.data.random_split(
                            dataset, [size["Training"], size["Validation"]]
                        )
                    ),
                )
            )
            self.logger.info(
                "Data split into Traning set: {} and Validation set: {}".format(
                    size["Training"], size["Validation"]
                )
            )
            dataloaders = {
                x: torch.utils.data.DataLoader(
                    data[x],
                    batch_size=self.parent.config.batch_size,
                    shuffle=True,
                )
                for x in ["Training", "Validation"]
            }
        else:
            dataloaders = {
                "Training": torch.utils.data.DataLoader(
                    dataset,
                    batch_size=self.parent.config.batch_size,
                    shuffle=True,
                )
            }

        # Metrics
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            self._model.fc.parameters(), lr=0.001, momentum=0.9
        )
        exp_lr_scheduler = lr_scheduler.StepLR(
            optimizer, step_size=7, gamma=0.1
        )
        since = time.time()

        best_model_wts = copy.deepcopy(self._model.state_dict())
        best_acc = 0.0

        for epoch in range(self.parent.config.epochs):
            self.logger.debug(
                "Epoch {}/{}".format(epoch + 1, self.parent.config.epochs)
            )
            self.logger.debug("-" * 10)

            for phase in dataloaders.keys():
                if phase == "Training":
                    self._model.train()
                else:
                    self._model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "Training"):
                        outputs = self._model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == "Training":
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == "Training":
                    exp_lr_scheduler.step()

                epoch_loss = running_loss / size[phase]
                epoch_acc = running_corrects.double() / size[phase]

                self.logger.debug(
                    "{} Loss: {:.4f} Acc: {:.4f}".format(
                        phase, epoch_loss, epoch_acc
                    )
                )

                if phase == "Validation" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self._model.state_dict())

            self.logger.debug("")

        time_elapsed = time.time() - since
        self.logger.debug(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )

        if self.parent.config.validation_split:
            self.logger.debug("Best Validation Acc: {:4f}".format(best_acc))
            self._model.load_state_dict(best_model_wts)

        torch.save(
            self._model, os.path.join(self._model_dir_path(), "model.pt")
        )

    async def accuracy(self, sources: Sources) -> Accuracy:
        if not os.path.isfile(os.path.join(self.model_dir_path, "model.pt")):
            raise ModelNotTrained("Train model before assessing for accuracy.")

        dataset, size = await self.dataset_generator(sources)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.parent.config.batch_size, shuffle=True
        )

        self._model.eval()
        running_corrects = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(inputs)
            labels = labels.to(inputs)
            with torch.set_grad_enabled(False):
                outputs = self._model(inputs)
                _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.data)
            acc = running_corrects.double() / size

        return Accuracy(acc)

    async def predict(
        self, sources: SourcesContext
    ) -> AsyncIterator[Tuple[Record, Any, float]]:
        """
        Uses trained data to make a prediction about the quality of a record.
        """
        if not os.path.isfile(os.path.join(self.model_dir_path, "model.pt")):
            raise ModelNotTrained("Train model before prediction.")

        self._model.eval()
        async for record in sources.with_features(self.features):
            feature_data = record.features(self.features)[self.features[0]]
            predict = await self.prediction_data_generator(feature_data)
            target = self.parent.config.predict.name

            with torch.no_grad():
                for val in predict:
                    val = val.to(self.device)

                    outputs = self._model(val)
                    prob = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, prediction_value = prob.topk(1, dim=1)
                    record.predicted(
                        target, self.cids[prediction_value.item()], confidence,
                    )

            yield record


@entrypoint("resnet18")
class ResNet18Model(Model):

    CONFIG = ResNet18ModelConfig
    CONTEXT = ResNet18ModelContext
