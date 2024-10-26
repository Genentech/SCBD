from enum import Enum


class Task(Enum):
    BATCH_CORRECTION = "batch_correction"
    DOMAIN_GENERALIZATION = "domain_generalization"


class Dataset(Enum):
    CAMELYON17 = "camelyon17"
    CMNIST = "cmnist"
    FUNK22 = "funk22"
    RXRX1 = "rxrx1"


class DataSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class EncoderType(Enum):
    RESNET18 = "resnet18"
    RESNET50 = "resnet50"
    DENSENET121 = "densenet121"


class ExperimentGroup(Enum):
    TREATMENT = "treatment"
    CONTROL = "control"


class EType(Enum):
    NONE = "none"
    PLATE_WELL = "plate_well"
    PLATE_WELL_TILE = "plate_well_tile"