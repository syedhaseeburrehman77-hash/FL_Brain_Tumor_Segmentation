from .fets_dataset import FeTSDatasetLoader

DATASET_REGISTRY = {
    "fets2022": FeTSDatasetLoader,
    "brain_tumor": FeTSDatasetLoader,  # alias for backwards compatibility
}