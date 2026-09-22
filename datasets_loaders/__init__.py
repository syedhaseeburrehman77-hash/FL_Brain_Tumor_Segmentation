from .registry import DATASET_REGISTRY


def create_dataset(name: str, root_dir: str, **kwargs):

    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")

    return DATASET_REGISTRY[name](root_dir,**kwargs)