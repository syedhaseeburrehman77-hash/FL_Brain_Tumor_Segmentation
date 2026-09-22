import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from datasets_loaders.fets_dataset import FeTSDatasetLoader


def test_dataset_length():
    loader = FeTSDatasetLoader(
        root_dir="C:/Users/syedh/Documents/BraT_Segmentation_FeTs2022/MICCAI_FeTS2022_TrainingData",
        partition_csv="C:/Users/syedh/Documents/BraT_Segmentation_FeTs2022/MICCAI_FeTS2022_TrainingData/partitioning_1.csv",
    )
    groups = loader._get_partitioned_groups()
    total_cases = sum(len(records) for _, records in groups)

    # In FeTS 2022: 23 institutions and 1,251 labelled 3D MRI scans
    assert len(groups) == 23
    assert total_cases == 1251