import pytest

class Test_Datasets:
    def test_csv_dataset_exercise_test(self):
        from imm.datasets.csv_dataset import CSVDataset
        # TODO: rewrite this test with a tempdir
        dset = CSVDataset(
            "/export/home/sabraun/code/imm/data/datasets/exercise_dataset/",
            "/export/home/sabraun/code/imm/data/datasets/exercise_dataset/denseposed_csv/denseposed_instance_level_test_split.csv",
            id_col_name="id",
            fname_col_name="im1",
            subset="test",
            dataset="csv"
        )
        image_pair = next(dset.sample_image_pair())
        assert len(dset) == 5816


    def test_csv_dataset_exercise_train(self):
        from imm.datasets.csv_dataset import CSVDataset
        # TODO: rewrite this test with a tempdir
        dset = CSVDataset(
            "/export/home/sabraun/code/imm/data/datasets/exercise_dataset/",
            "/export/home/sabraun/code/imm/data/datasets/exercise_dataset/csvs/instance_level_train_split.csv",
            id_col_name="id",
            fname_col_name="im1",
            subset="train",
            dataset="csv"
        )
        image_pair = next(dset.sample_image_pair())
        assert len(dset) == 40338

    def test_csv_dataset_max_num_samples(self):
        dset = CSVDataset(
            "/export/home/sabraun/code/imm/data/datasets/exercise_dataset/",
            "/export/home/sabraun/code/imm/data/datasets/exercise_dataset/denseposed_csv/denseposed_instance_level_test_split.csv",
            id_col_name="id",
            fname_col_name="im1",
            subset="test",
            dataset="csv",
            max_samples=10,
            order_stream=True
        )

        for i, t in enumerate(dset.sample_image_pair()):
            pass
        assert i == 2000