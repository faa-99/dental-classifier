import json

import splitfolders


def split_data():
    with open("../../config.json", "r", encoding="utf-8") as config_file:
        config = json.load(config_file)["dataset"]
    ratios = config["ratios"]
    train_r, val_r, test_r = (
        ratios["train_ratio"],
        ratios["val_ratio"],
        ratios["test_ratio"],
    )
    splitfolders.ratio(
        config["original_dataset_directory"],
        output=config["split_dataset_directory"],
        seed=1337,
        ratio=(train_r, val_r, test_r),
        group_prefix=None,
    )
