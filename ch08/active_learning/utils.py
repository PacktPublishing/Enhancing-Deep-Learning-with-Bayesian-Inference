from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def save_images_and_labels_added(output_path: Path, iter_x: np.ndarray, iter_y: np.ndarray):
    df = pd.DataFrame()
    df["label"] = np.argmax(iter_y, axis=1)
    iter_x_normalised = (np.squeeze(iter_x, axis=-1) * 255).astype(np.uint8)
    df["image"] = iter_x_normalised.reshape(10, 28 * 28).tolist()
    df.to_parquet(output_path / "added.parquet", index=False)


def save_results(accuracies: Dict[int, float], added_indices: List[int], model_dir: Path):
    df = pd.DataFrame(accuracies.items(), columns=["i", "accuracy"])
    df["added"] = added_indices
    df.to_csv(f"{model_dir}/results.csv", index=False)
