
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from PIL import Image
from pathlib import Path
from typing import Dict
sns.set_style("darkgrid")
sns.set_context("paper")


def plot(uuid: str, acquisition: str, ax=None):
    acq_name = acquisition.replace("_", " ")
    df = pd.read_parquet(f"./models/{acquisition}/{uuid}/results.parquet")[:-1]
    df = df.rename(columns={"accuracy": acq_name})
    df["n_samples"] = df["i"].apply(lambda x: x*10 + 20)
    return df.plot.line(
        x="n_samples", y=acq_name, style='.-', figsize=(8,5), ax=ax
    )


def get_imgs_per_label(model_dirs) -> Dict[int, np.ndarray]:
    imgs_per_label = {i: [] for i in range(10)}
    for model_dir in model_dirs:
        df = pd.read_parquet(model_dir / "images_added.parquet")
        df.image = df.image.apply(
            lambda x: x.reshape(28, 28).astype(np.uint8)
        )
        for label in df.label.unique():
            dff = df[df.label == label]
            if len(dff) == 0:
                continue
            imgs_per_label[label].append(np.hstack(dff.image))
    return imgs_per_label


def get_added_images(
    acquisition: str, uuid: str, n_iter: int = 5
) -> Image:
    base_dir = Path("./models") / acquisition / uuid
    model_dirs = filter(lambda x: x.is_dir(), base_dir.iterdir())
    model_dirs = sorted(model_dirs, key=lambda x: int(x.stem))
    imgs_per_label = get_imgs_per_label(model_dirs)
    imgs = []
    for i in range(10):
        label_img = np.hstack(imgs_per_label[i])[:, -(28 * n_iter):]
        imgs.append(label_img)
    return Image.fromarray(np.vstack(imgs))


def main():
    ax = plot("bc1adec5-bc34-44a6-a0eb-fa7cb67854e4", "random")
    ax = plot(
        "5c8d6001-a5fb-45d3-a7cb-2a8a46b93d18", "knowledge_uncertainty", ax=ax
    )
    plt.xticks(np.arange(0, 1050, 50))
    plt.yticks(np.arange(54, 102, 2))
    plt.ylabel("Accuracy")
    plt.xlabel("Number of acquired samples")
    plt.show()

    uuid = "bc1adec5-bc34-44a6-a0eb-fa7cb67854e4"
    img_random = get_added_images("random", uuid)
    uuid = "5c8d6001-a5fb-45d3-a7cb-2a8a46b93d18"
    img_ku = get_added_images("knowledge_uncertainty", uuid)


if __name__ == "__main__":
    main()
