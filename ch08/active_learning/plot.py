from pathlib import Path
from typing import Dict

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

sns.set_style("darkgrid")
sns.set_context("paper")


def plot(uuid: str, acquisition: str, ax=None):
    acq_name = acquisition.replace("_", " ")
    df = pd.read_csv(f"./models/{acquisition}/{uuid}/results.csv")[:-1]
    df = df.rename(columns={"accuracy": acq_name})
    df["n_samples"] = df["i"].apply(lambda x: x * 10 + 20)
    return df.plot.line(x="n_samples", y=acq_name, style=".-", figsize=(8, 5), ax=ax)


def get_imgs_per_label(model_dirs) -> Dict[int, np.ndarray]:
    imgs_per_label = {i: [] for i in range(10)}
    for model_dir in model_dirs:
        df = pd.read_parquet(model_dir / "added.parquet")
        df.image = df.image.apply(lambda x: x.reshape(28, 28).astype(np.uint8))
        for label in df.label.unique():
            dff = df[df.label == label]
            if len(dff) == 0:
                continue
            imgs_per_label[label].append(np.hstack(dff.image))
    return imgs_per_label


def get_added_images(acquisition: str, uuid: str, n_iter: int = 5) -> Image:
    base_dir = Path("./models") / acquisition / uuid
    model_dirs = filter(lambda x: x.is_dir(), base_dir.iterdir())
    model_dirs = sorted(model_dirs, key=lambda x: int(x.stem))
    imgs_per_label = get_imgs_per_label(model_dirs)
    imgs = []
    for i in range(10):
        label_img = np.hstack(imgs_per_label[i])[:, -(28 * n_iter) :]
        imgs.append(label_img)
    return Image.fromarray(np.vstack(imgs))


@click.command()
@click.option("--output-dir", default="output", help="Directory to save the output images")
@click.option("--uuid1", required=True, help="UUID for the first model")
@click.option("--uuid2", required=True, help="UUID for the second model")
@click.option("--acq1", required=True, help="Acquisition type for the first model")
@click.option("--acq2", required=True, help="Acquisition type for the second model")
def main(output_dir, uuid1, uuid2, acq1, acq2):
    ax = plot(uuid1, acq1)
    ax = plot(uuid2, acq2, ax=ax)
    plt.xticks(np.arange(0, 1050, 50))
    plt.yticks(np.arange(54, 102, 2))
    plt.ylabel("Accuracy")
    plt.xlabel("Number of acquired samples")
    plt.show()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    img1 = get_added_images(acq1, uuid1)
    img2 = get_added_images(acq2, uuid2)

    img1.save(output_path / f"{acq1}_added_images.png")
    img2.save(output_path / f"{acq2}_added_images.png")

    print(f"Images saved in {output_path}")


if __name__ == "__main__":
    main()
