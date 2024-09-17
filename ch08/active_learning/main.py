import uuid
from pathlib import Path

import click
import tensorflow as tf

from acquisition import acquisition_factory
from config import Config
from data import Data, get_data, get_initial_ds, update_ds
from metrics import get_accuracy
from model import build_model, get_callback
from utils import save_images_and_labels_added, save_results


@click.command()
@click.option(
    "--acquisition-type",
    type=click.Choice(
        [
            "knowledge_uncertainty",
            "random",
        ]
    ),
    default="random",
)
@click.option("--n-iter", type=int, default=100)
@click.option("--n-epochs", type=int, default=50)
@click.option("--n-samples-per-iter", type=int, default=10)
@click.option("--initial-n-samples", type=int, default=20)
@click.option("--n-total-samples", type=int, default=1000)
@click.option("--use-wandb", is_flag=True, help="Enable Weights & Biases logging")
def main(
    acquisition_type: str,
    n_iter: int,
    n_epochs: int,
    n_samples_per_iter: int,
    initial_n_samples: int,
    n_total_samples: int,
    use_wandb: bool,
):
    if use_wandb:
        import wandb

        wandb.init(
            project="active-learning",
            config={
                "acquisition_type": acquisition_type,
                "n_iter": n_iter,
                "n_epochs": n_epochs,
                "n_samples_per_iter": n_samples_per_iter,
                "initial_n_samples": initial_n_samples,
                "n_total_samples": n_total_samples,
            },
        )

    cfg = Config(
        initial_n_samples=initial_n_samples,
        n_total_samples=n_total_samples,
        n_epochs=n_epochs,
        n_samples_per_iter=n_samples_per_iter,
        acquisition_type=acquisition_type,
        n_iter=n_iter,
    )

    data: Data = get_initial_ds(get_data(), cfg.initial_n_samples)
    accuracies = {}
    added_indices = []

    run_uuid = str(uuid.uuid4())
    model_dir = Path("./models") / cfg.acquisition_type / run_uuid
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model directory: {model_dir}")

    for i in range(cfg.n_total_samples // cfg.n_samples_per_iter):
        print(f"Iteration: {i} / {cfg.n_total_samples // cfg.n_samples_per_iter}")
        iter_dir = model_dir / str(i)
        model = build_model()

        callbacks = [get_callback(iter_dir)]
        if use_wandb:
            wandb_callback = wandb.keras.WandbCallback(save_model=False)
            callbacks.append(wandb_callback)

        model.fit(
            x=data.x_train_al,
            y=data.y_train_al,
            validation_data=(data.x_test, data.y_test),
            epochs=cfg.n_epochs,
            callbacks=callbacks,
            verbose=2,
        )

        model = tf.keras.models.load_model(str(iter_dir / "model.keras"))
        indices_to_add = acquisition_factory(cfg.acquisition_type)(
            data.x_train,
            cfg.n_samples_per_iter,
            n_iter=cfg.n_iter,
            model=model,
        )
        added_indices.append(indices_to_add)
        data, (iter_x, iter_y) = update_ds(data, indices_to_add)

        save_images_and_labels_added(iter_dir, iter_x, iter_y)
        preds = model(data.x_test)
        accuracy = get_accuracy(data.y_test, preds)
        accuracies[i] = accuracy
        save_results(accuracies, added_indices, model_dir)

        if use_wandb:
            wandb.log(
                {
                    "iteration": i,
                    "accuracy": accuracy,
                    "train_size": len(data.x_train_al),
                }
            )

        print(f"Accuracy: {accuracy}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
