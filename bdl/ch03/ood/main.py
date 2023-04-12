from pathlib import Path
from typing import Optional

from bdl.ch03.ood.data import get_train_val_data, get_test_data, preprocess_image, DATA_ROOT, AUTOTUNE, MODEL_DIR
from bdl.ch03.ood.model import fit_model
import tensorflow as tf
import matplotlib.pyplot as plt


def main():
    train_dataset, validation_dataset = get_train_val_data()
    model = fit_model(train_dataset, validation_dataset)
    model.save(MODEL_DIR)
    accuracy = get_test_accuracy(model)
    print(accuracy)
    output_dir = Path(__file__).parent.parent.parent.parent / "figures" / "ch03" / "ood"
    output_dir.mkdir(parents=True, exist_ok=True)
    single_ood_example(model, output_path=output_dir / "parachute_example.png")
    ood_dataset_example(model, output_path=output_dir / "ood_scores.png")


def get_test_accuracy(model: tf.keras.Model) -> float:
    test_dataset, df_test = get_test_data()
    test_predictions = model.predict(test_dataset)
    softmax_scores = tf.nn.softmax(test_predictions, axis=1)
    df_test["predicted_label"] = tf.argmax(softmax_scores, axis=1)
    df_test["prediction_correct"] = df_test.apply(
        lambda x: x.predicted_label == x.breed, axis=1
    )
    accuracy = df_test.prediction_correct.value_counts(True)[True]
    return accuracy


def single_ood_example(model: tf.keras.Model, output_path: Optional[Path] = None):
    image_path = DATA_ROOT / "imagenette-160/val/n03888257/ILSVRC2012_val_00018229.JPEG"
    image = preprocess_image(str(image_path)).numpy()

    plt.figure(figsize=(5, 5))
    plt.imshow(image.astype(int))
    plt.axis("off")
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()

    logits = model.predict(tf.expand_dims(image, 0))
    dog_score = tf.nn.softmax(logits, axis=1)[0][1].numpy()
    print(f"Image classified as a dog with {dog_score:.4%} confidence")


def ood_dataset_example(model: tf.keras.Model, output_path: Optional[Path] = None):

    parachute_image_dir = DATA_ROOT / "imagenette-160/train/n03888257"
    parachute_image_paths = [
        str(filepath) for filepath in parachute_image_dir.iterdir()
    ]
    parachute_dataset = (tf.data.Dataset.from_tensor_slices(parachute_image_paths)
                         .map(lambda x: preprocess_image(x))
                         .batch(256)
                         .prefetch(buffer_size=AUTOTUNE))

    predictions = model.predict(parachute_dataset)
    dog_scores = tf.nn.softmax(predictions, axis=1)[:, 1]

    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(10, 5))
    plt.hist(dog_scores, bins=10)
    plt.xticks(tf.range(0, 1.1, 0.1))
    plt.grid()
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


if __name__ == '__main__':
    main()
