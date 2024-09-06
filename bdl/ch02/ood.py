import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from ch02.train import AUTOTUNE, preprocess_image

def display_image(image_path):
    image = preprocess_image(image_path).numpy()
    plt.figure(figsize=(5,5))
    plt.imshow(image.astype(int))
    plt.axis("off")
    plt.show()

def predict_dog_score(model, image):
    logits = model.predict(tf.expand_dims(image, 0))
    dog_score = tf.nn.softmax(logits, axis=1)[0][1].numpy()
    print(f"Image classified as a dog with {dog_score:.4%} confidence")
    return dog_score

def create_parachute_dataset(image_dir):
    parachute_image_paths = [str(filepath) for filepath in Path(image_dir).iterdir()]
    return (tf.data.Dataset.from_tensor_slices(parachute_image_paths)
            .map(lambda x: preprocess_image(x))
            .batch(256)
            .prefetch(buffer_size=AUTOTUNE))

def plot_dog_score_histogram(dog_scores):
    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(10,5))
    plt.hist(dog_scores, bins=10)
    plt.xticks(tf.range(0, 1.1, 0.1))
    plt.grid()
    plt.show()

def main():
    # Load the model (assuming it's defined elsewhere)
    model = tf.keras.models.load_model("path/to/your/model")

    # Display and classify single image
    image_path = "imagenette-160/val/n03888257/ILSVRC2012_val_00018229.JPEG"
    display_image(image_path)
    image = preprocess_image(image_path)
    predict_dog_score(model, image)

    # Process parachute dataset
    parachute_image_dir = "imagenette-160/train/n03888257"
    parachute_dataset = create_parachute_dataset(parachute_image_dir)

    # Get predictions for parachute dataset
    predictions = model.predict(parachute_dataset)
    dog_scores = tf.nn.softmax(predictions, axis=1)[:, 1]

    # Plot histogram of dog scores
    plot_dog_score_histogram(dog_scores)

if __name__ == "__main__":
    main()
