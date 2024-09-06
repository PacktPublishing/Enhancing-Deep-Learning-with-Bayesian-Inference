import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Union


IMG_SIZE = (160, 160)
AUTOTUNE = tf.data.AUTOTUNE

def load_and_preprocess_data(file_path: str, is_test: bool = False) -> Union[Tuple[pd.Series, pd.Series, pd.Series, pd.Series], pd.DataFrame]:
    df = pd.read_csv(file_path, sep=" ")
    df.columns = ["path", "species", "breed", "ID"]
    df["breed"] = df.breed.apply(lambda x: x - 1)
    df["path"] = df["path"].apply(
        lambda x: f"/content/oxford-iiit-pet/images/{x}.jpg"
    )
    if not is_test:
        return train_test_split(df["path"], df["breed"], test_size=0.2, random_state=0)
    return df

@tf.function
def preprocess_image(filename: tf.Tensor) -> tf.Tensor:
    raw = tf.io.read_file(filename)
    image = tf.image.decode_png(raw, channels=3)
    return tf.image.resize(image, IMG_SIZE)

@tf.function
def preprocess(filename: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    return preprocess_image(filename), tf.one_hot(label, 2)

def create_dataset(paths: Union[pd.Series, tf.Tensor], labels: Union[pd.Series, tf.Tensor]) -> tf.data.Dataset:
    return (tf.data.Dataset.from_tensor_slices((paths, labels))
            .map(lambda x, y: preprocess(x, y))
            .batch(256)
            .prefetch(buffer_size=AUTOTUNE))

def get_model() -> tf.keras.Model:
    IMG_SHAPE = IMG_SIZE + (3,)
    base_model = tf.keras.applications.ResNet50(
        input_shape=IMG_SHAPE, include_top=False, weights='imagenet'
    )
    base_model.trainable = False
    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = tf.keras.applications.resnet50.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(2)(x)
    return tf.keras.Model(inputs, outputs)

def compile_model(model: tf.keras.Model) -> tf.keras.Model:
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

def train_model(model: tf.keras.Model, train_dataset: tf.data.Dataset, validation_dataset: tf.data.Dataset, epochs: int = 3) -> tf.keras.callbacks.History:
    return model.fit(train_dataset, epochs=epochs, validation_data=validation_dataset)

def evaluate_model(model: tf.keras.Model, test_dataset: tf.data.Dataset) -> tf.Tensor:
    test_predictions = model.predict(test_dataset)
    softmax_scores = tf.nn.softmax(test_predictions, axis=1)
    return tf.argmax(softmax_scores, axis=1)

def calculate_accuracy(df_test: pd.DataFrame, predicted_labels: tf.Tensor) -> float:
    df_test["predicted_label"] = predicted_labels
    df_test["prediction_correct"] = df_test.apply(
        lambda x: x.predicted_label == x.breed, axis=1
    )
    return df_test.prediction_correct.value_counts(True)[True]

def main() -> None:
    # Load and preprocess training data
    paths_train, paths_val, labels_train, labels_val = load_and_preprocess_data("oxford-iiit-pet/annotations/trainval.txt")
    
    # Create datasets
    train_dataset = create_dataset(paths_train, labels_train)
    validation_dataset = create_dataset(paths_val, labels_val)
    
    # Create and compile model
    model = get_model()
    model = compile_model(model)
    
    # Train model
    train_model(model, train_dataset, validation_dataset)
    
    # Load and preprocess test data
    df_test = load_and_preprocess_data("oxford-iiit-pet/annotations/test.txt", is_test=True)
    test_dataset = create_dataset(df_test["path"], df_test["breed"])
    
    # Evaluate model
    predicted_labels = evaluate_model(model, test_dataset)
    
    # Calculate and print accuracy
    accuracy = calculate_accuracy(df_test, predicted_labels)
    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
