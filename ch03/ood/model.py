import tensorflow as tf

from bdl.ch03.ood.data import IMG_SIZE


def get_model():
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


def fit_model(train_dataset, validation_dataset) -> tf.keras.Model:
    model = get_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    model.fit(train_dataset, epochs=3, validation_data=validation_dataset)
    return model
