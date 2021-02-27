from typing import Tuple

import tensorflow as tf
import tensorflow_text as text

with open("./spm.model", "rb") as spm_model:
    tokenizer = text.SentencepieceTokenizer(spm_model.read(), add_bos=True, add_eos=True)

def make_model_input(x: tf.Tensor) -> tf.Tensor:
    x = text.normalize_utf8(x, "NFD")
    return tokenizer.tokenize(x)

def parse_batch_tsv_rows(x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    splits = tf.strings.split(x, sep="\t").to_tensor(shape=[tf.size(x), 3])
    model_inputs = make_model_input(splits[:, 1])
    labels = tf.strings.to_number(splits[:, 2])
    return model_inputs, labels

train_data = (
    tf.data.TextLineDataset("nsmc/ratings_train.txt")
    .skip(1)
    .shuffle(10000, reshuffle_each_iteration=True)
    .batch(64)
    .map(parse_batch_tsv_rows)
)
dev_data = train_data.take(100)
train_data = train_data.skip(100)
test_data = tf.data.TextLineDataset("nsmc/ratings_test.txt").skip(1).batch(256).map(parse_batch_tsv_rows)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=[None], dtype=tf.int32, ragged=True),
        tf.keras.layers.Embedding(5000, 256),
        tf.keras.layers.LSTM(256),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(2, activation="softmax"),
    ]
)
model.summary()
model.compile(optimizer="rmsprop", loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics="acc")
model.fit(train_data, validation_data=dev_data, epochs=3)
model.evaluate(test_data)

@tf.function(input_signature=[tf.TensorSpec([None], dtype=tf.string)])
def call(x: tf.Tensor) -> tf.Tensor:
    model_input = make_model_input(x)
    return model(model_input)

model.tokenizer = tokenizer
tf.saved_model.save(model, 'nsmc-model/0', call)
