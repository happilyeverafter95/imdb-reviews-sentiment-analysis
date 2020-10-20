import logging
import re
import string
import time
from typing import Tuple, Union, List, Dict

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

level = logging.INFO
logging.basicConfig(level=level)
logger = logging.getLogger(__name__)


class TFModel(tf.Module):
    def __init__(self, model: tf.keras.Model) -> None:
        self.model = model

    @tf.function(input_signature=[tf.TensorSpec(shape=(1, ), dtype=tf.string)])
    def prediction(self, review: str) -> Dict[str, Union[str, List[float]]]:
        return {'prediction': self.model(review),
                'description': 'prediction ranges from 0 (negative) to 1 (positive)'}


class ModelTrainer:
    def __init__(self) -> None:
        self.tf_model_wrapper: TFModel

        # Model Architecture parameters
        self.embed_size = 128
        self.max_features = 20000
        self.epochs = 1
        self.batch_size = 128
        self.max_len = 500

    def fetch_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        train_data, test_data = tfds.load(name='imdb_reviews', split=['train', 'test'],
                                          batch_size=-1, as_supervised=True)

        train_examples, train_labels = tfds.as_numpy(train_data)
        logger.info(f'There are {train_examples.shape[0]} reviews in the training set')
        test_examples, test_labels = tfds.as_numpy(test_data)
        logger.info(f'There are {test_examples.shape[0]} reviews in the testing set')
        return train_examples, train_labels, test_examples, test_labels

    def custom_preprocessing(self, raw_text: str) -> tf.string:
        lowercase = tf.strings.lower(raw_text)
        stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
        return tf.strings.regex_replace(stripped_html, "[%s]" % re.escape(string.punctuation), '')

    def init_vectorize_layer(self, text_dataset: np.ndarray) -> TextVectorization:
        text_vectorizer = TextVectorization(max_tokens=self.max_features,
                                            standardize=self.custom_preprocessing,
                                            output_mode='int',
                                            output_sequence_length=self.max_len)
        text_vectorizer.adapt(text_dataset)
        return text_vectorizer

    def init_model(self, text_dataset: np.ndarray) -> tf.keras.Model:
        vectorize_layer = self.init_vectorize_layer(text_dataset)
        raw_input = tf.keras.Input(shape=(1,), dtype=tf.string)
        x = vectorize_layer(raw_input)
        x = tf.keras.layers.Embedding(self.max_features + 1, self.embed_size)(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(100, return_sequences=True))(x)
        x = tf.keras.layers.GlobalMaxPooling1D()(x)
        predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(raw_input, predictions)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self) -> None:
        train_examples, train_labels, _, _ = self.fetch_data()
        model = self.init_model(train_examples)
        model.fit(train_examples, train_labels, epochs=self.epochs, batch_size=self.batch_size)
        self.tf_model_wrapper = TFModel(model)
        tf.saved_model.save(self.tf_model_wrapper.model, f'classifier/saved_models/{int(time.time())}',
                            signatures={'serving_default': self.tf_model_wrapper.prediction})
        logger.info('saving SavedModel to saved_models/1')


if __name__ == '__main__':
    model_trainer = ModelTrainer()
    model_trainer.train()
