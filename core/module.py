import tensorflow_datasets as tfds 
import tensorflow as tf 

from typing import (
    List,
    Dict,
    Any,
)

def get_mnist( 
              PATH: str,
              SPLIT: str,
              **kargs: Dict[str, Any],
              ) -> bool:
    try:
        mnist = tfds.load(
            name='mnist',
            split=SPLIT,
            data_dir=PATH,
        )
    
    except:
        print("Failed to load mnist dataset!")

def load_dataset(
                 PATH, 
                 SPLIT,
                 ):
    dataset, dataset_info = tfds.load(
            name='mnist',
            split=SPLIT,
            data_dir=PATH,
            download=False,
            with_info=True, 
            as_supervised=True,
        )
    return dataset, dataset_info


def preprocessing(
                  x: tf.Tensor, 
                  y: tf.Tensor, 
                  ):
    x = x/255
    x = tf.reshape(x, [784])

    y = tf.one_hot(y, 10)  
    return x, y

def build_model(MODEL_NAME: str) -> tf.keras.Model:
    model = tf.keras.Sequential([
                                 tf.keras.layers.InputLayer(input_shape=(784)), 
                                 tf.keras.layers.Dense(128, activation='relu'), 
                                 tf.keras.layers.Dense(64, activation='relu'), 
                                 tf.keras.layers.Dense(10, activation='softmax'),
                                 ],
                                 name=MODEL_NAME, 
                                 )

    model.summary()
    return model

def save_model(
               MODEL: tf.keras.Model,
               CONFIG: Dict[str, Any],
               ) -> None:
    MODEL.save(f"{CONFIG['MODEL_PATH']}/{CONFIG['MODEL_NAME']}")

def load_model(
               CONFIG: Dict[str, Any],
               ) -> tf.keras.Model:
    loaded_model = tf.keras.models.load_model(f"{CONFIG['MODEL_PATH']}/{CONFIG['MODEL_NAME']}")
    return loaded_model

