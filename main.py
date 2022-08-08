import tensorflow_datasets as tfds 
import tensorflow as tf 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import os

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

def load_dataset(PATH, SPLIT):
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
                  x, 
                  y, 
                  ):
    print(type(x), type(y))
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
    MODEL.save(f"{CONFIG['PATH']}/{CONFIG['MODEL_NAME']}")

def load_model(
               CONFIG: Dict[str, Any],
               ) -> tf.keras.Model:
    loaded_model = tf.keras.models.load_model(f"{CONFIG['PATH']}/{CONFIG['MODEL_NAME']}")
    return loaded_model

def train(
          DATA_PATH: str, 
          CONFIG: Dict[str, Any],
          SPLIT = 'train',
          ) -> None:
    get_mnist(DATA_PATH, SPLIT)
    train_dataset, _ = load_dataset(DATA_PATH, SPLIT)
    train_dataset = train_dataset.map(lambda x, y: preprocessing(x, y)).batch(CONFIG['BATCH_SIZE'])

    if CONFIG['OPTIMIZER'] == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate = CONFIG['LR'])
    model = build_model(CONFIG['MODEL_NAME'])
    model.compile(
                  optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['acc'],
                  )
    
    history = model.fit(
                        train_dataset, 
                        epochs=CONFIG['EPOCHS'],
                        verbose=1,
                        )
    save_model(
               model, 
               CONFIG,
               )

def test(
         DATA_PATH: str, 
         CONFIG: Dict[str, Any],
         SPLIT = 'test') -> None:    
    get_mnist(DATA_PATH, SPLIT)
    test_dataset, _ = load_dataset(DATA_PATH, SPLIT)
    test_dataset = test_dataset.map(lambda x, y: preprocessing(x, y)).batch(CONFIG['BATCH_SIZE'])

    model = load_model(CONFIG)
    test_loss, test_acc = model.evaluate(test_dataset)

    prediction = model.predict(test_dataset)

def main(stage='train') -> None:
    PATH = './data'
    basic_config = {
        "EPOCHS": 100,
        "BATCH_SIZE": 128,
        "LR": 0.01,
        "OPTIMIZER": 'adam',
        "LOSS_FN ": 'categorical_crossentropy',
        "MODEL_NAME": "DEMO_MNIST_MODEL",
        "MODEL_PATH": "./model",
    }
    train(PATH, basic_config)
    test(PATH, basic_config)
     
     

if __name__ == '__main__': 
    main()