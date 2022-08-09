import tensorflow as tf 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import os
import sys
import tensorflow_datasets as tfds 

from argparse import ArgumentParser

from typing import (
    List,
    Dict,
    Any,
)

from core.module import (
    get_mnist,
    load_dataset,
    preprocessing,
    build_model,
    save_model,
    load_model,
)



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
    test_dataset, _ = load_dataset(DATA_PATH, SPLIT)
    test_dataset = test_dataset.map(lambda x, y: preprocessing(x, y)).batch(CONFIG['BATCH_SIZE'])

    model = load_model(CONFIG)
    test_loss, test_acc = model.evaluate(test_dataset)

    prediction = model.predict(test_dataset)

def main(stage='train') -> None:
    DATA_PATH = f'{os.getcwd()}/data'
    config_dict = {
        "EPOCHS": 100,
        "BATCH_SIZE": 128,
        "LR": 0.01,
        "OPTIMIZER": 'adam',
        "MODEL_NAME": "DEMO_MNIST_MODEL",
        "MODEL_PATH": f"{os.getcwd()}/model",
    }
    AVAILABLE_STAGE = [
                       'prepare_data', 
                       'train', 
                       'test'
                       ]
    if stage not in AVAILABLE_STAGE:
        print("The selected stage is not available. Abort!")
        sys.exit()

    if stage=='prepare_data':
        SPLIT = "train+test"
        get_mnist(DATA_PATH, SPLIT)
    elif stage == 'train':
        train(DATA_PATH, config_dict)
    elif stage == 'test':
        test(DATA_PATH, config_dict)
    
     
     

if __name__ == '__main__': 
    parser = ArgumentParser()
    parser.add_argument("-s", "--stage", dest="stage", help="Define the stage to run.")
    args = parser.parse_args()
    main(stage=args.stage)