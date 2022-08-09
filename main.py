import tensorflow as tf 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import os
import sys
import tensorflow_datasets as tfds 
import yaml

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
          CONFIG: Dict[str, Any],
          SPLIT = 'train',
          ) -> None:

    train_dataset, _ = load_dataset(
                                    CONFIG['data_configuration']['PATH'], 
                                    SPLIT,
                                    )
    train_dataset = train_dataset.map(lambda x, y: preprocessing(x, y)).batch(CONFIG['hyperparameters']['BATCH_SIZE'])

    if CONFIG['hyperparameters']['OPTIMIZER'] == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate = CONFIG['hyperparameters']['LR'])

    model = build_model(CONFIG['model_configuration']['MODEL_NAME'])
    model.compile(
                  optimizer=optimizer,
                  loss=CONFIG['hyperparameters']['LOSS_FUNC'],
                  metrics=['acc'],
                  )
    
    history = model.fit(
                        train_dataset, 
                        epochs=CONFIG['hyperparameters']['EPOCHS'],
                        verbose=1,
                        )
    save_model(
               model, 
               CONFIG,
               )

def test(
         CONFIG: Dict[str, Any],
         SPLIT = 'test') -> None:    

    test_dataset, _ = load_dataset(
                                   CONFIG['data_configuration']['PATH'], 
                                   SPLIT,
                                   )
    test_dataset = test_dataset.map(lambda x, y: preprocessing(x, y)).batch(CONFIG['hyperparameters']['BATCH_SIZE'])

    model = load_model(CONFIG)
    test_loss, test_acc = model.evaluate(test_dataset)

    prediction = model.predict(test_dataset)

def main(stage='train') -> None:
    with open(f"{os.getcwd()}/config/config.yml", 'r') as stream:
        config = yaml.load(stream, Loader=yaml.CLoader)

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
        get_mnist(config['data_configuration']['PATH'], SPLIT)
    elif stage == 'train':
        train(config)
    elif stage == 'test':
        test(config)
    
     
     

if __name__ == '__main__': 
    parser = ArgumentParser()
    parser.add_argument("-s", "--stage", dest="stage", help="Define the stage to run.")
    args = parser.parse_args()
    main(stage=args.stage)