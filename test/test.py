import os
import yaml 
import numpy as np

from core.module import (
    get_mnist,
    load_dataset,
    preprocessing,
    build_model,
    save_model,
    load_model,
)

def test_if_config_correct():
    with open(f"{os.getcwd()}/config/config.yml", 'r') as stream:
        config = yaml.load(stream, Loader=yaml.CLoader)
    
    FIRST_LEVEL_CONTENT = [
                           "hyperparameters", 
                           "model_configuration",
                           "data_configuration",
                           ]
    SECOND_LEVEL_CONTENT = {
                            "hyperparameters": [
                                                "EPOCHS",
                                                "BATCH_SIZE",
                                                "LR",
                                                "OPTIMIZER",
                                                "LOSS_FUNC",
                                                ],
                            "model_configuration": [
                                                    "MODEL_NAME",
                                                    "MODEL_PATH",
                                                    ],
                            "data_configuration": [
                                                   "PATH",
                                                   ],
                            }
    result = True if config.keys() in FIRST_LEVEL_CONTENT else False
    for key in FIRST_LEVEL_CONTENT:
        print(key)
        result = True if config[key].keys() in SECOND_LEVEL_CONTENT[key] else False 
    
    return result

def test_if_data_path_exists():
    if os.path.isdir(f"{os.getcwd()}/data"):
        return True
    else: 
        return False

def test_if_dataset_exists():
    if os.path.isdir(f"{os.getcwd()}/data/mnist/"):
        return True
    else: 
        return False

def test_if_preprocessing_work_properly():
    with open(f"{os.getcwd()}/config/config.yml", 'r') as stream:
        config = yaml.load(stream, Loader=yaml.CLoader)
    data, _ = load_dataset(
        f"{os.getcwd()}/data",
        "train",
    )

    data = data.map(lambda x, y: preprocessing(x, y)).batch(config['hyperparameters']['BATCH_SIZE'])

    result = True if len(list(data)) == round(60000/config['hyperparameters']['BATCH_SIZE']) else False

    for image, label in data.take(1):
        image = np.array(image).squeeze()
        result = True if image.shape ==(config['hyperparameters']['BATCH_SIZE'], 784) else False
        result = True if label.shape == (config['hyperparameters']['BATCH_SIZE'], 10) else False 

    return result
