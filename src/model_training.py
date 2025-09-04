import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
import yaml

import logging
import os

log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

#logging configuration
logger=logging.getLogger('model_training')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,'model_training.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path:str)->pd.DataFrame:
    """
    Load data from a CSV file.
    """
    try:
        df=pd.read_csv(file_path)
        logger.debug(f'Data loaded from {file_path}')
        return df
    except pd.errors.ParserError as e:
        logger.error(f'Error parsing CSV file: {e}')
        raise
    except FileNotFoundError as e:
        logger.error(f'File not found: {e}')
        raise
    except Exception as e:
        logger.error(f'Error loading data: {e}')
        raise

def train_model(x_train:np.ndarray,y_train:np.ndarray,params:dict)->RandomForestClassifier:
    """
    Train the RandomForest model.
    
    :param X_train: Training features
    :param y_train: Training labels
    :param params: Dictionary of hyperparameters
    :return: Trained RandomForestClassifier
    """

    try:
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError("The number of samples in X_train and y_train must be the same.")
        logger.debug(f'Starting model training with parameters: {params}')

        clf=RandomForestClassifier(n_estimators=params['n_estimators'],random_state=params['random_state'])
        logger.debug(f'Fitting the model with {x_train.shape[0]} samples')

        clf.fit(x_train,y_train)
        logger.debug('Model training completed successfully')

        return clf
    except ValueError as e:
        logger.error('ValueError during model training: %s', e)
        raise
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise

def save_model(model:RandomForestClassifier,model_path:str)->None:
    """
    Save the trained model to a file.
    
    :param model: Trained model object
    :param file_path: Path to save the model file
    """

    try:
        os.makedirs(os.path.dirname(model_path),exist_ok=True)

        with open(model_path,'wb') as file:
            pickle.dump(model,file)
        logger.debug(f'Model saved successfully at {model_path}')
    except FileNotFoundError as e:
        logger.error('File path not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise

def main():
    try:
        params={'n_estimators':25,'random_state':42}
        train_data=load_data('./data/featured/train_tfidf.csv')
        x_train=train_data.drop(columns=['label']).values
        y_train=train_data['label'].values

        clf=train_model(x_train,y_train,params)
        model_path='models/model.pkl'
        save_model(clf,model_path)

        logger.debug('Model training process completed successfully')
    except Exception as e:
        logger.error(f'Error in model training process: {e}')
        raise

if __name__=='__main__':
    main()

