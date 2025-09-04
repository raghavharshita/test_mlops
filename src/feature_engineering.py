import pandas as pd
import os
import yaml
import logging
from sklearn.feature_extraction.text import TfidfVectorizer

import logging
import os

log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

#logging configuration
logger=logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,'feature_engineering.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_data(file_path:str)->pd.DataFrame:
    """
    Load data from a CSV file.
    """
    try:
        df=pd.read_csv(file_path)
        df.fillna('',inplace=True)
        logger.debug(f'Data loaded and NaN filled from {file_path}')
        return df
    except pd.errors.ParserError as e:
        logger.error(f'Error parsing CSV file: {e}')
        raise
    except Exception as e:
        logger.error(f'Error loading data: {e}')
        raise

def apply_tfidf(train_data:pd.DataFrame,test_data:pd.DataFrame,max_features:int)->tuple:
    """Apply TF-IDF vectorization to text data."""
    try:
        vectorizer=TfidfVectorizer(max_features=max_features)
        x_train=train_data['text'].values
        y_train=train_data['target'].values
        x_test=test_data['text'].values 
        y_test=test_data['target'].values

        x_train_bow=vectorizer.fit_transform(x_train)
        x_test_bow=vectorizer.transform(x_test)

        train_df=pd.DataFrame(x_train_bow.toarray())
        train_df['label']=y_train

        test_df=pd.DataFrame(x_test_bow.toarray())
        test_df['label']=y_test

        logger.debug('TF-IDF vectorization applied successfully')
        return train_df,test_df
    except Exception as e:
        logger.error(f'Error in TF-IDF vectorization: {e}')
        raise


def save_data(df:pd.DataFrame,file_path:str)->None:
    """Save DataFrame to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        df.to_csv(file_path,index=False)
        logger.debug(f'Data saved to {file_path}')
    except Exception as e:
        logger.error(f'Error saving data: {e}')
        raise

def main():
    try:
        params=load_params(params_path='params.yaml')
        max_features=params['feature_engineering']['max_features']
        # max_features=50
        train_data=load_data('./data/processed/train.csv')
        test_data=load_data('./data/processed/test.csv')

        train_df,test_df=apply_tfidf(train_data,test_data,max_features)

        save_data(train_df,os.path.join('./data','featured','train_tfidf.csv'))
        save_data(test_df,os.path.join('./data','featured','test_tfidf.csv'))

        logger.info('Feature engineering completed successfully')
    except Exception as e:
        logger.error(f'Error in main execution: {e}')
        raise

if __name__=='__main__':
    main()

