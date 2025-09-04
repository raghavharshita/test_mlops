import os
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import yaml
from dvclive import Live

import logging
import os

log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

#logging configuration
logger=logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,'model_evaluation.log')
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

def load_model(model_path:str):
    """Load model from a given path."""
    try:
        with open(model_path,'rb') as file:
            model=pickle.load(file)

        logger.debug(f'model loaded from {model_path}')
        return model
    except FileNotFoundError:
        logger.error('File not found: %s', model_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model: %s', e)
        raise

def load_data(file_path:str)->pd.DataFrame:
    """Load data from a CSV file."""
    try:
        data=pd.read_csv(file_path)
        logger.debug(f'Data loaded from {file_path}')
        return data
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def evaluate_model(model,x_test:np.ndarray,y_test:np.ndarray)->dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        y_pred=model.predict(x_test)
        y_pred_proba=model.predict_proba(x_test)[:,1]

        accuracy=accuracy_score(y_test,y_pred)
        precision=precision_score(y_test,y_pred)
        recall=recall_score(y_test,y_pred)  
        roc_auc_score_value=roc_auc_score(y_test,y_pred_proba)

        metric_dict={
            'accuracy':accuracy,
            'precision':precision,  
            'recall':recall,
            'roc_auc_score':roc_auc_score_value
        }

        logger.debug('Model evaluation metrics computed successfully')
        return metric_dict
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise

def save_metrics(metrics:dict,output_path:str)->None:
    """save metrics to a json file"""
    try:
        os.makedirs(os.path.dirname(output_path),exist_ok=True)
        with open(output_path,'w') as file:
            json.dump(metrics,file,indent=4)
        logger.debug(f'Metrics saved to {output_path}')
    except Exception as e:
        logger.error('Error saving metrics to %s: %s', output_path, e)
        raise

def main():
    try:
        params=load_params('./params.yaml') 
        clf=load_model('./models/model.pkl')
        test_data=load_data('./data/featured/test_tfidf.csv')

        x_test=test_data.drop('label',axis=1).values
        y_test=test_data['label'].values

        metrics=evaluate_model(clf,x_test,y_test)
        with Live(save_dvc_exp=True) as live:
            # live.log_metric('accuracy', accuracy_score(y_test, y_test))
            # live.log_metric('precision', precision_score(y_test, y_test))
            # live.log_metric('recall', recall_score(y_test, y_test))
            for key,value in metrics.items():
                live.log_metric(key,value)  
            
            live.log_params(params)
        save_metrics(metrics,'./reports/metrics.json')
        logger.debug('Model evaluation completed successfully')
    except Exception as e:
        logger.error('Model evaluation failed: %s', e)  
        raise

if __name__=='__main__':
    main()
