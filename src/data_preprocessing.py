import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
import string

nltk.download('stopwords')
nltk.download('punkt')

log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

#logging configuration
logger=logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,'data_preprocessing.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    """
    Transforms the input text by converting it to lowercase, tokenizing, removing stopwords and punctuation, and stemming.
    """
    ps=PorterStemmer()
    #convert to lowercase
    text=text.lower()
    #tokenize the text
    text=nltk.word_tokenize(text)
    #remove all non-alphanumeric characters
    text=[word for word in text if word.isalnum()]
    #remove stopwords and punctuation
    text=[word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    #stemming
    text=[ps.stem(word) for word in text]
    return " ".join(text)

def preprocess_data(df, text_column='text',target_column='target'):
    """
    Preprocesses the input dataframe by transforming the text column, removing the duplicates and encoding the target column.
    """
    try:
        logger.debug('Starting data preprocessing')
        #encode the target column
        encoder=LabelEncoder()
        df[target_column]=encoder.fit_transform(df[target_column])
        logger.debug('Target column encoded successfully')

        #remove the duplicate rows
        df.drop_duplicates(inplace=True)
        logger.debug('Duplicate rows removed successfully')

        #transform the text column
        df.loc[:,text_column]=df[text_column].apply(transform_text)
        logger.debug('Text column transformed successfully')
        return df
    except KeyError as e:
        logger.error(f'Missing column in the dataframe: {e}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error during preprocessing: {e}')
        raise

def main(text_column='text',target_column='target'):
    """
    Main function to load raw data, preprocess it, and save the processed data.
    """
    try:
        #fetch data from data/raw
        train_data=pd.read_csv('data/raw/train.csv')
        test_data=pd.read_csv('data/raw/test.csv')
        logger.debug('Raw data loaded successfully')

        #preprocess the data
        train_data=preprocess_data(train_data)
        test_data=preprocess_data(test_data)
        logger.debug('Data preprocessing completed successfully')

        # save the data to data/processed
        data_path=os.path.join('./data','processed')
        os.makedirs(data_path,exist_ok=True)
        train_data.to_csv(os.path.join(data_path,'train.csv'),index=False)
        test_data.to_csv(os.path.join(data_path,'test.csv'),index=False)
        logger.debug(f'Processed data saved successfully at {data_path}')
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__=='__main__':
    main()
