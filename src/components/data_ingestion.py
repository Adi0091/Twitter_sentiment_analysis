import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path : str=os.path.join('artifacts', 'train.csv')
    test_data_path : str=os.path.join('artifacts', 'test.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")

        try:
            df = pd.read_csv(r'C:\Users\Admin\Documents\AIML\Machine Learning\twitter_analysis\Twitter_sentiment_analysis\notebook\data\filter_twitter_data.csv')
            logging.info("Read dataset.")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("splited data")

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("ingestion of data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        except Exception as e:
            raise CustomException(e,sys)
        

