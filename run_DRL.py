# common library
import sys
import argparse
import pandas as pd
import numpy as np
import time
from stable_baselines.common.vec_env import DummyVecEnv

# preprocessor
from preprocessing.preprocessors import *
# config
from config.config import *
# model
from model.models import *
import os

def model_dict_create():
    """Parse arguments and save them in model_dict"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', '--test_final_date', default=20200707, type=int,
                        help='Specify final time for testing')
    parser.add_argument('--small', action='store_true',
                        help='use a smaller dataset')
    parser.add_argument('--no_ind', action='store_true',
                        help='omit indicators')
    args = parser.parse_args()

    model_dict = {}
    model_dict.update({'T': args.test_final_date})
    if args.small:
        model_dict.update({'small': 1})
    else:
        model_dict.update({'small': None})
    if args.no_ind:
        model_dict.update({'no_ind': 1})
    else:
        model_dict.update({'no_ind': None})
        

    return model_dict

def run_model(argv) -> None:
    """Train the model."""

    model_dict = model_dict_create()
    print("model_dict: " + str(model_dict))
    # read and preprocess data
    preprocessed_path = "done_data.csv"
    no_ind = model_dict['no_ind']
    small = model_dict['small']
    T = model_dict['T']
    if small:
        preprocessed_path = "done_data_small.csv"
    else:
        preprocessed_path = "done_data.csv"
    
    print("using preprocessed path " + preprocessed_path)

    if os.path.exists(preprocessed_path):
        data = pd.read_csv(preprocessed_path, index_col=0)
    else:
        data = preprocess_data(small)
        data = add_turbulence(data)
        data.to_csv(preprocessed_path)

    print(data.head())
    print(data.size)

    # 2015/10/01 is the date that validation starts
    # 2016/01/01 is the date that real trading starts
    # unique_trade_date needs to start from 2015/10/01 for validation purpose
    unique_trade_date = data[(data.datadate > 20151001)&(data.datadate <= T)].datadate.unique()
    print(unique_trade_date)

    # rebalance_window is the number of months to retrain the model
    # validation_window is the number of months to validation the model and select for trading
    rebalance_window = 63
    validation_window = 63
    
    ## Ensemble Strategy
    run_ensemble_strategy(df=data, 
                          unique_trade_date= unique_trade_date,
                          rebalance_window = rebalance_window,
                          validation_window=validation_window,
                          small=small)

    #_logger.info(f"saving model version: {_version}")

if __name__ == "__main__":
    run_model(sys.argv[1:])
