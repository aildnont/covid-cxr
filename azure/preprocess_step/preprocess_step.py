import os
import yaml
import argparse
from src.data.preprocess import preprocess

parser = argparse.ArgumentParser()
parser.add_argument('--coviddatadir', type=str, help="COVID data directory")
parser.add_argument('--rsnadatadir', type=str, help="RSNA data directory")
parser.add_argument('--preprocesseddir', type=str, help="preprocessed output")
args = parser.parse_args()

cfg = yaml.full_load(open(os.getcwd() + "./config.yml", 'r'))  # Load config data
cfg['PATHS']['RAW_COVID_DATA'] = args.coviddatadir
cfg['PATHS']['RAW_OTHER_DATA'] = args.rsnadatadir
cfg['PATHS']['PROCESSED_DATA'] = args.preprocesseddir
cfg['PATHS']['TRAIN_SET'] = cfg['PATHS']['PROCESSED_DATA'] + '/' + cfg['PATHS']['TRAIN_SET'].split('/')[-1]
cfg['PATHS']['VAL_SET'] = cfg['PATHS']['PROCESSED_DATA'] + '/' + cfg['PATHS']['VAL_SET'].split('/')[-1]
cfg['PATHS']['TEST_SET'] = cfg['PATHS']['PROCESSED_DATA'] + '/' + cfg['PATHS']['TEST_SET'].split('/')[-1]

preprocess(cfg)