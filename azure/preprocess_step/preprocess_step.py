import os
import yaml
import argparse
from src.data.preprocess import preprocess

parser = argparse.ArgumentParser()
parser.add_argument('--miladatadir', type=str, help="Mila dataset directory")
parser.add_argument('--fig1datadir', type=str, help="Figure 1 dataset directory")
parser.add_argument('--rsnadatadir', type=str, help="RSNA dataset directory")
parser.add_argument('--preprocesseddir', type=str, help="preprocessed output")
args = parser.parse_args()

cfg = yaml.full_load(open(os.getcwd() + "./config.yml", 'r'))  # Load config data
cfg['PATHS']['MILA_DATA'] = args.miladatadir
cfg['PATHS']['FIGURE1_DATA'] = args.fig1datadir
cfg['PATHS']['RSNA_DATA'] = args.rsnadatadir
cfg['PATHS']['PROCESSED_DATA'] = args.preprocesseddir
cfg['PATHS']['TRAIN_SET'] = cfg['PATHS']['PROCESSED_DATA'] + '/' + cfg['PATHS']['TRAIN_SET'].split('/')[-1]
cfg['PATHS']['VAL_SET'] = cfg['PATHS']['PROCESSED_DATA'] + '/' + cfg['PATHS']['VAL_SET'].split('/')[-1]
cfg['PATHS']['TEST_SET'] = cfg['PATHS']['PROCESSED_DATA'] + '/' + cfg['PATHS']['TEST_SET'].split('/')[-1]

print(cfg['PATHS']['PROCESSED_DATA'])
print(cfg['PATHS']['MILA_DATA'])
print(cfg['PATHS']['TRAIN_SET'])

preprocess(cfg)