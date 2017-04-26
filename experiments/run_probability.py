import sys, os
sys.path.insert(0,os.getcwd())
from models import probability_model

dirname, filename = os.path.split(os.path.abspath(__file__))
DIR = "/".join(dirname.split("/")[:-1])

params = {
    'run_dir': DIR,
    'exp_name': 'prob_9',
    'data_dir': 'den_prob_100',
    'train_data': 'cpr_train.txt.gz',
    'test_data': 'cpr_test.txt.gz',
    'dev_data': 'cpr_dev.txt.gz',
    'vector_file': 'glove.ALL.txt.gz',

    'method': 'test', # 'train' or 'test'

    'batch_size': 10,
    'hidden_dim': 512,  # hidden dim of LSTM
    'output_dim': 512,
    'dropout': 0.5,  # 1 = no dropout, 0.5 = dropout
    'num_epochs': 10,

    'lambda_px': 1,
    'lambda_cpr': 1,

    'learning_rate': 0.001,
}

tmp_dir = DIR+'/tmp/'
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

probability_model.run(**params)
