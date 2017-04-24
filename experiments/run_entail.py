import os
import sys
sys.path.insert(0,os.getcwd())
from models import entail_prob_model
from models import entail_lstm_model


phase = sys.argv[1]
dirname, filename = os.path.split(os.path.abspath(__file__))
DIR = "/".join(dirname.split("/")[:-1])
params = {
    'run_dir': DIR,
    'exp_name_lstm': 'isa_0',
    'exp_name_full': 'cat_0',
    'data_dir': 'snli_100',

    'train_prob_data': 'snli_predict_dev_99.txt',
    'test_prob_data': 'snli_predict_dev_99.txt',
    'dev_prob_data': 'snli_predict_dev_99.txt',

    'train_entail_data': 'snli_1.0_dev.txt',
    'test_entail_data': 'snli_1.0_dev.txt',
    'dev_entail_data': 'snli_1.0_dev.txt',

    'vector_file': 'glove.ALL.txt.gz',

    'method': 'train',  # 'train' or 'test'

    'batch_size': 10,
    'hidden_dim': 100,  # hidden dim of LSTM
    'dropout_lstm': 0.5,  # 1 = no dropout, 0.5 = dropout

    'dropout_cpr': 0.5,
    'output_dim_cpr': 300,

    'num_epochs': 1,

    'phase': phase, # intermed or classifier
}

if params['phase'] == 'lstm':

    # Train LSTM entailment model from SNLI data
    entail_lstm_model.run(**params)

elif params['phase'] == 'lstm_plus_feat':

    # Train entailment prediction model by appending predicted probability features from file to output of previously trained LSTM model
    entail_prob_model.run(**params)

