from lenet import params
from machinepredict.interface import train

params['model'] = {
    'name': 'densenet',
    'params':{
        'depth': 30,
        'nb_dense_block': 3,
        'growth_rate': 12,
        'nb_filter': 16,
        'dropout_rate': 0.2,
        'weight_decay': 1e-4,
        'output_activation': 'softmax',
    }
}

if __name__ == '__main__':
    train(params)
