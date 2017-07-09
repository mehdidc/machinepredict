from lenet import params
from machinepredict.interface import train

params['model'] = {
    'name': 'resnet',
    'params':{
        "size_blocks":[5,  5,  5],
        "nb_filters": [16, 32, 64],
        "block": "basic",
        "option": "B",
        "output_activation": "softmax",
    }
}

if __name__ == '__main__':
    train(params)
