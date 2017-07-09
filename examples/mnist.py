from machinepredict.interface import train

train_file = "mnist/train.npz"
test_file = "mnist/test.npz"

params = {
    'input_col': 'X',
    'output_col': 'y',
    'model': {
        'name': 'lenet',
        'params':{
            "nb_filters": [32, 64, 128],
            "dropout": 0.5,
            "fc_dropout": 0.5,
            "batch_norm": False,
            "fc": [512, 256, 128],
            "size_filters": 3,
            "activation": "prelu"
         }
    },
    'data': {
        'train': {
            'pipeline':[
                {"name": "load_numpy", "params": {"filename": train_file, "start": 0, "nb": 55000, "cols": ["X", "y"]}},
                {"name": "divide_by", "params": {"value": 255}},
                {"name": "onehot", "params": {"nb_classes": 10}}
            ]
        },
        'valid': {
            'pipeline':[
                {"name": "load_numpy", "params": {"filename": train_file, "start": 55000, "nb": 5000, "cols": ["X", "y"]}},
                {"name": "divide_by", "params": {"value": 255}},
                {"name": "onehot", "params": {"nb_classes": 10}}
            ]
        },
        'test': {
            'pipeline':[
                {"name": "load_numpy", "params": {"filename": test_file, "cols": ["X", "y"]}},
                {"name": "divide_by", "params": {"value": 255}},
                {"name": "onehot", "params": {"nb_classes": 10}}
            ]
        },
        'transformers':[
        ]
    },
    'report':{
        'outdir': 'out',
        'checkpoint': {
            'loss': 'valid_accuracy',
            'save_best_only': True
        },
        'metrics': ['accuracy'],
        'callbacks': [],
    },
    'optim':{
        'algo': {
            'name': 'adam',
            'params': {'lr': 1e-3}
        },
        'lr_schedule':{
            'name': 'constant',
            'params': {}
        },
        'early_stopping':{
            'name': 'none',
            'params': {
            }
        },
        'max_nb_epochs': 100,
        'batch_size': 128,
        'pred_batch_size': 128,
        "loss": "categorical_crossentropy",
        'budget_secs': 86400,
        'seed': 42
    },
}

train(params)
