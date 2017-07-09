from keras.models import load_model

from machinedesign.interface import  train as _train
from machinedesign.interface import default_config as config
from machinedesign.transformers import transform_one

from .model_builders import builders

config.model_builders.update(builders)


def accuracy(y_true, y_pred):
    return (y_true.argmax(axis=1) == y_pred.argmax(axis=1))

config.metrics['accuracy'] = accuracy

def train(params):
    return _train(params, config=config)


def load(folder):
    model = load_model(os.path.join(folder, 'model.h5'))
    with open(os.path.join(folder, 'transformers.pkl'), 'rb') as fd:
        transformers = pickle.load(fd)
    model.transformers = transformers
    return model


def predict(model, X):
    X = transform_one(X, model.transformers)
    return model.predict(X)
