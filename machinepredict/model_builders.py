from .models.lenet import lenet
from .models.densenet import densenet
from .models.resnet import resnet

builders = {
    'lenet': lenet,
    'densenet': densenet,
    'resnet': resnet,
}
