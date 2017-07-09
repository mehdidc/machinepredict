from .models.lenet import lenet
from .models.densenet import densenet

builders = {
    'lenet': lenet,
    'densenet': densenet
}
