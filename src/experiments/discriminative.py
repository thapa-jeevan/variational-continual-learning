import torch
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import Compose
from torch.utils.data import ConcatDataset
from models.vcl_nn import DiscriminativeVCL
from models.coreset import RandomCoreset
from util.experiment_utils import run_task
from util.transforms import Flatten, Permute
from util.outputs import write_as_json, save_model
from util.datasets import NOTMNIST


# input and output dimensions of an FCFF MNIST classifier
MNIST_FLATTENED_DIM = 28 * 28
MNIST_N_CLASSES = 10
EPOCHS = 100
BATCH_SIZE = 256
LR = 0.001
# settings specific to permuted MNIST experiment
NUM_TASKS_PERM = 10
# settings specific to split MNIST experiment
LABEL_PAIRS_SPLIT = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

CORESET_SIZE = 100

def permuted_mnist():
    """
    Runs the 'Permuted MNIST' experiment from the VCL paper, in which each task
    is obtained by applying a fixed random permutation to the pixels of each image.
    """
    # flattening and permutation used for each task
    transforms = [Compose([Flatten(), Permute(torch.randperm(MNIST_FLATTENED_DIM))]) for _ in range(NUM_TASKS_PERM)]

    # create model, single-headed in permuted MNIST experiment
    model = DiscriminativeVCL(in_size=MNIST_FLATTENED_DIM, out_size=MNIST_N_CLASSES, layer_width=100, n_hidden_layers=2, n_tasks=1)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    coreset = RandomCoreset(size=CORESET_SIZE)

    mnist_train = ConcatDataset(
        [MNIST(root='../data/', train=True, download=True, transform=t) for t in transforms]
    )
    task_size = len(mnist_train) // NUM_TASKS_PERM
    train_task_ids = torch.cat(
        [torch.full((task_size,), id) for id in range(NUM_TASKS_PERM)]
    )

    mnist_test = ConcatDataset(
        [MNIST(root='../data/', train=False, download=True, transform=t) for t in transforms]
    )
    task_size = len(mnist_test) // NUM_TASKS_PERM
    test_task_ids = torch.cat(
        [torch.full((task_size,), id) for id in range(NUM_TASKS_PERM)]
    )

    # each task is classification of MNIST images with permuted pixels
    for task in range(NUM_TASKS_PERM):
        run_task(
            model = model, train_data = mnist_train,
            train_task_ids = train_task_ids, test_data = mnist_test,
            test_task_ids = test_task_ids, task_idx = task, coreset= coreset,
            optimizer = optimizer, epochs = EPOCHS, batch_size = BATCH_SIZE,
            save_as = "disc_p_mnist", multiheaded=False
        )

def split_mnist():
    """
        Runs the 'Split MNIST' experiment from the VCL paper, in which each task
        is a binary classification task carried out on a subset of the MNIST dataset.
    """
    # download dataset
    mnist_train = MNIST(root='../data/', train=True, download=True, transform=Flatten())
    mnist_test = MNIST(root='../data/', train=False, download=True, transform=Flatten())

    # create model
    # fixme needs to be multi-headed
    # todo does it make sense to do binary classification with out_size=2 ?
    model = DiscriminativeVCL(in_size=MNIST_FLATTENED_DIM, out_size=2, layer_width=100, n_hidden_layers=2, n_tasks=5)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    coreset = RandomCoreset(size=CORESET_SIZE)

    label_to_task_mapping = {
        0: 0,  1: 0,
        2: 1,  3: 1,
        4: 2,  5: 2,
        6: 3,  7: 3,
        8: 4,  9: 4,
    }

    train_task_ids = torch.Tensor([label_to_task_mapping[y.item()] for _, y in mnist_train])
    test_task_ids  = torch.Tensor([label_to_task_mapping[y.item()] for _, y in mnist_test])

    # each task is a binary classification task for a different pair of digits
    for task_idx in range(5):
        binarize_y = lambda y: y == (2*task_idx + 1)
        run_task(
            model = model, train_data = mnist_train, train_task_ids = train_task_ids,
            test_data = mnist_test, test_task_ids = test_task_ids, optimizer = optimizer,
            coreset=coreset, task_idx = task_idx, epochs = EPOCHS,
            batch_size = BATCH_SIZE, save_as = "disc_s_mnist",
            y_transform = binarize_y
        )


def split_not_mnist():
    """
        Runs the 'Split not MNIST' experiment from the VCL paper, in which each task
        is a binary classification task carried out on a subset of the not MNIST
        character recognition dataset.
    """
    not_mnist_train = NOTMNIST(train=True, overwrite=False, transform=Flatten(), limit_size=50000)
    not_mnist_test = NOTMNIST(train=False, overwrite=False, transform=Flatten())

    model = DiscriminativeVCL(in_size=MNIST_FLATTENED_DIM, out_size=2, layer_width=100, n_hidden_layers=2)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # todo: are the y classes integers?  Or characters?
    label_to_task_mapping = {
        0: 0,  1: 0,
        2: 1,  3: 1,
        4: 2,  5: 2,
        6: 3,  7: 3,
        8: 4,  9: 4,
    }

    train_task_ids = torch.Tensor([label_to_task_mapping(y) for _, y in mnist_train])
    test_task_ids  = torch.Tensor([label_to_task_mapping(y) for _, y in mnist_test])

    # each task is a binary classification task for a different pair of characters
    for task_idx in range(5):
        binarize_y = lambda y: y == (2*task_idx + 1)
        run_task(
            model = model, train_data = not_mnist_train, train_task_ids = train_task_ids,
            test_data = not_mnist_test, test_task_ids = test_task_ids,
            optimizer = optimizer, coreset=coreset, task_idx = task_idx, epochs = EPOCHS,
            batch_size = BATCH_SIZE, save_as = "disc_s_n_mnist",
            y_transform = binarize_y
        )
