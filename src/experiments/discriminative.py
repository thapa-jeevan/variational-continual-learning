import os
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST
from torchvision.transforms import Compose

from src.models.coreset import RandomCoreset
from src.models.vcl_nn import DiscriminativeVCL
from src.util.datasets import NOTMNIST
from src.util.experiment_utils import run_point_estimate_initialisation, run_task
from src.util.transforms import Flatten, Scale, Permute

MNIST_FLATTENED_DIM = 28 * 28
LR = 0.001
INITIAL_POSTERIOR_VAR = 1e-3

device = torch.device("cuda:0")
print("Running on device", device)


def permuted_mnist():
    """
    Runs the 'Permuted MNIST' experiment from the VCL paper, in which each task
    is obtained by applying a fixed random permutation to the pixels of each image.
    """
    N_CLASSES = 10
    LAYER_WIDTH = 100
    N_HIDDEN_LAYERS = 2
    N_TASKS = 10
    MULTIHEADED = False
    CORESET_SIZE = 200
    EPOCHS = 100
    BATCH_SIZE = 256
    TRAIN_FULL_CORESET = True

    # flattening and permutation used for each task
    transforms = [Compose([Flatten(), Scale(), Permute(torch.randperm(MNIST_FLATTENED_DIM))]) for _ in range(N_TASKS)]

    # create model, single-headed in permuted MNIST experiment
    model = DiscriminativeVCL(
        in_size=MNIST_FLATTENED_DIM, out_size=N_CLASSES,
        layer_width=LAYER_WIDTH, n_hidden_layers=N_HIDDEN_LAYERS,
        n_heads=(N_TASKS if MULTIHEADED else 1),
        initial_posterior_var=INITIAL_POSTERIOR_VAR
    ).to(device)

    coreset = RandomCoreset(size=CORESET_SIZE)

    mnist_train = ConcatDataset(
        [MNIST(root="data", train=True, download=True, transform=t) for t in transforms]
    )
    task_size = len(mnist_train) // N_TASKS
    train_task_ids = torch.cat(
        [torch.full((task_size,), id) for id in range(N_TASKS)]
    )

    mnist_test = ConcatDataset(
        [MNIST(root="data", train=False, download=True, transform=t) for t in transforms]
    )
    task_size = len(mnist_test) // N_TASKS
    test_task_ids = torch.cat(
        [torch.full((task_size,), id) for id in range(N_TASKS)]
    )

    summary_logdir = os.path.join("logs", "disc_p_mnist", datetime.now().strftime('%b%d_%H-%M-%S'))
    writer = SummaryWriter(summary_logdir)

    run_point_estimate_initialisation(model=model, data=mnist_train,
                                      epochs=EPOCHS, batch_size=BATCH_SIZE,
                                      device=device, lr=LR,
                                      multiheaded=MULTIHEADED,
                                      task_ids=train_task_ids)

    # each task is classification of MNIST images with permuted pixels
    for task in range(N_TASKS):
        run_task(
            model=model, train_data=mnist_train, train_task_ids=train_task_ids,
            test_data=mnist_test, test_task_ids=test_task_ids, task_idx=task,
            coreset=coreset, epochs=EPOCHS, batch_size=BATCH_SIZE,
            device=device, lr=LR, save_as="disc_p_mnist",
            multiheaded=MULTIHEADED, train_full_coreset=TRAIN_FULL_CORESET,
            summary_writer=writer
        )

    writer.close()


def split_mnist():
    """
    Runs the 'Split MNIST' experiment from the VCL paper, in which each task is
    a binary classification task carried out on a subset of the MNIST dataset.
    """
    N_CLASSES = 2  # TODO does it make sense to do binary classification with out_size=2 ?
    LAYER_WIDTH = 256
    N_HIDDEN_LAYERS = 2
    N_TASKS = 5
    MULTIHEADED = True
    CORESET_SIZE = 40
    EPOCHS = 1
    BATCH_SIZE = 50000
    TRAIN_FULL_CORESET = True

    transform = Compose([Flatten(), Scale()])

    # download dataset
    mnist_train = MNIST(root="data", train=True, download=True, transform=transform)
    mnist_test = MNIST(root="data", train=False, download=True, transform=transform)

    model = DiscriminativeVCL(
        in_size=MNIST_FLATTENED_DIM, out_size=N_CLASSES,
        layer_width=LAYER_WIDTH, n_hidden_layers=N_HIDDEN_LAYERS,
        n_heads=(N_TASKS if MULTIHEADED else 1),
        initial_posterior_var=INITIAL_POSTERIOR_VAR
    ).to(device)

    coreset = RandomCoreset(size=CORESET_SIZE)

    label_to_task_mapping = {
        0: 0, 1: 0,
        2: 1, 3: 1,
        4: 2, 5: 2,
        6: 3, 7: 3,
        8: 4, 9: 4,
    }

    if isinstance(mnist_train[0][1], int):
        train_task_ids = torch.Tensor([label_to_task_mapping[y] for _, y in mnist_train])
        test_task_ids = torch.Tensor([label_to_task_mapping[y] for _, y in mnist_test])
    elif isinstance(mnist_train[0][1], torch.Tensor):
        train_task_ids = torch.Tensor([label_to_task_mapping[y.item()] for _, y in mnist_train])
        test_task_ids = torch.Tensor([label_to_task_mapping[y.item()] for _, y in mnist_test])

    summary_logdir = os.path.join("logs", "disc_s_mnist", datetime.now().strftime('%b%d_%H-%M-%S'))
    writer = SummaryWriter(summary_logdir)

    # each task is a binary classification task for a different pair of digits
    binarize_y = lambda y, task: (y == (2 * task + 1)).long()

    run_point_estimate_initialisation(model=model, data=mnist_train,
                                      epochs=EPOCHS, batch_size=BATCH_SIZE,
                                      device=device, multiheaded=MULTIHEADED,
                                      lr=LR, task_ids=train_task_ids,
                                      y_transform=binarize_y)

    for task_idx in range(N_TASKS):
        run_task(
            model=model, train_data=mnist_train, train_task_ids=train_task_ids,
            test_data=mnist_test, test_task_ids=test_task_ids, coreset=coreset,
            task_idx=task_idx, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
            save_as="disc_s_mnist", device=device, multiheaded=MULTIHEADED,
            y_transform=binarize_y, train_full_coreset=TRAIN_FULL_CORESET,
            summary_writer=writer
        )

    writer.close()


def split_not_mnist():
    """
    Runs the 'Split not MNIST' experiment from the VCL paper, in which each task
    is a binary classification task carried out on a subset of the not MNIST
    character recognition dataset.
    """
    N_CLASSES = 2  # TODO does it make sense to do binary classification with out_size=2 ?
    LAYER_WIDTH = 150
    N_HIDDEN_LAYERS = 4
    N_TASKS = 5
    MULTIHEADED = True
    CORESET_SIZE = 40
    EPOCHS = 120
    BATCH_SIZE = 400000
    TRAIN_FULL_CORESET = True

    transform = Compose([Flatten(), Scale()])

    not_mnist_train = NOTMNIST(train=True, overwrite=False, transform=transform)
    not_mnist_test = NOTMNIST(train=False, overwrite=False, transform=transform)

    model = DiscriminativeVCL(
        in_size=MNIST_FLATTENED_DIM, out_size=N_CLASSES,
        layer_width=LAYER_WIDTH, n_hidden_layers=N_HIDDEN_LAYERS,
        n_heads=(N_TASKS if MULTIHEADED else 1),
        initial_posterior_var=INITIAL_POSTERIOR_VAR
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    coreset = RandomCoreset(size=CORESET_SIZE)

    # The y classes are integers 0-9.
    label_to_task_mapping = {
        0: 0, 1: 1,
        2: 2, 3: 3,
        4: 4, 5: 0,
        6: 1, 7: 2,
        8: 3, 9: 4,
    }

    train_task_ids = torch.from_numpy(np.array([label_to_task_mapping[y] for _, y in not_mnist_train]))
    test_task_ids = torch.from_numpy(np.array([label_to_task_mapping[y] for _, y in not_mnist_test]))

    summary_logdir = os.path.join("logs", "disc_s_n_mnist", datetime.now().strftime('%b%d_%H-%M-%S'))
    writer = SummaryWriter(summary_logdir)

    # each task is a binary classification task for a different pair of digits
    # binarize_y(c, n) is 1 when c is is the nth digit - A for task 0, B for task 1
    binarize_y = lambda y, task: (y == task).long()

    run_point_estimate_initialisation(model=model, data=not_mnist_train,
                                      epochs=EPOCHS, batch_size=BATCH_SIZE,
                                      device=device, multiheaded=MULTIHEADED,
                                      task_ids=train_task_ids, lr=LR,
                                      y_transform=binarize_y)

    for task_idx in range(N_TASKS):
        run_task(
            model=model, train_data=not_mnist_train, train_task_ids=train_task_ids,
            test_data=not_mnist_test, test_task_ids=test_task_ids,
            coreset=coreset, task_idx=task_idx, epochs=EPOCHS, lr=LR,
            batch_size=BATCH_SIZE, save_as="disc_s_n_mnist", device=device,
            multiheaded=MULTIHEADED, y_transform=binarize_y,
            train_full_coreset=TRAIN_FULL_CORESET,
            summary_writer=writer
        )

    writer.close()
