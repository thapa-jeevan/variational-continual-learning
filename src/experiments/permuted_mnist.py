import os
from datetime import datetime

import torch
from torch.utils.data import ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST
from torchvision.transforms import Compose

from src.models.coreset import RandomCoreset
from src.models.vcl_nn import DiscriminativeVCL
from src.util.experiment_utils import run_point_estimate_initialisation, run_task
from src.util.transforms import Flatten, Scale, Permute

MNIST_FLATTENED_DIM = 28 * 28
LR = 0.001
INITIAL_POSTERIOR_VAR = 1e-3

device = torch.device("cuda:0")
print("Running on device", device)

N_CLASSES = 10
LAYER_WIDTH = 100
N_HIDDEN_LAYERS = 2
N_TASKS = 10
MULTIHEADED = False
CORESET_SIZE = 200
EPOCHS = 100
BATCH_SIZE = 1024
TRAIN_FULL_CORESET = True


def permuted_mnist():
    """
    Runs the 'Permuted MNIST' experiment from the VCL paper, in which each task
    is obtained by applying a fixed random permutation to the pixels of each image.
    """

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
            model=model,
            task_idx=task,
            train_data=mnist_train, train_task_ids=train_task_ids,
            test_data=mnist_test, test_task_ids=test_task_ids,
            coreset=coreset,
            epochs=EPOCHS, batch_size=BATCH_SIZE, device=device, lr=LR,
            save_as="disc_p_mnist",
            multiheaded=MULTIHEADED, train_full_coreset=TRAIN_FULL_CORESET,
            summary_writer=writer
        )

    writer.close()
