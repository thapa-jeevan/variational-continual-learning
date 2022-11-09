from copy import deepcopy
from random import shuffle

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm

from src.util.operations import task_subset


class Coreset:
    """
    Base class for the the coreset.  This version of the class has no
    coreset but subclasses will replace the select method.
    """

    def __init__(self, size=0, lr=0.001):
        self.size = size
        self.coreset = None
        self.coreset_task_ids = None
        self.lr = lr

    def select(self, d: data.Dataset, task_id: int):
        """
        Given a torch dataset, will choose k datapoints.  Will then update
        the coreset with these datapoints.
        Returns: the subset that was not selected as a torch dataset.
        """

        return d

    def coreset_train(self, m, old_optimizer, tasks, epochs, device,
                      y_transform=None, multiheaded=True, batch_size=256):
        """
        Returns a new model, trained on the coreset.  The returned model will
        be a deep copy, except when coreset is empty (when it will be identical)

        tasks can be either a list, in which case the coreset will be trained
        on all tasks in the list, or an integer, in which case it will be
        trained on only that task.
        """

        if self.coreset is None:
            return m

        model = deepcopy(m)

        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        optimizer.load_state_dict(old_optimizer.state_dict())

        # if tasks is an integer, turn it into a singleton.
        if isinstance(tasks, int):
            tasks = [tasks]

        # create dict of train_loaders
        train_loaders = {
            task_idx: data.DataLoader(
                task_subset(self.coreset, self.coreset_task_ids, task_idx),
                batch_size
            )
            for task_idx in tasks
        }

        print('CORESET TRAIN')
        for _ in tqdm(range(epochs), 'Epochs: '):
            # Randomize order of training tasks
            shuffle(tasks)
            for task_idx in tasks:
                head = task_idx if multiheaded else 0

                for batch in train_loaders[task_idx]:
                    optimizer.zero_grad()
                    x, y_true = batch
                    x = x.to(device)
                    y_true = y_true.to(device)

                    if y_transform is not None:
                        y_true = y_transform(y_true, task_idx)

                    loss = model.vcl_loss(x, y_true, head, len(self.coreset))
                    loss.backward()
                    optimizer.step()

        return model

    def coreset_train_generative(self, m, old_optimizer, up_to_task, epochs, device,
                                 multiheaded=True, batch_size=256):
        """
        Returns a new model, trained on the coreset.  The returned model will
        be a deep copy, except when coreset is empty (when it will be identical)
        """

        if self.coreset is None:
            return m

        model = deepcopy(m)

        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        optimizer.load_state_dict(old_optimizer.state_dict())

        task_subsets = [task_subset(self.coreset, self.coreset_task_ids, task_idx) for task_idx in
                        range(up_to_task + 1)]
        train_loaders = [data.DataLoader(task_data, batch_size) for task_data in task_subsets]

        print('CORESET TRAIN')
        for _ in tqdm(range(epochs), 'Epochs: '):
            for task_idx in torch.randperm(up_to_task + 1):
                head = task_idx if multiheaded else 0

                for batch in train_loaders[task_idx]:
                    optimizer.zero_grad()
                    x = batch[0].to(device)

                    loss = model.vae_loss(x, head, len(self.coreset))
                    loss.backward()
                    optimizer.step()

        return model


class RandomCoreset(Coreset):
    def __init__(self, size):
        super().__init__(size)

    def select(self, d: data.Dataset, task_id: int):

        new_cs_data, non_cs = data.random_split(d, [self.size, max(0, len(d) - self.size)])

        # Need to split the x from the y values to also include the task values.
        # I don't like this way of doing it, but I couldn't find something better.
        new_cs_x = torch.tensor(np.array([x for x, _ in new_cs_data]))
        new_cs_y = torch.tensor(np.array([y for _, y in new_cs_data]))

        new_cs = data.TensorDataset(new_cs_x, new_cs_y)
        new_task_ids = torch.full((len(new_cs_data),), task_id)

        if self.coreset is None:
            self.coreset = new_cs
            self.coreset_task_ids = new_task_ids
        else:
            self.coreset = data.ConcatDataset((self.coreset, new_cs))
            self.coreset_task_ids = torch.cat((self.coreset_task_ids, new_task_ids))

        return non_cs
