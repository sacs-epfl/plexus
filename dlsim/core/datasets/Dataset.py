import logging

from dlsim.util import utils
from dlsim.core.mappings.Mapping import Mapping


class Dataset:
    """
    This class defines the Dataset API.
    All datasets must follow this API.

    """

    def __init__(
        self,
        rank: int,
        machine_id: int,
        mapping: Mapping,
        train_dir="",
        test_dir="",
        sizes="",
        test_batch_size="",
    ):
        """
        Constructor which reads the data files, instantiates and partitions the dataset

        Parameters
        ----------
        rank : int
            Rank of the current process (to get the partition).
        machine_id : int
            Machine ID
        mapping : decentralizepy.mappings.Mapping
            Mapping to convert rank, machine_id -> uid for data partitioning
            It also provides the total number of global processes
        train_dir : str, optional
            Path to the training data files. Required to instantiate the training set
            The training set is partitioned according to the number of global processes and sizes
        test_dir : str. optional
            Path to the testing data files Required to instantiate the testing set
        sizes : list(int), optional
            A list of fractions specifying how much data to alot each process. Sum of fractions should be 1.0
            By default, each process gets an equal amount.
        test_batch_size : int, optional
            Batch size during testing. Default value is 64

        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.rank = rank
        self.machine_id = machine_id
        self.mapping = mapping
        self.uid = self.mapping.get_uid(rank, machine_id)
        # the number of global processes, needed to split-up the dataset
        self.n_procs = mapping.get_n_procs()
        self.train_dir = utils.conditional_value(train_dir, "", None)
        self.test_dir = utils.conditional_value(test_dir, "", None)
        self.sizes = utils.conditional_value(sizes, "", None)
        self.test_batch_size = utils.conditional_value(test_batch_size, "", 64)
        if self.sizes:
            if type(self.sizes) == str:
                self.sizes = eval(self.sizes)

        if train_dir:
            self.__training__ = True
        else:
            self.__training__ = False

        if test_dir:
            self.__testing__ = True
        else:
            self.__testing__ = False

    def get_trainset(self, batch_size=1, shuffle=False):
        """
        Function to get the training set

        Returns
        -------
        torch.utils.Dataset(decentralizepy.datasets.Data)

        Raises
        ------
        RuntimeError
            If the training set was not initialized

        """
        raise NotImplementedError

    def get_testset(self):
        """
        Function to get the test set

        Returns
        -------
        torch.utils.Dataset(decentralizepy.datasets.Data)

        Raises
        ------
        RuntimeError
            If the test set was not initialized

        """
        raise NotImplementedError
