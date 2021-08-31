from abc import abstractmethod
from torch.utils.data import Dataset


class DatasetBase(Dataset):
    """ Base Dataset Class """

    @abstractmethod
    def __len__(self):
        """ Sized """
        pass

    @abstractmethod
    def __getitem__(self, idx: int):
        """ Retrieve by indexing """
        pass

    @classmethod
    @abstractmethod
    def load(cls, data_dir_or_file: str, is_train: bool = True, **kwargs):
        """ Load from directory or files """
        pass

    @staticmethod
    def from_file():
        """ Read from files """
        pass

    @staticmethod
    def from_csv():
        """ Read from csv """
        pass

    @staticmethod
    def from_dataframe():
        """ Read from Pandas DataFrame """
        pass

    @staticmethod
    def from_json():
        """ Read from json """
        pass
