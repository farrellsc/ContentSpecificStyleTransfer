class DataSet:
    def __init__(self):
        super(DataSet, self).__init__()
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, ind):
        raise NotImplementedError
