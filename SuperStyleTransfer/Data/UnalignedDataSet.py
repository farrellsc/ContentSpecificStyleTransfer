import random
from torch.utils.data import DataLoader


class UnalignedDataset(DataLoader):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, content_dataset, style_dataset, batch_size=1):
        super(UnalignedDataset, self).__init__(None, batch_size=batch_size)
        self.content_dataset = content_dataset
        self.style_dataset = style_dataset
        self.A_size = len(content_dataset)
        self.B_size = len(style_dataset)

    def __getitem__(self, index):
        A = self.content_dataset[index % self.A_size][0]
        index_B = random.randint(0, self.B_size - 1)
        B = self.style_dataset[index_B][0]
        return A, B

    def __len__(self):
        return self.A_size

    def name(self):
        return 'UnalignedDataset'
