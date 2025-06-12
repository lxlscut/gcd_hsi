class Datasets:
    def __init__(self, x, y, lo_index, index, transform=None):
        self.x = x
        self.y = y
        self.lo_index = lo_index
        self.index = index
        self.transform = transform  # Callable transform applied to x

    def __getitem__(self, idx):
        x_item = self.x[idx].copy()
        y_item = self.y[idx]
        index_item = self.index[idx]

        if self.transform:
            x_item = self.transform(x_item)

        return x_item, y_item, index_item

    def __len__(self):
        return len(self.x)
