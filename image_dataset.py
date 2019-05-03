from PIL import Image
from torch.utils import data

class DataSet(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, args, examples, labels, transform, is_train):
        'Initialization'
        self.labels = labels
        self.examples = examples
        self.transform = transform
        self.image_dir = args['image_dir']
        self.args = args
        self.n_classes = self.args['n_classes']
        self.is_train = is_train

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.examples)

    def __getitem__(self, idx):
        'Generates one sample of data'
        id = self.examples[idx]
        # Convert to RGB to avoid png.
        X = Image.open(self.image_dir + id).convert('RGB')
        X = self.transform(X)
        label = self.labels[id]
        return X,label

