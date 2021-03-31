import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import batchify_data, run_epoch, train_model, Flatten
import utils_multiMNIST as U
path_to_data_dir = '../Datasets/'
use_mini_dataset = True

batch_size = 64
nb_classes = 10
nb_epoch = 30
num_classes = 10
img_rows, img_cols = 42, 28 # input image dimensions

class MLP(nn.Module):

    def __init__(self, input_dimension):
        super(MLP, self).__init__()
        self.flatten = Flatten()
        # TODO initialize model layers here
        # https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_module.html

        D_in = input_dimension
        H = 64  #Single hidden layer with 64 units as specified
        D_out = 20
        self.linear1 = nn.Linear(D_in, H) # use of Linear layers in PyTorch as specified
        self.linear2 = nn.Linear(H, D_out) # use of Linear layers in PyTorch as specified

    def forward(self, x):
        xf = self.flatten(x)

        # TODO use model layers to predict the two digits
        # https://discuss.pytorch.org/t/a-model-with-multiple-outputs/10440/23
        # x = F.relu(self.conv1(x))
        # return F.relu(self.conv2(x))
        # shal91 comment on indexing tensor
        # You can slice tensors the same way you slice multidimensional np.ndarrays. If you have matrix
        # m = np.array([[1, 2, 3], [4, 5, 6]]) =  [142536]
        # then m[a:b, c:d] slices your matrix as follows:
        # rows from a to b & columns from c to d; b, d exclusive
        # In our cases rows correspond to samples, columns correspond to features/dimensions.
        # E.g. m[:, :10] selects all samples and for those, keeps columns 0 to 9

        x1 = self.linear1(xf)
        x2 = self.linear2(x1)

        out_first_digit = x2[:,:10]
        out_second_digit = x2[:,10:]
        return out_first_digit, out_second_digit

def main():
    X_train, y_train, X_test, y_test = U.get_data(path_to_data_dir, use_mini_dataset)

    # print('X_train', X_train.shape)
    # print('y_train', y_train.shape)
    # print('X_test', X_test.shape)
    # print('y_test', y_train.shape)


    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = [y_train[0][dev_split_index:], y_train[1][dev_split_index:]]
    X_train = X_train[:dev_split_index]
    y_train = [y_train[0][:dev_split_index], y_train[1][:dev_split_index]]

    # print('X_train', len(X_train))
    # print('y_train', len(y_train))

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [[y_train[0][i] for i in permutation], [y_train[1][i] for i in permutation]]

    # Split dataset into batches
    # print('X_train', len(X_train))
    # print('y_train', len(y_train))
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    # Load model
    input_dimension = img_rows * img_cols
    model = MLP(input_dimension) # TODO add proper layers to MLP class above

    # Train
    train_model(train_batches, dev_batches, model)

    ## Evaluate the model on test data
    loss, acc = run_epoch(test_batches, model.eval(), None)
    print('Test loss1: {:.6f}  accuracy1: {:.6f}  loss2: {:.6f}   accuracy2: {:.6f}'.format(loss[0], acc[0], loss[1], acc[1]))

    # print(y_train[0], y_train[1])
    # print(X_train, y_train)

if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    main()
