
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as funcs

DEBUG=False

# Static functions
def calculate_error ( y_pred, y_true ):
    return np.sum(y_pred != y_true) / len(y_true)

def evaluate ( model, data_loader, device ):
    '''
    Calculate the classification error in (%) 
    '''
    y_true = np.array([])
    y_pred = np.array([])

    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            y_true = np.concatenate((y_true, labels.cpu()))
            y_pred = np.concatenate((y_pred, predicted.cpu()))
    
    return calculate_error(y_pred, y_true)

def run(model, epochs, train_loader, test_loader, criterion, optimizer, RESULTS_PATH, scheduler=None, MODEL_PATH=None):
    # Check if GPU is available and run there
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f' Using {device}')
    model.to(device)

    # Setup results file
    col_names = ['epoch', 'train_loss', 'train_err', 'test_err']
    results_df = pd.DataFrame(columns=col_names).set_index('epoch')
    if DEBUG:
        print('Epoch \tBatch \tNegative_Log_Likelihood_Train_Loss')

    for epoch in range(int(epochs)):
        # TODO: set the model training mode to true

        # Initialize loss and error variables        
        total_loss = []
        best_test_err = 1.0
        best_train_err = 1.0

        for i, data in enumerate(train_loader, 0): # Batch iteration
            # TODO Transfer all inputs and labels from your data loader to the proper device
        
            # TODO: As we saw in Assignment1, with each new batch, we need to reset the gradients
            # Set the gradients of all optmized tensors to 0

            # TODO: Generate predictions, calculate the loss, run backpropagation, and increment the optimizer
            loss = None

            # print avg loss for last 50 mini-batches
            total_loss.append ( loss.item() )
            if i % 50 == 49:
                if DEBUG:
                    print(f'{epoch +1} \t{i+1} \t{np.mean(total_loss)}' )
        
        # TODO: update the learning rate scheduler 
        
        # TODO: Evaluate.
        train_loss = None
        train_err = None
        test_err = None

        # Write out metrics to pandas dataframe
        results_df.loc[epoch] = [train_loss, train_err, test_err ]
        results_df.to_csv ( RESULTS_PATH )
        print(f'epoch: {epoch+1} train_err: {train_err} test_err: {test_err} train_loss: {train_loss}')

        # TODO: Save the checkpoint of the model with the best test performance
        # and the best train performance
        test_model_path = f'{MODEL_PATH}_test.pt'
        train_model_path = f'{MODEL_PATH}_train.pt'
    
    print('Finished Training')
    print(f'FLOPs: {model.flops}')

class Block(nn.Module):
    '''
    A 2-layer residual learning building block (Fig. 2)
    Params:
        - out_channel (int): the number of filters for all layers in this block ( {16,32,64} )
        - stride (int): stride size (in the paper, some blocks subsample by a stride of 2)
    
    Functions:
        - skip_connections (bool): indication whether the network has skip_connections or not
    '''
    def __init__(self, out_channels, stride=1):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = 3     # TODO: fill in Kernel size for 3x3 convs
        self.padding = 1         # TODO: Add 1-pixel of padding
        self.stride = stride        # Set stride to 1 unless otherwise specified
        
        # TODO: Specify the block's functions
        # Specify the block's functions
        self.conv1 = nn.Sequential(
                            nn.Conv2d(out_channels//self.stride, out_channels, kernel_size=self.kernel_size, 
                               stride=self.stride, padding=self.padding, bias=False),
                            nn.BatchNorm2d(self.out_channels),
                            nn.ReLU())
        
        self.conv2 = nn.Sequential(
                            nn.Conv2d(out_channels, out_channels, kernel_size=self.kernel_size, 
                               stride=1, padding=self.padding, bias=False),
                            nn.BatchNorm2d(self.out_channels))
        
        self.gap = nn.AvgPool2d(kernel_size=1, stride=2)
        
        # TODO: End the layer with a global average-pooling with a stride of 2

        # Initialise weights according to the method described in 
        # “Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification” 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)   
    
    # functions added to pass in skip_connections param; otherwise would use nn.Module.forward function
    def skip_connection(self, F, X):
        if F.shape[1] != X.shape[1]:
            projection = self.gap(X)
            padding = torch.zeros((projection.shape[0], F.shape[1] - projection.shape[1], projection.shape[2], projection.shape[3]))

            projection = torch.cat((projection, padding), 1)
        else:
            projection = X
        return F + projection



    def forward(self, X, skip_connections=False):
        out = self.conv1(X)


        out = self.conv2(out)


        if skip_connections:
            out = self.skip_connection(out, X)

        out = funcs.relu(out)
        
        return out

class Resnet(nn.Module):
    def __init__(self, N, skip_connections=True, stride=2):
        super().__init__()
        self.skip_connections = skip_connections # If false, this is just a plain conv net

        self.flops = 0 # TODO: fill me in when asked to compute this

        # Input
        self.conv = nn.Conv2d ( 3, 16, kernel_size=3, stride=1, padding=1, bias=False )
        self.bn = nn.BatchNorm2d ( 16, track_running_stats=True )
        self.relu = nn.ReLU()

        # TODO: stack1 - create a modulelist that contains N Blocks with filter_size 16
        self.stack1 = nn.ModuleList([Block(16) for i in range(N)])

        # TODO: stack2 - create a Block with filter size 32 with subsampling followed by N-1 blocks without subsampling
        self.stack2 = nn.ModuleList([Block(32, stride=stride)])
        self.stack2.extend([Block(32) for i in range(N-1)])

        # TODO: stack3 - create a Block with filter size 64 with subsampling followed by N-1 blocks without subsampling
        self.stack3 = nn.ModuleList([Block(64, stride=stride)])
        self.stack3.extend([Block(64) for i in range(N-1)])

        # TODO: output: set average pooling function, fully-connected function, and softmax function
        # Average pooling parameters unspecified in the paper (We will use AveragePool2d with a 8x8 kernel)
        # followed by a 10-way fully connected layer. NOTE: we add a bias term to the 10-way fully connected layer.
        # and a logarithmic softmax function
        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(64, 10)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        # init weights in the fully connected layer
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    
    def forward(self, X):
        with torch.profiler.profile(
            profile_memory=False,
            record_shapes=False,
            with_flops=True
        ) as prof:
            # Convolution 1
            out = self.conv(X)
            out = self.bn(out)
            out = self.relu(out)

            # First layer
            for block in self.stack1:
                out = block(out, skip_connections=self.skip_connections)

            # Second layer
            for block in self.stack2:
                out = block(out, skip_connections=self.skip_connections)

            # Third layer
            for block in self.stack3:
                out = block(out, skip_connections=self.skip_connections)

            # Finish the network with global average pooling
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)

            # Finish the network with a 10-way fully-connected layer
            out = self.fc(out)

            # Add softmax
            out = self.logsoftmax(out)
        
        return out

