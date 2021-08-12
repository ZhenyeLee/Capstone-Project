import time
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from data import loadZipFile, NYU_TrainAugmentDataset, NYU_TestDataset
from loss import depth_loss
from model import DepthModel


def main():
    # send the tensor to GPU if you have a GPU; otherwise, send the tensor to CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Create model
    model = DepthModel().to(device)
    
    # set batch size and the numer of epoches
    batch_size = 4
    num_epoch = 3
    
    # Adam optimizer with learning rate of 0.0001
    optimizer = torch.optim.Adam( model.parameters(), lr=0.0001 )

    # load train and test datasets
    data, train_names, test_names = loadZipFile('nyu_data.zip')
    # a list of tuples, each tuple corrresponding to one RGB tensor and depth ground truth tensor.
    train_set = NYU_TrainAugmentDataset(data, train_names)
    test_set = NYU_TestDataset(data, test_names)
    # slice the dataset to mini-batches, each one mini-batch can be sent to a loop.
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)
    
    train_avg_losses = []
    test_avg_losses = []
    
    # Start training...
    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        train_loss = 0.0
        test_loss = 0.0
        
        # Switch to train mode
        model.train()
        for i, sample_batched in enumerate(train_loader):
            # zero the gradients
            optimizer.zero_grad()
            # Prepare RGB sample and corresponding depth ground truth, and send only one batch to the device.
            train_image = torch.autograd.Variable(sample_batched['image'].to(device))
            train_depth = torch.autograd.Variable(sample_batched['depth'].to(device))
            # Predict depth
            train_output = model(train_image)
            # Compute the loss between the prediction and the ground truth
            batch_loss = depth_loss(train_output, train_depth)
            # auto-compute all gradients 
            batch_loss.backward()
            # update the parameters of the model using the computed gradients
            optimizer.step()
            
            # accumulate loss
            train_loss += batch_loss.item()
            
            # display information about running speed and batch loss
            if (i+1) % 10 == 0:
                print('Epoch [{}/{}][{}/{}], {:.2f} sec(s), Batch loss:{:.5f} (Avg:{:.5f})'
                  .format(epoch+1, num_epoch, i+1, train_loader.__len__(), (time.time()-epoch_start_time)*10/(i+1), batch_loss, train_loss/(i+1)))
        
        
        # switch to test mode 
        model.eval()
        # disable any gradient calculation
        with torch.no_grad():
            for i, sample_batched in enumerate(test_loader):
                # prepare test dataset
                test_image = torch.autograd.Variable(sample_batched['image'].to(device))
                test_depth = torch.autograd.Variable(sample_batched['depth'].to(device))
                # predict
                test_output = model(test_image)
                # loss
                batch_loss = depth_loss(test_output, test_depth)
                
                # accumulate loss
                test_loss += batch_loss.item()
                
                
                # display the depth prediction of last RGB image in test dataset
                if i == len(test_loader)-1:
                    # change the dimensions of the output tensor from (N x C x H x W) to (N x H x W x C)
                    output_depth = test_output.permute(0, 2, 3, 1)
                    # removes all dimensions with a length of one from tensor, it will return a tensor with the size of (H x W)
                    # transfer from tensor to numpy after removing gradients using torch.detach()
                    output_depth = torch.squeeze(output_depth[-1]).cpu().detach().numpy()
                    plt.figure(1)
                    plt.imshow( output_depth, cmap='plasma' )
                    plt.show()

            # record average batch losses for training and test sets at one epoch
            train_avg_losses.append(train_loss/train_loader.__len__())
            test_avg_losses.append(test_loss/test_loader.__len__())            

            # display information about running speed of one epoch and batch loss
            print('Epoch [{}/{}], {:.2f} sec(s), Avg Train loss:{:.5f}, Avg Test loss:{:.5f}'
                  .format(epoch+1, num_epoch, time.time()-epoch_start_time, train_loss/train_loader.__len__(), test_loss/test_loader.__len__()))
            
    # plot average batch losses for training and test sets
    plt.figure(2)
    plt.plot(train_avg_losses, 'o-', label='average train loss')
    plt.plot(test_avg_losses, 'o-', label='average test loss')
    plt.legend()
    plt.title('train/test losses')
    plt.savefig('losses.png')
    plt.show()

    # save model's parameters
    path = 'nyusmall_para.pt'
    torch.save(model.state_dict(), path)

if __name__ == '__main__':
    main()
