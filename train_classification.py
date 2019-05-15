import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data import load_CIFAR_dataset
from network import CAE, CAESKC, CAEshallow, SUperNet, weights_init
import helper


ROOT_PROJECT = '.'
MODELS_PATH = os.path.join(ROOT_PROJECT, 'models')
IMAGE_PATH = os.path.join(ROOT_PROJECT, 'images')


def _devices(device):
    train_on_gpu = (device == 'cuda')
    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')
    return train_on_gpu


def set_device(model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return model.to(device)


def train(X, l_r=0.001, n_epochs=20, save_every_step=200):
    torch.manual_seed(0)

    # Track traiining losses
    super_losses, crossentropy_losses, mse_losses = [], [], []
    val_losses = []
    # Load model
    model = SUperNet()
    # Initialize weights
    model.apply(weights_init)
    # model.load_state_dict(torch.load(MODELS_PATH + '/CAEwb.pth'))
    print(model)
    # Set loss to MSE and optimizer to Adam
    criterion1 = nn.MSELoss()
    criterion2 = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=l_r)
    # optimizer = optim.SGD(model.parameters(), lr=l_r, momentum=0.9)
    # Set model to device
    model = set_device(model)

    if torch.cuda.is_available():
        print('CUDA is available!  Training on GPU ...')
    else:
        print('CUDA is not available.  Training on CPU ...')

    for e in range(n_epochs):
        super_loss, crossentropy_loss, mse_loss = 0, 0, 0
        iter_loss = 0
        train_accuracy = 0
        train_accuracy_all_batches = 0
        model.train()

        for ii, data in enumerate(X, 0):
            images, label = data
            images, label = set_device(images), set_device(label)
            # reset gradients
            optimizer.zero_grad()
            # Forward pass, compute loss, do backprop, update weights per min-batch:
            images_hat, train_logits = model(images)
            l1 = criterion1(images_hat, images)
            l2 = criterion2(train_logits, label)
            loss = l1 + l2
            loss.backward()
            optimizer.step()

            # increment losses
            super_loss += loss.item()
            mse_loss += l1.item()
            crossentropy_loss = l2.item()

            # Make trainset evaluation
            _, label_hat = torch.max(train_logits, 1)
            train_accuracy_all_batches += (label_hat == label).sum().item()


            if ii % save_every_step == 0:
                # iter_loss = train_loss
                # print('#Iteration/Loss : {}/{:.3f}:'.format(ii, iter_loss/(len(images)*save_every_step)))
                helper.save_image_batch(images_hat, str(e+1)+'_'+str(ii), IMAGE_PATH)   
        # Losses
        super_loss = super_loss/len(X)
        mse_loss = mse_loss/len(X)
        crossentropy_loss = crossentropy_loss/len(X)

        super_losses.append(super_loss)
        mse_losses.append(mse_loss)
        crossentropy_losses.append(crossentropy_loss)

        # accuracies
        train_accuracy = train_accuracy_all_batches/len(X)

        print('Epochs: {}/{}   '.format(e+1, n_epochs),
              'Train loss: {:.5f}   '.format(super_loss),
              'MSE loss: {:.5f}   '.format(mse_loss),
              'CLass loss: {:.5f}   '.format(crossentropy_loss),
              'Train accuracy: {:.2f}'.format(100*train_accuracy)+'%   ',

              )

    print('Training Finished.')
    print('Saving model: ' + MODELS_PATH + '/SUperNet_wb_.pth')
    torch.save(model.state_dict(), MODELS_PATH + '/SUperNet_wb.pth')

    return super_losses, mse_losses, crossentropy_losses

if __name__ == "__main__":
    trainloader, _, _ = load_CIFAR_dataset(batch_size=32, dev_size=0.2)
    trainloss = train(trainloader)

    plt.plot(trainloss, label='Training Losses')
    plt.title("Train Losses")
    plt.xlabel("#Iterations")
    plt.show()


    # model1 = CAE()
    # model3 = CAESKC()
    # model1.apply(weights_init)
    # # model3.apply(weights_init)

    # print(model1)

    # _, model1 = helper.test_network(model1, trainloader)
    # data_iter = iter(trainloader)
    # images, labels = next(data_iter)
    # output1 = model1.forward(images)
    # helper.imshow2(images[0], normalize=True)
    # helper.imshow(output1[0].detach().numpy(), normalize=True)
    # helper.imshow(output3[0].detach().numpy(), normalize=True)


