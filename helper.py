import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
from torch.autograd import Variable


def test_network(net, trainloader):

    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    print('Testing the network...')

    losses = 0

    net.train()

    num_iterations = 10

    for i in range(num_iterations):
        dataiter = iter(trainloader)
        images, labels = dataiter.next()

        # Create Variables for the inputs and targets
        inputs = Variable(images)
        targets = Variable(labels)

        # Clear the gradients from all Variables
        optimizer.zero_grad()

        # Forward pass, then backward pass, then update weights
        output = net(inputs)
        # print(output.shape)
        # print(targets.shape)
        loss = criterion(output, targets)        
        loss.backward()
        optimizer.step()
        losses += loss
    print(losses/num_iterations)
    print('Test finished: ')

    return True, net


def test_network_output_size(net, tensor):

    print('Testing the network...')
    net.train()
    # Create Variables for the inputs and targets
    inputs = Variable(tensor)
    output = net(inputs)
    print('Test finished!')

    return output.shape


def test_network_img2img(net, trainloader):

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    print('Testing the network...')

    for i in range(100):
        dataiter = iter(trainloader)
        images, labels = dataiter.next()

        # Create Variables for the inputs and targets
        inputs = Variable(images)
        targets = Variable(images)

        # Clear the gradients from all Variables
        optimizer.zero_grad()

        # Forward pass, then backward pass, then update weights
        output = net.forward(inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
    print('Test finished: ')

    return True, net


def test_super_network(net, trainloader):

    criterion1 = nn.MSELoss()
    criterion2 = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    print('Testing the network...')

    for i in range(1):
        dataiter = iter(trainloader)
        images, labels = dataiter.next()

        # Create Variables for the inputs and targets
        inputs = Variable(images)
        targets = Variable(labels)

        # Clear the gradients from all Variables
        optimizer.zero_grad()

        # Forward pass, then backward pass, then update weights
        output = net.forward(inputs)
        print('Output: ', output.shape)
        output1 = output[:, :3, :, :]
        print(output1.shape)
        output2 = output[:, 3:, 0, 0]
        print(output2.shape)
        l1 = criterion1(output1, inputs)
        l2 = criterion2(output2, targets)
        loss = l1 + l2
        loss.backward()
        optimizer.step()
        print(output.shape)
    print('Test finished: ')

    return True, net


def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    plt.show()

    return ax


def imshow2(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    plt.show()

    return ax


def view_recon(img, recon):
    ''' Function for displaying an image (as a PyTorch Tensor) and its
        reconstruction also a PyTorch Tensor
    '''

    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
    axes[0].imshow(img.numpy().squeeze())
    axes[1].imshow(recon.data.numpy().squeeze())
    for ax in axes:
        ax.axis('off')
        ax.set_adjustable('box-forced')


def view_classify(img, ps, version="MNIST"):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    if version == "MNIST":
        ax2.set_yticklabels(np.arange(10))
    elif version == "Fashion":
        ax2.set_yticklabels(['T-shirt/top',
                            'Trouser',
                            'Pullover',
                            'Dress',
                            'Coat',
                            'Sandal',
                            'Shirt',
                            'Sneaker',
                            'Bag',
                            'Ankle Boot'], size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()
    plt.show()


def save_image_batch(image, epoch, path):
    """Imshow for Tensor."""
    r, c = 5, 5
    image = image.detach().numpy()

    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])

    fig, axs = plt.subplots(r, c)
    cnt = 0
    # save 25 images
    for i in range(r):
        for j in range(c):
            img = image[cnt].transpose((1, 2, 0))
            img = std * img + mean
            img = np.clip(img, 0, 1)
            axs[i, j].imshow(img)
            # axs[i, j].axis('off')
            cnt += 1
    fig.savefig(path+'/{}.png'.format(epoch))
    plt.close()
