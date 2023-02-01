import json, time
from dataset import galaxy_images
import torch
import torchvision.models as models
import torch.nn as nn

try:
    import nvidia_smi
    NVIDIA_SMI = True
except:
    NVIDIA_SMI = False

# Parse configuration file
config_file = 'config_resnet.json'
with open(config_file) as json_file:
    hyperparameters = json.load(json_file)

timestr = time.strftime("%Y%m%d-%H%M%S")
hyperparameters['timestr'] = timestr
gpu = hyperparameters['gpu']
size_x = hyperparameters['size_x']
size_y = hyperparameters['size_y']

# check if the GPU is available
cuda = torch.cuda.is_available()
device = torch.device(f"cuda:{gpu}" if cuda else "cpu")
print('Checking the GPU availability...')
if cuda:
    print('GPU is available')
    print('Using GPU {}\n'.format(gpu))
else:
    print('GPU is not available')
    print('Using CPU\n')
    print(device)

# Load the pretrained Inception-ResNet v2 model
print('Loading the model...\n')
# print('Available models: {}'.format(models.__dict__.keys()))
net = models.resnet18(pretrained=False)

# Replace the final fully connected layer with a new one
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 2)

# Define the loss function and optimizer
loss_funct = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)


# Load the data
print('Loading the data...')
dataset = galaxy_images(hyperparameters['readir'], True, size_x, size_y)
dataset_test = galaxy_images(hyperparameters['readir'], False, size_x, size_y)
dataload = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
dataload_test = torch.utils.data.DataLoader(dataset_test, batch_size=32, shuffle=True)

# Train the network
for epoch in range(10):
    for i, (inputs, labels) in enumerate(dataload):
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_funct(outputs, labels)
        loss.backward()
        optimizer.step()

