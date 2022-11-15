from torchvision import datasets, transforms, models
import torch
from torch import nn, optim
from collections import OrderedDict
import json
import numpy as np

architectures = {
  'alexnet': models.alexnet,
  'vgg13': models.vgg13,
  'vgg16': models.vgg16,
  'densenet121': models.densenet121,
}

input_units = {
  'alexnet': 9216,
  'vgg13': 25088,
  'vgg16': 25088,
  'densenet121': 1024,
}

def create_model(arch, hidden_layers):
    """Create the model from the pretrained model.

    Args:
      arch: name of pretrained architecture from pytorch
      hidden_layers: array of hidden layers

    Returns:
      model: the pretrainded model for the given architecture
    """
    
    model = architectures.get(arch, 'alexnet')(pretrained=True)

    for param in model.parameters():
      param.requires_grad = False

    # Input Layer
    classifier = OrderedDict([
      ('input', nn.Linear(input_units[arch], hidden_layers[0])),
      ('relu1', nn.ReLU()),
      ('dropout1', nn.Dropout(0.2)),
    ])
    # Hidden Layers
    idx = 1
    for h1, h2 in zip(hidden_layers[:-1], hidden_layers[1:]):
      classifier.update({f'hidden_layer{idx}': nn.Linear(h1, h2)})
      classifier.update({f'relu{idx+1}': nn.ReLU()})
      classifier.update({f'dropout{idx+1}': nn.Dropout(0.2)})
      idx += 1

    # Output Layer
    classifier.update({f'hidden_layer{idx}': nn.Linear(hidden_layers[-1], 102)})
    classifier.update({'output': nn.LogSoftmax(dim=1)})

    model.classifier = nn.Sequential(classifier)
    model.architecture = arch
    model.hidden_layers = hidden_layers

    return model

def get_loader(dir, transform, shuffle=False):
  # Load the datasets with ImageFolder
  dataset = datasets.ImageFolder(dir, transform=transform)
  # Using the image datasets and the transforms, define the dataloaders
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=shuffle)

  return dataloader, dataset.class_to_idx

def load_datasets(data_dir):
  """Load the datasets from a given directory path.

  Args:
    data_dir: path where data is stored

  Returns:
    dataloaders: pytorch dataloader for train, valid, test dataset 
  """
  train_dir = data_dir + '/train'
  valid_dir = data_dir + '/valid'
  test_dir = data_dir + '/test'

  # Transforms for the training, validation, and testing sets
  train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                        transforms.RandomRotation(45),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225]),])

  test_transform = transforms.Compose([transforms.Resize(224),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225]),])

  dataloaders = {}
  dataloaders['train'], class_to_idx = get_loader(train_dir, train_transform, shuffle=True)
  dataloaders['valid'], _ = get_loader(valid_dir, test_transform)
  dataloaders['test'], _ = get_loader(test_dir, test_transform)

  return dataloaders, class_to_idx

def select_device(is_gpu):
  """Select device on which trained is going to performed. 
  """

  return torch.device("cuda:0" if is_gpu and torch.cuda.is_available() else "cpu")

def train_network(dataloader, model, criterion, optimizer, device):
  """Train the model using the given loss function and optimizer

  Args:
    dataloader: pytorch dataloader for loading the data batches
    model: the NN model
    criterion: Loss Function (Ex- NLLLoss)
    optimizer: pytorch optimizer function (Ex- SGD, Adams)
    device: device on which validation performed (either CPU or GPU)

  Returns:
    loss: the total loss during training
    accuracy: accuracy of the model
  """

  running_loss = 0
  model.train()
  for images, labels in dataloader:
    images, labels = images.to(device), labels.to(device)

    optimizer.zero_grad()
    
    logps = model.forward(images)
    loss = criterion(logps, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()

  return model, running_loss

def train(dataloaders, model, epochs, device, lr=0.003):
  """Performing the validation for the NN model.

  Args:
    dataloader: pytorch dataloader for loading the data batches
    model: the NN model
    epochs: number of times training  is going to performed
    device: device on which validation performed (either CPU or GPU)
    lr: learning rate 

  Returns:
    accuracy: accuracy of the model after training
  """

  train_losses, valid_losses = [], []

  criterion = nn.NLLLoss()
  optimizer = optim.Adam(model.classifier.parameters(), lr)
  
  model.to(device)
  for epoch in range(epochs):
    model, running_loss =  train_network(dataloaders['train'], model, criterion, optimizer, device)
    valid_loss, accuracy = validate(dataloaders['valid'], model, criterion, device)
    
    train_losses.append(running_loss/len(dataloaders['train']))
    valid_losses.append(valid_loss/len(dataloaders['valid']))

    print("Epoch:               {}/{}.. ".format(epoch+1, epochs),
          "Training Loss:       {:.3f}.. ".format(running_loss/len(dataloaders['train'])),
          "Validation Loss:     {:.3f}.. ".format(valid_loss/len(dataloaders['valid'])),
          "Validation Accuracy: {:.3f}".format(accuracy/len(dataloaders['valid'])))

  # Perform final test
  test_loss, accuracy = validate(dataloaders['test'], model, criterion, device)
  
  print("\nFinal Test Results:-")
  print("Test Loss:            {:.3f}.. ".format(test_loss/len(dataloaders['test'])),
        "Test Accuracy:        {:.3f}".format(accuracy/len(dataloaders['test'])))

  return model, accuracy

def validate(dataloader, model, criterion, device):
  """Performing the validation for the trained model.

  Args:
    dataloader: pytorch dataloader for loading the data batches
    model: the NN model
    criterion: Loss Function
    device: device on which validation performed (either CPU or GPU)

  Returns:
    loss: the total loss during validation
    accuracy: accuracy of the model
  """
  
  accuracy = 0
  loss = 0
  with torch.no_grad():
    model.eval()
    for images, labels in dataloader:
      images, labels = images.to(device), labels.to(device)
      logps = model(images)
      loss += criterion(logps, labels).item()

      ps = torch.exp(logps)
      top_p, top_class = ps.topk(1, dim=1)
      equals = top_class == labels.view(*top_class.shape)
      accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

  return loss, accuracy

def save_checkpoint(filepath, model):
  """Save the model to given location

  Args:
    filepath: filepath where to save the model
    model: the pretrained model you want to save

  Returns:
    None
  """
  
  checkpoint = {
    'arch': model.architecture,
    'hidden_layers': model.hidden_layers,
    'classifier': model.classifier,
    'state_dict': model.state_dict(),
    'mapping': model.class_to_idx
  }

  torch.save(checkpoint, filepath)

def load_checkpoint(filepath):
  """Load a checkpoint from a given location and rebuilds the model

  Args: 
    filepath: path to the saved model file

  Returns:
    model: the saved model
  """

  checkpoint = torch.load(filepath, map_location='cpu')

  model = create_model(checkpoint['arch'], checkpoint['hidden_layers'])
  
  model.classifier = checkpoint['classifier']
  model.load_state_dict(checkpoint['state_dict'])
  model.class_to_idx = checkpoint['mapping']
  
  return model

def predict(image, model, topk=5, device='cpu'):
  """Predict the class (or classes) of an image using a trained deep learning model.

  Args:
    image: path to image file
    model: the NN model used for prediction
    topk: number of classes or predictions
    device: device on which prediction performed (either CPU or GPU)

  Returns:
    top_ps: top probabilities
    top_classes: top classes or predictions
  """

  img_tensor = torch.from_numpy(image).type(torch.FloatTensor).to(device)
  img_tensor = img_tensor.unsqueeze(dim=0)


  model.to(device)
  with torch.no_grad():
    ps = model.forward(img_tensor)
    probs = np.exp(ps)

    top_ps, top_classes = probs.topk(topk, dim=1)
    top_ps = top_ps.numpy().tolist()[0]
    top_classes = top_classes.numpy().tolist()[0]

    idx_to_class = {model.class_to_idx[k]: k for k in model.class_to_idx}
    top_classes = [idx_to_class[x] for x in top_classes]

    return top_ps, top_classes

