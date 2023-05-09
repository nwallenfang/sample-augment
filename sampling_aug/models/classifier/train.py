import torch
from torch.utils.data import Dataset
from tqdm import tqdm  # For nice progress bar!
from PIL import Image
import matplotlib.pyplot as plt
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.utils.data import DataLoader  # Gives easier dataset managment




def train(dataset: Dataset, model, num_epochs = 23,    batch_size = 64
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Train Network
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    train_batch_losses = []

    val_losses = []
    val_batch_losses = []

    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (image, label) in enumerate(tqdm(train_loader)):
            # Get data to cuda if possible
            image = image.to(device=device)
            label = label.to(device=device)

            # forward
            preds = model(image) # Pass batch
            train_loss = criterion(preds, label) # Calculate the loss

            # backward
            optimizer.zero_grad() # 
            train_loss.backward() # Calculate the gradients

            # gradient descent or adam step
            optimizer.step() # Uptade the weights
            
            # store loss
            train_batch_losses.append(train_loss.item())
            
        train_losses.append(sum(train_batch_losses)/len(train_batch_losses))
        print (f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{n_total_steps}], Train Loss: {train_loss.item():.4f}')
            
        model.eval()
        for batch_idx, (image, label) in enumerate(tqdm(test_loader)):
            # Get data to cuda if possible
            image = image.to(device=device)
            label = label.to(device=device)

            # forward
            with torch.no_grad():
                preds = model(image) # Pass batch
                
            val_loss = criterion(preds, label) # Calculate the loss
            
            # store loss
            val_batch_losses.append(val_loss.item())
            
        val_losses.append(sum(val_batch_losses)/len(val_batch_losses))
        print (f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{n_total_steps}], Val Loss: {val_loss.item():.4f}')



def main():
    """
    test the classifier training by training a Densenet201 on GC-10
    this code is taken in large part from Michel's notebook,
    see references/Michel_99_base_line_DenseNet_201_PyTorch.ipynb
    """
    in_channels = 3
    num_classes = 10
    learning_rate = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # use the model pretrained on imagenet
    model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', weights='IMAGENET1K_V1')

    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False

    # Modify the classifier part of the model
    # TODO figure out why -> Michel
    model.classifier = nn.Sequential(
                                nn.Linear(1920, 960),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(960, 240),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(240, 30),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(30, num_classes))
    model.to(device)


    

if __name__ == '__main__':
    
    main()