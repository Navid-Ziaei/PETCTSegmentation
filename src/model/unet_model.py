import torch
import torch.nn as nn
import torch.optim as optim


class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)  # Ensure num_classes output channels
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def fit(self, train_loader, val_loader, optimizer, criterion, num_epochs, device):
        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            for batch in train_loader:
                ctres = batch['ctres'].unsqueeze(1).to(device)  # Add channel dimension
                suv = batch['suv'].unsqueeze(1).to(device)  # Add channel dimension
                label = batch['label'].to(device)

                # Concatenate CTres and SUV images along the channel dimension
                inputs = torch.cat((ctres, suv), dim=1)

                optimizer.zero_grad()
                outputs = self(inputs)

                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    ctres = batch['ctres'].unsqueeze(1).to(device)  # Add channel dimension
                    suv = batch['suv'].unsqueeze(1).to(device)  # Add channel dimension
                    label = batch['label'].to(device)

                    # Concatenate CTres and SUV images along the channel dimension
                    inputs = torch.cat((ctres, suv), dim=1)

                    outputs = self(inputs)
                    loss = criterion(outputs, label)

                    val_loss += loss.item()

            val_epoch_loss = val_loss / len(val_loader)
            print(f'Validation Loss: {val_epoch_loss:.4f}')
