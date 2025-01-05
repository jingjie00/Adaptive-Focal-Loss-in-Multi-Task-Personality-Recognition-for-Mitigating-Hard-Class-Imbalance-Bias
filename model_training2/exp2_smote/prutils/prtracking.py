
import os
import time
import torch

class TrackingManager:
    def __init__(self, checkpointname):
        self.best_validation_loss = float('inf')
        self.epoch = 0
        self.current_validation_loss = 0
        self.current_training_loss = 0

        # time
        if checkpointname is not None:
            self.checkpointname = checkpointname
        else:
            self.checkpointname = time.strftime("%Y%m%d-%H%M%S")

        #create a folder
        if not os.path.exists(f"checkpoints/{self.checkpointname}"):
            os.makedirs(f"checkpoints/{self.checkpointname}")

    def training_push(self, batch_train_loss):
        self.current_training_loss += batch_train_loss

    def validation_push(self, batch_validation_loss):
        self.current_validation_loss = batch_validation_loss

    def go_to_new_epoch(self, network):
        # save the network state
        torch.save(network.state_dict(), f"checkpoints/{self.checkpointname}/latest.pt")

        if self.current_validation_loss < self.best_validation_loss:
            self.best_validation_loss = self.current_validation_loss
            torch.save(network.state_dict(), f"checkpoints/{self.checkpointname}/best.pt")
            return_info = {"training_loss": self.current_training_loss, "validation_loss": self.current_validation_loss, "mark": "*"}
        else:
            return_info = {"training_loss": self.current_training_loss, "validation_loss": self.current_validation_loss, "mark": " "}
        
        self.epoch +=1
        self.current_training_loss = 0
        self.current_validation_loss = 0

        return return_info

        

if __name__ == "__main__":
    import torch.nn as nn
    import torch.optim as optim

    # Ensure the TrackingManager class is defined here as provided in your code

    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc = nn.Linear(10, 1)

        def forward(self, x):
            return self.fc(x)

    # Dummy training and validation loop
    def train_and_validate(model, manager, optimizer, loss_fn, epochs=5):
        for epoch in range(epochs):
            # Simulate a training loop
            model.train()
            for _ in range(10):  # Assuming 10 batches per epoch
                optimizer.zero_grad()
                dummy_input = torch.randn(10, 10)  # 10 samples per batch
                dummy_target = torch.randn(10, 1)
                output = model(dummy_input)
                loss = loss_fn(output, dummy_target)
                loss.backward()
                optimizer.step()

                # Update training loss
                manager.training_push(loss.item())

            # Simulate a validation loop
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for _ in range(5):  # Assuming 5 validation batches
                    dummy_input = torch.randn(10, 10)
                    dummy_target = torch.randn(10, 1)
                    output = model(dummy_input)
                    loss = loss_fn(output, dummy_target)
                    val_loss += loss.item()

                # Update validation loss
                manager.validation_push(val_loss / 5)  # Average validation loss

            # New epoch updates
            manager.new_epoch(model)

    # Create a simple neural network, optimizer, and loss function
    model = SimpleNN()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # Instantiate the TrackingManager
    manager = TrackingManager("dummy_test")

    # Run the training and validation
    train_and_validate(model, manager, optimizer, loss_fn)
