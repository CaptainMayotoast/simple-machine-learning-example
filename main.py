from torch.utils.data import DataLoader
from iris_dataloader import IrisDataset
from iris_nn import NeuralNetwork
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as f

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

if __name__ == "__main__":

    training_data = IrisDataset()
    test_data = IrisDataset(train=False)
    iris_model = NeuralNetwork()

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=30, shuffle=True)

    test_features, test_labels = next(iter(test_dataloader))
    print(f"Feature batch shape: {test_features.size()}")
    print(f"Labels batch shape: {test_labels.size()}")
    print(f"Feature batch {test_features}")
    print(f"Labels batch: {test_labels}")

    output = iris_model(test_features)
    print(output)
    print(output.shape)

    learning_rate = 1e-3
    batch_size = 64
    epochs = 50

    # Initialize the loss function
    loss_fn = nn.CrossEntropyLoss()

    optimizer = Adam(iris_model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, iris_model, loss_fn, optimizer)
    print("Done!")

    # predictions
    preds = iris_model(test_features)
    print(preds.argmax(dim=1))
    print(test_labels)
    # sum and item is a hack to not convert to numeric entry
    # "how accurate was my neural networking in classifying the data?"
    correct = (preds.argmax(dim=1) == test_labels).sum().item()
    print(correct)

    correct = (preds.argmax(dim=1) == test_labels)
    print(test_labels[21])

    # the number of correct outcomes
    # print(correct)
    # print logits
    print(preds[21, :])

    # display probabilities
    print(f.softmax(preds[21, :]))



    # test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    ################################################################
    # next steps: expand to the entire 150 entry dataset
    # save and load the model: https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
    # look into torch vision