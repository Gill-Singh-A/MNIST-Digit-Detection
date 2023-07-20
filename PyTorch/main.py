#!/usr/bin/env python3

import torch, torchvision, pickle, pygame, numpy
from torch import nn
from torch import optim
from pathlib import Path
from datetime import date
from optparse import OptionParser
from torchvision import transforms
from torch.nn import functional as function
from matplotlib import pyplot as plot
from colorama import Fore, Back, Style
from time import strftime, localtime, time

batch_size = 128
learning_rate = 0.001
momentum = 0.9
epoches = 10
side = 30
border = 1

WHITE = (255, 255, 255)
GRAY = (50, 50, 50)
BRIGHT_GRAY = (150, 150, 150)
BLACK = (0, 0, 0)

COLOR = {0: BLACK, 1: WHITE}

status_color = {
	'+': Fore.GREEN,
	'-': Fore.RED,
	'*': Fore.YELLOW,
	':': Fore.CYAN,
	' ': Fore.WHITE,
}

def get_time():
	return strftime("%H:%M:%S", localtime())
def display(status, data):
	print(f"{status_color[status]}[{status}] {Fore.BLUE}[{date.today()} {get_time()}] {status_color[status]}{Style.BRIGHT}{data}{Fore.RESET}{Style.RESET_ALL}")

def get_arguments(*args):
	parser = OptionParser()
	for arg in args:
		parser.add_option(arg[0], arg[1], dest=arg[2], help=arg[3])
	return parser.parse_args()[0]

class neuralNetwork(nn.Module):
    def __init__(self):
        super(neuralNetwork, self).__init__()
        self.conv_1 = nn.Conv2d(1, 32, 3)              # Making a 2D Convolutional Filter Layer with Input Channel = 1, Output Channel = 32, Kernel Size = 3
        self.conv_2 = nn.Conv2d(32, 64, 3)             # Making a 2D Convolutional Filter Layer with Input Channel = 32, Output Channel = 64, Kernel Size = 3
        self.pool = nn.MaxPool2d(2, 2)                 # Making a 2D Max Pool Layer with Kernel Size = 2, Stride = 2
        self.fc_1 = nn.Linear(9216, 128)               # Flattening the Previous Layer into a Fully Connected Layer
        self.fc_2 = nn.Linear(128, 10)                 # Fully Connecting Input 128 Nodes with Output 10 Nodes
    def forward(self, data):                           # Forward Propogation Function
        data = function.relu(self.conv_1(data))
        data = self.pool(function.relu(self.conv_2(data)))
        data = data.view(-1, 9216)
        data = function.relu(self.fc_1(data))
        data = self.fc_2(data)
        return data

def getDevice(demand):
	if demand == "gpu" and torch.cuda.is_available():
		display('+', f"Using {Back.MAGENTA}GPU{Back.RESET} for training Neural Network")
		return "cuda"
	elif demand == "gpu" and not torch.cuda.is_available():
		display('-', f"Can't use {Back.MAGENTA}GPU{Back.RESET} for training Neural Network")
		display('*', f"Using {Back.MAGENTA}GPU{Back.RESET} for training Neural Network")
		return "cpu"
	else:
		display('+', f"Using {Back.MAGENTA}CPU{Back.RESET} for training Neural Network")
		return "cpu"
def plotAccuracyLoss(epoch_log, accuracy_log, loss_log):
    figure, axis = plot.subplots()
    plot.title("Accuracy & Loss vs Epoch")
    axis_1 = axis.twinx()
    axis.plot(epoch_log, loss_log, 'g-')
    axis_1.plot(epoch_log, accuracy_log, 'b-')
    axis.set_xlabel("Epochs")
    axis.set_ylabel("Loss", color='g')
    axis_1.set_ylabel("Test Accuracy", color='b')
    plot.show()
def drawPixels(window, matrix, border, hover):
    side = window.get_width() // 28
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            pygame.draw.rect(window, COLOR[value], (col_index*side, row_index*side, side, side))
            pygame.draw.rect(window, GRAY, (col_index*side, row_index*side, side, side), border)
    pygame.draw.rect(window, BRIGHT_GRAY, (hover[0]*side, hover[1]*side, side, side))
    pygame.draw.rect(window, GRAY, (hover[0]*side, hover[1]*side, side, side), border)

if __name__ == "__main__":
    data = get_arguments(('-d', "--device", "device", "Device to use for training the Neural Network (cpu/gpu)"),
                         ('-s', "--save", "save", "Name for the Model File to be saved (Default=Current Date and Time) in 'models' Folder"),
                         ('-l', "--load", "load", "Load an existing Model File (stored in 'models' folder)"),
                         ('-b', "--batch", "batch", f"Batch Size for the Training (Default={batch_size})"),
                         ('-r', "--learning-rate", "learning_rate" , f"Learning Rate for Loss Function (Default={learning_rate})"),
                         ('-m', "--momentum", "momentum", f"Momentum for Loss Function (Default={momentum})"),
						 ('-e', "--epoches", "epoches", f"Number of Epoches for Training (Default={epoches})"),
                         ('-w', "--side", "side", f"Size of a single Square (Default={side})"),
                         ('-c', "--border-size", "border_size", f"Size of Border of Square (Default={border})"))
    device = getDevice(data.device)
    if not data.batch:
        data.batch = batch_size
    else:
        data.batch = int(data.batch)
    if not data.learning_rate:
        data.learning_rate = learning_rate
    else:
        data.learning_rate = int(data.learning_rate)
    if not data.momentum:
        data.momentum = momentum
    else:
        data.momentum = int(data.momentum)
    if not data.epoches:
        data.epoches = epoches
    else:
        data.epoches = int(data.epoches)
    if not data.side:
        data.side = side
    else:
        data.side = int(data.side)
    if not data.border_size:
        data.border_size = border
    else:
        data.border_size = int(data.border_size)
    if not data.load:
        display('+', f"Setting up the {Back.MAGENTA}Transformer{Back.RESET}")
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])                         # Making Transformer Object for pre-processing Data and converting the 0-255 GrayScale Pixel Values between -1 and 1 and converting them to tensor Object
        display('+', f"Setting up the {Back.MAGENTA}Training Set{Back.RESET}")
        trainset = torchvision.datasets.MNIST("mnist", train=True, download=True, transform=transform)
        display('+', f"Setting up the {Back.MAGENTA}Testing Set{Back.RESET}")
        testset = torchvision.datasets.MNIST("mnist", train=False, download=True, transform=transform)
        display(':', f"Number of Image Samples for {Back.MAGENTA}Train Set{Back.RESET} = {Back.MAGENTA}{trainset.data.shape[0]}{Back.RESET}")
        display(':', f"Number of Image Samples for {Back.MAGENTA}Test Set{Back.RESET}  = {Back.MAGENTA}{testset.data.shape[0]}{Back.RESET}")
        display(':', f"Dimensions of Images = {Back.MAGENTA}{trainset.data.shape[2]}x{trainset.data.shape[1]}{Back.RESET}")
        display('+', f"Setting up the {Back.MAGENTA}Loader for Training Set{Back.RESET}")
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=data.batch, shuffle=True, num_workers=0)
        display('+', f"Setting up the {Back.MAGENTA}Loader for Testing Set{Back.RESET}")
        testloader = torch.utils.data.DataLoader(testset, batch_size=data.batch, shuffle=False, num_workers=0)
        display('+', f"Setting up the {Back.MAGENTA}Network{Back.RESET}")
        network = neuralNetwork()
        network.to(device)
        display('+', f"Setting up Loss {Back.MAGENTA}Functions{Back.RESET}")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(network.parameters(), lr=data.learning_rate, momentum=data.momentum)
        display('+', "Starting The Training for Neural Network")
        epoch_log, loss_log, accuracy_log = [], [], []
        time_1 = time()
        for epoch in range(data.epoches):
            display('+', f"Starting Epoch: {Back.MAGENTA}{epoch+1}{Back.RESET}")
            running_loss = 0.0
            for index, trainer_data in enumerate(trainloader):
                inputs, labels = trainer_data
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = network(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if index % 50 == 49:
                    correct, total = 0, 0
                    with torch.no_grad():
                        for test_data in testloader:
                            images, labels = test_data
                            images = images.to(device)
                            labels = labels.to(device)
                            outputs = network(images)
                            _, predicted = torch.max(outputs.data, dim=1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                    accuracy = 100 * correct / total
                    epoch_num = epoch + 1
                    actual_loss = running_loss / 50
                    display(':', f"Epoch: {Back.MAGENTA}{epoch_num}{Back.RESET}, Mini-Batches Completed: {Back.MAGENTA}{index+1}{Back.RESET}, Loss: {Back.MAGENTA}{actual_loss:.3f}{Back.RESET}, Test Accuracy: {Back.MAGENTA}{accuracy:.3f}{Back.RESET}")
                    running_loss = 0.0
                    epoch_log.append(epoch_num)
                    loss_log.append(actual_loss)
                    accuracy_log.append(accuracy)
        display('+', "Finished Training")
        time_2 = time()
        display(':', f"Time Taken to Train the Neural Network = {Back.MAGENTA}{time_2-time_1:.2f}{Back.RESET} seconds")
        if not data.save:
            data.save = str(date.today()) + ' ' + strftime("%H_%M_%S", localtime())
        directory = Path.cwd() / "models" / data.save
        directory.mkdir(exist_ok=True, parents=True)
        with open(f"models/{data.save}/epoch_log", 'wb') as file:
            pickle.dump(epoch_log, file)
        with open(f"models/{data.save}/loss_log", 'wb') as file:
            pickle.dump(loss_log, file)
        with open(f"models/{data.save}/accuracy_log", 'wb') as file:
            pickle.dump(accuracy_log, file)
        torch.save(network.state_dict(), f"models/{data.save}/neuralNetwork.nn")
        plotAccuracyLoss(epoch_log, accuracy_log, loss_log)
    else:
        display('+', f"Setting up the {Back.MAGENTA}Network{Back.RESET}")
        network = neuralNetwork()
        network.to(device)
        display(':', f"Loading the Model File {Back.MAGENTA}{data.load}{Back.RESET}")
        try:
            network.load_state_dict(torch.load(f"models/{data.load}/neuralNetwork.nn"))
        except:
            display('-', f"Can't load File {Back.MAGENTA}{data.load}{Back.RESET}")
            exit(0)
        with open(f"models/{data.load}/epoch_log", 'rb') as file:
            epoch_log = pickle.load(file)
        with open(f"models/{data.load}/loss_log", 'rb') as file:
            loss_log = pickle.load(file)
        with open(f"models/{data.load}/accuracy_log", 'rb') as file:
            accuracy_log = pickle.load(file)
        display('+', f"Loaded the Model File {Back.MAGENTA}{data.load}{Back.RESET}")
        plotAccuracyLoss(epoch_log, accuracy_log, loss_log)
    pygame.init()
    window = pygame.display.set_mode((data.side*28, data.side*28))
    pygame.display.set_caption("Draw Numbers")
    matrix = [[0 for __ in range(28)] for _ in range(28)]                             # 0 For Black, 1 for White
    running, mouse_button_pressed, draw = True, False, True
    while running:
        x_mouse, y_mouse = [position//data.side for position in pygame.mouse.get_pos()]
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d:
                    display(':', f"Set Mode to {Back.MAGENTA}Drawing{Back.RESET}")
                    draw = True
                if event.key == pygame.K_e:
                    display(':', f"Set Mode to {Back.MAGENTA}Erase{Back.RESET}")
                    draw = False
                if event.key == pygame.K_c:
                    display(':', "Clearing the Screen")
                    matrix = [[0 for __ in range(28)] for _ in range(28)]
                if event.key == pygame.K_EQUALS:
                    start = time()
                    image = numpy.array([matrix], dtype=numpy.float32)
                    image[image == 0] = -1.0
                    torch_image = torch.from_numpy(image)
                    torch_image = torch_image.to(device)
                    output = network(torch_image)
                    probabilities = output.data.tolist()[0]
                    end = time()
                    display('+', f"Predicted Number = {Back.MAGENTA}{probabilities.index(max(probabilities))}{Back.RESET}, Time Taken = {Back.MAGENTA}{end-start}{Back.RESET} seconds")
        if pygame.mouse.get_pressed()[0] and x_mouse >= 0 and x_mouse < 28 and y_mouse >= 0 and y_mouse < 28:
            matrix[y_mouse][x_mouse] = int(draw)
        drawPixels(window, matrix, data.border_size, (x_mouse, y_mouse))
        pygame.display.update()
    pygame.quit()