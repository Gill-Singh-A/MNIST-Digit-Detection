#!/usr/bin/env python3

import pickle, pygame, numpy
from pathlib import Path
from datetime import date
from optparse import OptionParser
from matplotlib import pyplot as plot
from colorama import Fore, Back, Style
from time import strftime, localtime, time
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import backend
from tensorflow.keras.optimizers import SGD

batch_size = 128
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

def createModel(dataset, truth_values):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=dataset.shape[1:]))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(truth_values.shape[1], activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer=SGD(0.001), metrics=["accuracy"])
    return model
def plotLossAccuracy(history_dict):
    display(':', "Plotting Loss vs Epoches")
    plot.title("Loss vs Epoches")
    plot.xlabel("Epoches")
    plot.ylabel("Losses")
    plot.plot([x+1 for x in range(data.epoches)], history_dict["loss"], "r*-")
    plot.plot([x+1 for x in range(data.epoches)], history_dict["val_loss"], "b*-")
    plot.legend(["Loss", "Validation Loss"])
    plot.show()
    display(':', "Plotting Accuracy vs Epoches")
    plot.title("Accuracy vs Epoches")
    plot.xlabel("Accuracy")
    plot.ylabel("Losses")
    plot.plot([x+1 for x in range(data.epoches)], history_dict["accuracy"], "r*-")
    plot.plot([x+1 for x in range(data.epoches)], history_dict["val_accuracy"], "b*-")
    plot.legend(["Accuracy", "Validation Accuracy"])
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
    data = get_arguments(('-s', "--save", "save", "Name for the Model File to be saved (Default=Current Date and Time) in 'models' Folder"),
                         ('-l', "--load", "load", "Load an existing Model File (stored in 'models' folder)"),
                         ('-b', "--batch", "batch", f"Batch Size for the Training (Default={batch_size})"),
						 ('-e', "--epoches", "epoches", f"Number of Epoches for Training (Default={epoches})"),
                         ('-w', "--side", "side", f"Size of a single Square (Default={side})"),
                         ('-c', "--border-size", "border_size", f"Size of Border of Square (Default={border})"))
    if not data.batch:
        data.batch = batch_size
    else:
        data.batch = int(data.batch)
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
        display(':', "Loading Data")
        (training_images, training_truth_values), (testing_images, testing_truth_values) = mnist.load_data()
        display('+', "Loaded Data")
        display(':', "Preprocessing Data")
        training_images = training_images.reshape(training_images.shape[0], training_images.shape[1], training_images.shape[2], 1)
        testing_images = testing_images.reshape(testing_images.shape[0], testing_images.shape[1], testing_images.shape[2], 1)
        training_images = training_images.astype("float32")
        testing_images = testing_images.astype("float32")
        training_images /= 255.0
        testing_images /= 255.0
        training_truth_values = to_categorical(training_truth_values)
        testing_truth_values = to_categorical(testing_truth_values)
        display('+', "Preprocessed Data")
        display(':', "Creating Model")
        model = createModel(training_images, training_truth_values)
        display('+', "Created Model")
        display(':', "Training Model")
        start = time()
        history = model.fit(training_images, training_truth_values, batch_size=data.batch, epochs=data.epoches, verbose=True, validation_data=(testing_images, testing_truth_values))
        end = time()
        display('+', f"Trained Model (Time Taken = {Back.MAGENTA}{end-start} seconds{Back.RESET})")
        history_dict = history.history
        plotLossAccuracy(history_dict)
        if not data.save:
            data.save = str(date.today()) + ' ' + strftime("%H_%M_%S", localtime())
        directory = Path.cwd() / "models" / data.save
        directory.mkdir(exist_ok=True, parents=True)
        display(':', "Saving Model")
        with open(f"models/{data.save}/history", 'wb') as file:
            pickle.dump(history_dict, file)
        model.save(f"models/{data.save}/model")
        display('+', "Saved Model")
    else:
        display(':', f"Loading the Model File {Back.MAGENTA}{data.load}{Back.RESET}")
        try:
            model = load_model(f"models/{data.load}/model")
        except:
            display('-', f"Can't load File {Back.MAGENTA}{data.load}{Back.RESET}")
            exit(0)
        with open(f"models/{data.load}/history", 'rb') as file:
            history_dict = pickle.load(file)
        display('+', f"Loaded the Model File {Back.MAGENTA}{data.load}{Back.RESET}")
        plotLossAccuracy(history_dict)
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
                    image = image.reshape(image.shape[0], image.shape[1], image.shape[2], 1)
                    probabilities = model.predict(image, verbose=False)
                    end = time()
                    display('+', f"Predicted Number = {Back.MAGENTA}{numpy.argmax(probabilities)}{Back.RESET}, Time Taken = {Back.MAGENTA}{end-start}{Back.RESET} seconds")
        if pygame.mouse.get_pressed()[0] and x_mouse >= 0 and x_mouse < 28 and y_mouse >= 0 and y_mouse < 28:
            matrix[y_mouse][x_mouse] = int(draw)
        drawPixels(window, matrix, data.border_size, (x_mouse, y_mouse))
        pygame.display.update()
    pygame.quit()