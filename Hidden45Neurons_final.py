# Importing
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon

# Define the coordinates for the pentagon
pentagon1_coords = [(0.5, 1.5), (1.5, 0), (-0.5, 0)]  # creating triangle 1
pentagon2_coords = [(0.5, -0.5), (-0.5, 1), (1.5, 1)]  # creating triangle 2 (upside down)

# Create a Polygon using the coordinates
pentagon1 = Polygon(pentagon1_coords)
pentagon2 = Polygon(pentagon2_coords)
combined_polygon = pentagon1.union(pentagon2)

# Plot the pentagon
fig, ax = plt.subplots()
fig.suptitle("The Star Shape from -0.5 to +1.5")
ax.fill(*combined_polygon.exterior.xy)

# Set the aspect ratio and axis limits
ax.set_aspect('equal')
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-0.5, 1.5)

# Show the plot
plt.show(block=False)


# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Define the function to determine if a point is inside the pentagon
def is_inside(point):
    return combined_polygon.contains(Point(point)) or combined_polygon.touches(Point(point))


# Define the feedforward function
def feedforward(inputs, input_weights, output_weights):
    hidden_layer = sigmoid(np.dot(inputs, input_weights))
    output = sigmoid(np.dot(hidden_layer, output_weights))
    return output, hidden_layer


# Define the backpropagation function
def backpropagation(inputs, hidden_layer, output, target, input_weights, output_weights, learning_rate):
    output_error = output - target
    output_gradient = output_error * sigmoid_derivative(np.dot(hidden_layer, output_weights))

    hidden_error = np.dot(output_gradient, output_weights.T)
    hidden_gradient = hidden_error * sigmoid_derivative(np.dot(inputs, input_weights))

    output_weights -= learning_rate * hidden_layer.reshape(-1, 1) * output_gradient
    input_weights -= learning_rate * inputs.reshape(-1, 1) * hidden_gradient

    return input_weights, output_weights


# Define the training function
def train(X_train, y_train, epochs, learning_rate, hidden_len):
    # np.random.seed(0)
    input_weights = np.random.uniform(-1, 1, (2, hidden_len))
    output_weights = np.random.uniform(-1, 1, (hidden_len, 1))

    # learning_rate = 0.1
    # learning_decay = 0.99
    error_rate = list()

    for epoch in range(epochs):
        correct_predictions = 0
        for inputs, target in zip(X_train, y_train):

            output, hidden_layer = feedforward(inputs, input_weights, output_weights)

            # learning_rate = learning_rate * learning_decay

            input_weights, output_weights = backpropagation(inputs, hidden_layer, output, target, input_weights,
                                                            output_weights, learning_rate)

            # Count correct predictions
            prediction = 1 if output >= 0.5 else 0
            if prediction == target:
                correct_predictions += 1

        success_rate = correct_predictions / len(X_train)
        print(f"Epoch {epoch + 1}/{epochs} - Success Rate: {success_rate * 100:.2f}%")

        error_rate.append((1 - (correct_predictions / len(X_train))) * 100)
    x_axis = [epoch + 1 for epoch in range(epochs)]
    fig, ax = plt.subplots()
    plt.plot(x_axis, error_rate)
    plt.xlabel('epochs')
    plt.ylabel('error rate')
    fig.suptitle('error rate progress', ha='center', fontsize=16)
    plt.show(block=False)

    return input_weights, output_weights


# Define the main function
def main():
    # np.random.seed(0)
    data = np.random.uniform(low=-0.5, high=1.5, size=(28000, 2))

    train_ratio = 0.8
    train_size = int(train_ratio * data.shape[0])
    X_train = data[:train_size]
    y_train = np.array([is_inside(point) for point in X_train])

    # Train the neural network
    epochs = 150
    learning_rate = 0.1
    hidden_len = 40
    input_weights, output_weights = train(X_train, y_train, epochs, learning_rate, hidden_len)

    X_test = data[train_size:]
    inside_points = []
    outside_points = []
    hidden_in = [[] for _ in range(hidden_len)]
    hidden_out = [[] for _ in range(hidden_len)]

    for i in range(len(X_test)):
        input_point = X_test[i]
        output, hidden_layer = feedforward(input_point, input_weights, output_weights)

        for j in range(hidden_len):
            if round(hidden_layer[j]) == 1:
                hidden_in[j].append(input_point)
            else:
                hidden_out[j].append(input_point)

        is_inside_pentagon = output >= 0.5
        if is_inside_pentagon == 1:
            inside_points.append(input_point)
        if is_inside_pentagon == 0:
            outside_points.append(input_point)
        print("Point:", input_point)
        print("Predicted Inside Pentagon:", is_inside_pentagon)
        print("Actual Inside Pentagon:", bool(is_inside(input_point)))
        print()

    inside_points = np.array(inside_points)
    outside_points = np.array(outside_points)
    fig, ax = plt.subplots()
    plt.plot(inside_points[:, 0], inside_points[:, 1], 'g+')
    # plt.scatter(inside_points[0], inside_points[1])
    plt.plot(outside_points[:, 0], outside_points[:, 1], 'r+')
    # plt.scatter(outside_points[0], outside_points[1])
    fig.suptitle('Points Classification', ha='center', fontsize=16)
    plt.show(block=False)

    fig = plt.figure(figsize=(30, 30))
    # hidden_in = np.array(hidden_in)
    # hidden_out = np.array(hidden_out)
    for image in range(hidden_len):
        ax = plt.subplot(5, 8, image + 1)
        ax.title.set_text('Neuron ' + str(image + 1))
        # Set the aspect ratio and axis limits
        ax.set_aspect('equal')
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        # hidden_in[image] = np.array(hidden_in[image])
        # hidden_out[image] = np.array(hidden_out[image])
        if len(hidden_in[image]) > 0:
            hidden_in_x = [hidden_in[image][i][0] for i in range(len(hidden_in[image]))]
            hidden_in_y = [hidden_in[image][i][1] for i in range(len(hidden_in[image]))]
            plt.plot(hidden_in_x, hidden_in_y, 'g+')
        if len(hidden_out[image]) > 0:
            hidden_out_x = [hidden_out[image][i][0] for i in range(len(hidden_out[image]))]
            hidden_out_y = [hidden_out[image][i][1] for i in range(len(hidden_out[image]))]
            plt.plot(hidden_out_x, hidden_out_y, 'r+')
    plt.legend(["inside star", "outside star"])
    # Adjust layout to prevent overlap
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.7, hspace=0.7)
    fig.suptitle('Classification of neurons in hidden layer', ha='center', fontsize=16)
    plt.show(block=False)

    # Final blocking show to keep all windows open
    plt.show()

# Run the main function
if __name__ == "__main__":
    main()
