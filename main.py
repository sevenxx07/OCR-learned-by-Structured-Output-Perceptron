import numpy as np
import utils

dict_letter_to_int = {chr(x): x-97 for x in range(ord('a'), ord('z')+1)}
dict_int_to_letter = {v: k for k, v in dict_letter_to_int.items()}

CLASSES = 26
FEATURES = 8257


class IndependentMultiClass:
    def __init__(self, input_data, labels):
        """
                Constructor for the IndependentMultiClass classifier.

                Parameters:
                - input_data: List of feature vectors for each training example.
                - labels: List of corresponding labels for each training example.
        """
        self.weights = np.zeros((CLASSES, FEATURES))
        self.training_data = []
        self.labels = []

        for x, y in zip(input_data, labels):
            for features, letter in zip(x, y):
                self.training_data.append(np.array(features))
                self.labels.append(letter)

    def training(self):
        done = False
        while not done:
            done = True
            for i in range(len(self.training_data)):
                x = self.training_data[i]
                y = self.labels[i]
                pred_data = np.append(x, 1).reshape((1, 1, -1))
                pred = []
                for image in pred_data:
                    name = ''
                    for features in image:
                        # Append 1 to the features if its length is 8256
                        if len(features) == FEATURES - 1:
                            features = np.append(features, 1)
                        prediction = self.weights @ features
                        name = name + dict_int_to_letter[np.argmax(prediction)]
                    pred.append(name)
                pred = pred[0]
                if y == pred:
                    continue
                # Update weights based on perceptron learning algorithm
                self.weights[dict_letter_to_int[y], :] += np.append(x, 1)
                self.weights[dict_letter_to_int[pred], :] -= np.append(x, 1)
                done = False

    def test_errors(self, images, labels):
        pred = []
        corr_seq = 0.0
        corr_char = 0.0
        total_char = 0.0
        for image in images:
            name = ''
            for features in image:
                # Append 1 to the features if its length is 8256
                if len(features) == FEATURES - 1:
                    features = np.append(features, 1)
                prediction = self.weights @ features
                name = name + dict_int_to_letter[np.argmax(prediction)]
            pred.append(name)
        # Iterate through predicted and true sequences
        for p_1, p_2 in zip(pred, labels):
            if p_1 != p_2:
                corr_seq += 1
            # Iterate through predicted and true characters within sequences
            for char_1, char_2 in zip(p_1, p_2):
                total_char += 1
                if char_1 != char_2:
                    corr_char += 1
        return corr_seq/len(pred), corr_char/total_char


class StructuredPairWiseClass:
    def __init__(self, input_data, labels):
        """
        Constructor for the StructuredPairWiseClass classifier.

        Parameters:
        - input_data: List of input sequences for each training example.
        - labels: List of corresponding labels for each training example.
        """
        self.weights = np.zeros((CLASSES, FEATURES))
        self.g = np.zeros((CLASSES, CLASSES))
        self.training_data = []
        self.labels = []

        for x, y in zip(input_data, labels):
            self.training_data.append(np.array(x))
            self.labels.append(y)

    def training(self):
        done = False
        while not done:
            done = True
            for i in range(len(self.training_data)):
                x = self.training_data[i]
                y = self.labels[i]
                pred_data = x.reshape((1, len(x), -1))
                pred = []
                for p in pred_data:
                    p = np.hstack((p, np.ones((len(p), 1)))).T
                    px = self.weights @ p
                    name = self.prediction_search(px)
                    pred.append(''.join(name))
                pred = pred[0]
                if y == ''.join(pred):
                    continue
                done = False
                for l in range(len(pred)):
                    if y[l] != pred[l]:
                        self.weights[dict_letter_to_int[y[l]], :] += np.append(x[l], 1)
                        self.weights[dict_letter_to_int[pred[l]], :] -= np.append(x[l], 1)
                    if l < len(pred) - 1:
                        y_1 = dict_letter_to_int[y[l]]
                        y_2 = dict_letter_to_int[y[l + 1]]
                        pred_1 = dict_letter_to_int[pred[l]]
                        pred_2 = dict_letter_to_int[pred[l + 1]]
                        self.g[y_1, y_2] += 1
                        self.g[pred_1, pred_2] -= 1

    def prediction_search(self, p):
        #dynamic programming
        num_letters = len(p[0, :])
        graph = np.zeros((CLASSES, num_letters, 2))
        output = [''] * num_letters
        graph[:, 0, 0] = p[:, 0]
        for y in range(1, num_letters):
            for y2 in range(CLASSES):
                max_index = np.argmax(graph[:, y - 1, 0] + self.g[:, y2])
                graph[y2, y, 0] = p[y2, y] + graph[max_index, y - 1, 0] + self.g[max_index, y2]
                graph[y2, y, 1] = max_index
        index = np.argmax(graph[:, num_letters - 1, 0])
        for y in range(num_letters - 1, -1, -1):
            output[y] = dict_int_to_letter[index]
            index = graph[int(index), y, 1]
        return output

    def test_errors(self, images, labels):
        pred = []
        corr_seq = 0.0
        corr_char = 0.0
        total_char = 0.0
        for image in images:
            x = np.hstack((image, np.ones((len(image), 1)))).T
            px = self.weights @ x
            name = self.prediction_search(px)
            pred.append(''.join(name))
        # Iterate through predicted and true sequences
        for p_1, p_2 in zip(pred, labels):
            if p_1 != p_2:
                corr_seq += 1
            # Iterate through predicted and true characters within sequences
            for char_1, char_2 in zip(p_1, p_2):
                total_char += 1
                if char_1 != char_2:
                    corr_char += 1
        return corr_seq / len(pred), corr_char / total_char


class StructuredFixedSequencesClass:
    def __init__(self, input_data, labels):
        # Initialize weights and bias
        self.weights = np.zeros((CLASSES, FEATURES))
        self.bias = np.zeros((20,))
        # Store input data and labels
        self.input_data = input_data
        self.labels = labels
        # Extract unique labels
        self.unique_labels = set(labels)
        self.unique_labels = list(self.unique_labels)

    def training(self):
        done = False
        while not done:
            done = True
            for i in range(len(self.input_data)):
                x = self.input_data[i]
                y = self.labels[i]
                pred_data = x.reshape((1, len(x), -1))
                pred = []
                for p in pred_data:
                    name = ''
                    # Find label options with the same number of letters as the input
                    options = []
                    for label in self.unique_labels:
                        if len(label) == len(p):
                            options.append(label)
                    p = np.hstack((p, np.ones((len(p), 1)))).T
                    # Calculate prediction using weights and bias
                    prediction = self.weights @ p
                    max_value = -np.inf
                    for option in options:
                        value = 0
                        for l in range(len(option)):
                            value += prediction[dict_letter_to_int[option[l]], l]
                        #Add bias for the current option
                        value += self.bias[self.unique_labels.index(option)]
                        if value > max_value:
                            max_value = value
                            name = option
                    pred.append(name)
                pred = pred[0]
                if pred == y:
                    continue
                for l in range(len(pred)):
                    if y[l] != pred[l]:
                        # Update weights for the correct and predicted labels
                        self.weights[dict_letter_to_int[y[l]], :] += np.append(x[l], 1)
                        self.weights[dict_letter_to_int[pred[l]], :] -= np.append(x[l], 1)
                # Update bias for the correct and predicted labels
                self.bias[self.unique_labels.index(y)] += 1
                self.bias[self.unique_labels.index(pred)] -= 1
                done = False

    def test_errors(self, images, labels):
        pred = []
        corr_seq = 0.0
        corr_char = 0.0
        total_char = 0.0
        for image in images:
            name = ''
            # Find label options with the same number of letters as the input
            options = []
            for label in self.unique_labels:
                if len(label) == len(image):
                    options.append(label)
            x = np.hstack((image, np.ones((len(image), 1)))).T
            # Calculate prediction using weights and bias
            prediction = self.weights @ x
            max_value = -np.inf
            for option in options:
                value = 0
                for l in range(len(option)):
                    value += prediction[dict_letter_to_int[option[l]], l]
                # Add bias for the current option
                value += self.bias[self.unique_labels.index(option)]
                if value > max_value:
                    max_value = value
                    name = option
            pred.append(name)
        # Iterate through predicted and true sequences
        for p_1, p_2 in zip(pred, labels):
            if p_1 != p_2:
                corr_seq += 1
            # Iterate through predicted and true characters within sequences
            for char_1, char_2 in zip(p_1, p_2):
                total_char += 1
                if char_1 != char_2:
                    corr_char += 1

        return corr_seq / len(pred), corr_char / total_char


if __name__ == '__main__':
    # load training examples
    train_X, train_Y = utils.load_images('ocr_names_images/trn', 'Load train data')
    # load testing examples
    test_X, test_Y = utils.load_images('ocr_names_images/tst', 'Load test data')

    IndependentMultiClassPerceptron = IndependentMultiClass(train_X, train_Y)

    StructuredPairWiseClassPerceptron = StructuredPairWiseClass(train_X, train_Y)

    StructuredFixedSequencesClassPerceptron = StructuredFixedSequencesClass(train_X, train_Y)

    IndependentMultiClassPerceptron.training()
    StructuredPairWiseClassPerceptron.training()
    StructuredFixedSequencesClassPerceptron.training()

    errorSeq1, errorChar1 = IndependentMultiClassPerceptron.test_errors(test_X, test_Y)
    errorSeq2, errorChar2 = StructuredPairWiseClassPerceptron.test_errors(test_X, test_Y)
    errorSeq3, errorChar3 = StructuredFixedSequencesClassPerceptron.test_errors(test_X, test_Y)

    print(f"IndependentMultiClass Sequence error:{errorSeq1:.04f}")
    print(f"IndependentMultiClass Character error:{errorChar1:.04f}")

    print(f"StructuredPairWiseClass Sequence error:{errorSeq2:.04f}")
    print(f"StructuredPairWiseClass Character error:{errorChar2:.04f}")

    print(f"StructuredFixedSequencesClass Sequence error:{errorSeq3:.04f}")
    print(f"StructuredFixedSequencesClass Character error:{errorChar3:.04f}")

