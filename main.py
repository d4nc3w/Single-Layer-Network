import csv
import numpy as np

lang_train = 'res/lang.train.csv'
lang_test = 'res/lang.test.csv'

def loadData(file):
    vectors = []
    labels = []
    def simplifyText(text):
        # Keep only letters A-Z and convert to lowercase
        cleaned_text = ''.join([char for char in text if char.isalpha()]).lower()
        return cleaned_text

    with open(file, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for line in reader:
            # Separate label from text
            label, text = line[0], line[1]
            text = simplifyText(text)

            # Create a frequency array for the text
            frequency_array = [0] * 26
            for char in text:
                index = ord(char) - ord('a')
                if 0 <= index < 26:
                    frequency_array[index] += 1

            # Convert to numpy
            input_vector = np.array(frequency_array)
            # Normalize vec
            norm = np.linalg.norm(input_vector)
            if norm != 0:
                input_vector = input_vector / norm

            vectors.append(input_vector)
            labels.append(label)
    return vectors, labels

def train(data, labels, learning_rate=0.01, epochs=100):
    num_languages = 4
    input_size = 26

    weights = np.random.rand(num_languages, input_size)
    biases = np.zeros(num_languages)

    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(data)):
            input_vector = data[i]
            expected_label = labels[i]

            expected_output = np.zeros(num_languages)
            language_index = ['English', 'German', 'Polish', 'Spanish'].index(expected_label)
            expected_output[language_index] = 1

            net_input = np.dot(weights, input_vector) + biases
            activation = net_input
            error = expected_output - activation

            weights += learning_rate * np.outer(error, input_vector)
            biases += learning_rate * error

            total_loss += np.sum(error ** 2)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    return weights, biases

def test(data, labels, weights, biases):
    correct_predictions = 0
    total_samples = len(data)

    for i in range(total_samples):
        input_vector = data[i]
        expected_label = labels[i]

        net_input = np.dot(weights, input_vector) + biases
        output = net_input

        predicted_language_index = np.argmax(output)
        languages = ['English', 'German', 'Polish', 'Spanish']
        predicted_language = languages[predicted_language_index]

        if predicted_language == expected_label:
            correct_predictions += 1
            print(f"Expected: {expected_label} | Predicted: {predicted_language}")
        else:
            print(f"Expected: {expected_label} | Predicted: {predicted_language}")

    accuracy = correct_predictions / total_samples
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return accuracy

isTrained = False
while True:
    print("----------MENU----------")
    print("(1) Train Model")
    print("(2) Test Model")
    print("(3) Exit")
    choice = int(input("Enter your choice: "))
    print("------------------------")
    if choice == 1:
        train_input_vectors, train_labels = loadData(lang_train)
        weights, biases = train(train_input_vectors, train_labels)
        isTrained = True
    if choice == 2:
        if(isTrained == False):
            print("Model is not trained yet!")
            continue
        else:
            test_input_vectors, test_labels = loadData(lang_test)
            test(test_input_vectors, test_labels, weights, biases)
    if choice == 3:
        exit()

# train_input_vectors, train_labels = loadData(lang_train)
# test_input_vectors, test_labels = loadData(lang_test)
# weights, biases = train(train_input_vectors, train_labels)
# test(test_input_vectors, test_labels, weights, biases)