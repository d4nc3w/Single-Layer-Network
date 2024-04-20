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

# def train(data, labels):
#     # Implement the training algorithm
#
# def test(data, labels):
#     # Implement the testing algorithm

train_input_vectors, train_labels = loadData(lang_train)
test_input_vectors, test_labels = loadData(lang_test)

print("Checking reading line logic...")
for i in range(3):
    print(f"Sample {i + 1}:")
    print(f"Label: {train_labels[i]}")
    print(f"Input vector: {train_input_vectors[i]}")
