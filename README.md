# Single-Layer-Network
The aim of the project is to create a single-layer netowrk to identify the language of input texts.

Documantation:

Perceptron Function

    def perceptron(vec, weights, biases):
        net_input = np.dot(weights, vec) + biases
        activation = net_input
        return activation
          
* This function calculates the net input and activation for the input vector using the provided weights and biases.
* It uses a linear activation function (the output is the same as net input).
* Returns the activation values.

Load Data Function

    def loadData(file):
      vectors = []
      labels = []
  
    def simplifyText(text):
        cleaned_text = ''.join([char for char in text if char.isalpha()]).lower()
        return cleaned_text
    
    with open(file, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for line in reader:
            # Separate label from text
            label, text = line[0], line[1]
            text = simplifyText(text)
            
            freq_array = [0] * 26
            for char in text:
                index = ord(char) - ord('a')
                if 0 <= index < 26:
                    freq_array[index] += 1
            
            input_vector = np.array(freq_array)
            norm = np.linalg.norm(input_vector)
            if norm != 0:
                input_vector = input_vector / norm
            
            vectors.append(input_vector)
            labels.append(label)
    
    return vectors, labels

* This function loads data from a CSV file.
* For each line, it separates the label from the text.
* Text is simplified to lowercase letters, and a frequency array is created to count occurrences of each letter.
* Input vectors are created, normalized, and appended to vectors.
* Labels are appended to labels.

Train Function 

    def train(data, labels, learning_rate=0.01, epochs=100):
        num_languages = len(np.unique(labels))
    
    weights = np.random.rand(num_languages, 26)
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

* This function trains the neural network using the provided data and labels.
* It initializes weights and biases based on the number of languages.
* For each epoch, it loops through all samples, calculates the net input and activation, and updates the weights and biases based on the error between expected and predicted outputs.
* It calculates the total loss for each epoch and prints it.

Test Function

    def test(data, labels, weights, biases):
        correct = 0
        total = len(data)
    
    for i in range(total):
        input_vector = data[i]
        expected_label = labels[i]

        output = perceptron(input_vector, weights, biases)

        predicted_language_index = np.argmax(output)

        languages = ['English', 'German', 'Polish', 'Spanish']
        predicted_language = languages[predicted_language_index]
        
        if predicted_language == expected_label:
            correct += 1
            print(f"(+) Expected: {expected_label} | Predicted: {predicted_language}")
        else:
            print(f"(-) Expected: {expected_label} | Predicted: {predicted_language} <----")
    
    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy

* This function tests the trained neural network using the provided test data and labels.
* For each sample, it uses the perceptron function to calculate the output and determines the predicted language.
* It keeps track of correct predictions and prints the expected and predicted labels for each sample.
* It calculates and prints the test accuracy.

Classify Function

    def classify(text, weights, biases):
        freq_array = [0] * 26
    
    for char in text:
        index = ord(char) - ord('a')
        if 0 <= index < 26:
            freq_array[index] += 1
    
    input_vector = np.array(freq_array)
    norm = np.linalg.norm(input_vector)
    if norm != 0:
        input_vector = input_vector / norm
    
    output = perceptron(input_vector, weights, biases)
    
    predicted_language_index = np.argmax(output)
    
    languages = ['English', 'German', 'Polish', 'Spanish']
    predicted_language = languages[predicted_language_index]
    
    return predicted_language

* This function classifies a given text using the trained neural network.
* It calculates the frequency of each letter in the text and normalizes the input vector.
* It uses the perceptron function to calculate the output and finds the index of the perceptron with the maximum activation.
* It maps the index to the corresponding language and returns the predicted language.

MENU (UI)

    isTrained = False
    while True:
        print("----------MENU----------")
        print("(1) Train Model")
        print("(2) Test Model")
        print("(3) Classify input text")
        print("(4) Exit")
        print("------------------------")
        choice = int(input("Enter your choice: "))
        
    if choice == 1:
        train_input_vectors, train_labels = loadData(lang_train)
        weights, biases = train(train_input_vectors, train_labels)
        isTrained = True
    
    elif choice == 2:
        if not isTrained:
            print("Model not trained yet")
        else:
            test_input_vectors, test_labels = loadData(lang_test)
            test(test_input_vectors, test_labels, weights, biases)
    
    elif choice == 3:
        if not isTrained:
            print("Model not trained yet")
        else:
            text = input("Enter the text to classify: ")
            predicted_language = classify(text, weights, biases)
            print(f"Predicted language: {predicted_language}")
    
    elif choice == 4:
        print("Closing...")
        exit()

* This section provides a menu-driven user interface that allows users to train the model, test the model, or classify input text.
* Depending on the user's choice, the program either trains the model, tests it, or allows the user to input text to classify using the trained model.
