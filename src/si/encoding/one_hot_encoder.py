import numpy as np

# do a onehotencoder class

class OneHotEncoder:
    """
    OneHotEncoder is a class that performs one-hot encoding on sequences of characters.
    """

    def __init__(self, padder : str = '0', max_length : int = None):
        """
        Initializes the OneHotEncoder object.

        Parameters:
        - padder (str): The character used for padding sequences. Default is '0'.
        - max_length (int): The maximum length of sequences. If None, it will be determined automatically. Default is None.
        """
        
        # Arguments
        
        self.padder = padder
        self.max_length = max_length
        
        # Estimated parameters

        self.alphabet = None
        self.char_to_index = {}
        self.index_to_char = {}

    def fit(self, data):
        """
        Fits the OneHotEncoder to the given data.

        Parameters:
        - data (list): The list of sequences to fit the encoder on.

        Returns:
        - self: The fitted OneHotEncoder object.
        """
        
        # get max length of sequence

        if self.max_length is None:
            self.max_length = max(len(seq) for seq in data)

        unique_chars = set(''.join(data))
        self.alphabet = sorted(unique_chars)
        
        self.char_to_index = {char: index for index, char in enumerate(self.alphabet)}
        self.index_to_char = {index: char for char, index in self.char_to_index.items()}

        # add padding character to alphabet if not already present

        if self.padder not in self.alphabet:
            self.alphabet.append(self.padder)
            self.char_to_index[self.padder] = len(self.alphabet) - 1
            self.index_to_char[len(self.alphabet) - 1] = self.padder

        return self
    
    def transform(self, data):
        """
        Transforms the given data into one-hot encoded matrices.

        Parameters:
        - data (list): The list of sequences to transform.

        Returns:
        - encoded_data (numpy.ndarray): The one-hot encoded matrices.
        """
        
        # Trim sequences to max_length
        data = [seq[:self.max_length] for seq in data]
        
        # Pad sequences with the padding character
        data = [seq.ljust(self.max_length, self.padder) for seq in data]
        
        # Encode the data to one hot encoded matrices
        encoded_data = np.zeros((len(data), self.max_length, len(self.alphabet)))
        for i, seq in enumerate(data):
            for j, char in enumerate(seq):
                if char in self.char_to_index:
                    index = self.char_to_index[char]
                    encoded_data[i, j, index] = 1

        return encoded_data
    
    def fit_transform(self, data):
        """
        Fits the OneHotEncoder to the given data and transforms it into one-hot encoded matrices.

        Parameters:
        - data (list): The list of sequences to fit and transform.

        Returns:
        - encoded_data (numpy.ndarray): The one-hot encoded matrices.
        """
        return self.fit(data).transform(data)
    
    def inverse_transform(self, data):
        """
        Decodes the one-hot encoded matrices back into sequences.

        Parameters:
        - data (numpy.ndarray): The one-hot encoded matrices.

        Returns:
        - decoded_data (list): The decoded sequences.
        """

        decoded_data = []
        for sample in data:
            decoded_seq = ""
            for char_one_hot in sample:
                char_index = np.argmax(char_one_hot)
                if char_index in self.index_to_char:
                    decoded_seq += self.index_to_char[char_index]
            decoded_data.append(decoded_seq.rstrip(self.padder))
        return decoded_data

            
# test the class

if __name__ == '__main__':
    from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder

    # create a dataset
    data = ['abc', 'def', 'ghi', 'jkl', 'yz']
    
   
    encoder = OneHotEncoder(padder='_')
    encoded = encoder.fit_transform(data)
    decoded = encoder.inverse_transform(encoded)

    print("Alphabet:\n", encoder.alphabet)
    print()
    print(" Encoder Encoded:\n", encoded)
    print()
    print("Encoder Decoded:\n", decoded)

    # compare with sklearn

    sklearn_encoder = SklearnOneHotEncoder(sparse=False, handle_unknown='ignore')
    sklearn_encoded = sklearn_encoder.fit_transform(np.array(data).reshape(-1, 1))
    sklearn_decoded = sklearn_encoder.inverse_transform(sklearn_encoded)

    print()
    print("Sklearn Encoder Encoded:\n", sklearn_encoded)
    print()
    print("Sklearn Encoder Decoded:\n", sklearn_decoded)
    print()
    print("Sklearn Encoder Alphabet:\n", sklearn_encoder.categories_)
    print()
        