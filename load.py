import numpy as np

# Load the saved numpy array
embeddings = np.loadtxt('embeddings.txt')

# Check the shape of the embeddings
print(f"Shape of embeddings: {embeddings.shape}")
