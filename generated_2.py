import torch

# Example data
X = torch.tensor([10, 20, 30]) # indices of vertices
Y = torch.tensor([[10, 20, 10], [40, 30, 20], [30, 10, 20], [30, 10, 20], [30, 50, 60]]) # faces
Y = Y.flatten()  # flatten faces to 1D
print("X:\n", X)
print("Y:\n", Y)
#Y = torch.tensor([10, 20, 10, 40, 30, 20, 30, 10, 20, 30, 10, 20, 30, 50, 60]) # faces, flattened

# Create Z of shape (X.size(0), 5)
Z = torch.zeros((X.size(0), 5), dtype=torch.long)

# Create a comparison tensor of shape (X.size(0), Y.size(0))
# where element [i, j] is True if X[i] == Y[j]
mask = X.unsqueeze(1) == Y.unsqueeze(0)  # shape: (len(X), len(Y))

indices = torch.arange(Y.size(0)) + 1
matches = torch.where(mask, indices, torch.tensor(0))

#Z = torch.stack([row[row != 0] for row in matches])
mask = matches != 0
non_zero_values = matches[mask].reshape(matches.size(0), -1)
non_zero_values -= 1  # convert back to original indices

print(non_zero_values)

print("matches:\n", matches)

# Get indices where matches occur
# For each row in matches, get the column indices where True appears
Z = torch.zeros((X.size(0), 5), dtype=torch.long)
for i in range(X.size(0)):
    Z[i] = matches[i].nonzero(as_tuple=True)[0]

print("Z:\n", Z)

