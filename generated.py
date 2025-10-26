import torch

tensor = torch.tensor([[0, 1, 2, 3], [3, 4, 5, 5], [2, 3, 6, 6], [0, 0, 0, 0], [4, 4, 4, 4]])
max_val = tensor.max().item()
num_rows = tensor.shape[0]

# Create a comparison tensor: shape (max_val + 1, num_rows, num_cols)
values = torch.arange(max_val + 1).unsqueeze(1).unsqueeze(2)  # (max_val+1, 1, 1)
mask = (tensor.unsqueeze(0) == values)  # (max_val+1, num_rows, num_cols)

# Check which rows contain each value
row_contains_value = mask.any(dim=2)  # (max_val+1, num_rows)

# Convert to indices with sorting trick
# For each value, get the row indices where it appears
k = num_rows  # max possible length
result = torch.full((max_val + 1, k), -1, dtype=torch.long)

# Use topk or argsort to get indices
row_indices = torch.arange(num_rows).unsqueeze(0).expand(max_val + 1, -1)
# Mask out non-matching rows by setting them to a large value
row_indices_masked = torch.where(row_contains_value, row_indices, -1)

# Sort to bring valid indices to the front
sorted_indices = row_indices_masked.sort(dim=1)[0]

# Only keep valid indices (< num_rows)
result = torch.where(sorted_indices < num_rows, sorted_indices, torch.tensor(-1))

print(result)