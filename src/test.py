import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
import pdb
# Step 2: Create a sample dataset
# For simplicity, let's create a dataset with 10 samples, each with a single feature
data = torch.arange(10).float().unsqueeze(1)
labels = torch.arange(10).long()

# Create a TensorDataset
dataset = TensorDataset(data, labels)

g = torch.Generator()
g.manual_seed(0)
# Step 3: Initialize a RandomSampler with the dataset
sampler1 = RandomSampler(dataset, replacement=False, generator=g)


data_loader = DataLoader(dataset, sampler=sampler1, batch_size=1)


w = torch.arange(10)
w_data = TensorDataset(w)  
g = torch.Generator()
g.manual_seed(0)
# Step 3: Initialize a RandomSampler with the dataset
sampler2 = RandomSampler(dataset, replacement=False, generator=g)

# Step 4: Create a DataLoader using the RandomSampler

# Step 5: Iterate over the DataLoader to access the randomly sampled batches
w_loader = DataLoader(w, sampler=sampler2, batch_size=1)
w_tensor = []
for wi in w_loader:
    wii = wi[0]
    w_tensor.append(wii)

w_gt = torch.tensor(w_tensor).bool()
print(w_gt) 
pdb.set_trace() 
for batch_data, batch_labels in data_loader:
    print(f"Data: {batch_data}, Labels: {batch_labels}")
    # print(f"Data: {w_loader}, Labels: {batch_labels}")
print("Done")
for wi in w_loader:
    print(f"Data: {wi}")

print(list(sampler1))
print(list(sampler2))