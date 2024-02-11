from transformers import BertTokenizer, BertModel
import torch
import torch
import torch.nn as nn
import collections.abc
import math
from vitEmbed import VITEmbeddings
from bertEmbed import TextEmbeddings
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings


import torch
import torch.nn.functional as F

# Example logits tensor (unnormalized scores)
logits = torch.tensor([[1.0, 2.0, 3.0],
                       [0.5, 2.5, 1.0]])

# Example labels tensor
labels = torch.tensor([1, 2])

# Compute log probabilities
log_probs = F.log_softmax(logits, dim=-1)

# Extract log probability of true labels
log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))

# Print the results
print("Logits:")
print(logits)
print("\nLog Probabilities:")
print(log_probs)
print("\nLog Probabilities of True Labels:")
print(log_probs_labels)


values=torch.tensor([[1,2,3,4,5],
                     [2,3,4,5,6]])
rewards=torch.tensor([[1,1,1,1,1],
                     [1,1,1,1,1]])


gamma=1
lam=0.8
start=0
lastgaelam = 0
advantages_reversed = []
length = int(rewards.size()[-1])

for t in reversed(range(start, length)):
        nextvalues = values[:, t + 1] if t < length - 1 else 0.0
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
        print(lastgaelam)
        advantages_reversed.append(lastgaelam)
print(advantages_reversed)
advantages = torch.stack(advantages_reversed[::-1], dim=1)
print(advantages)
returns = advantages + values[:, start:]
print(advantages.detach())
print(returns)



c_truncated_reward = torch.tensor([0,1,2,3,4,5],dtype=torch.float32)
r_truncated_reward = torch.tensor([1, 2, 3, 4, 5, 6],dtype=torch.float32)

# Calculate the element-wise difference
diff = c_truncated_reward - r_truncated_reward

# Compute the element-wise log-sigmoid
logsigmoid_result = torch.nn.functional.logsigmoid(diff)

# Calculate the mean of the log-sigmoid values
mean_logsigmoid = logsigmoid_result.mean()

print("c_truncated_reward:")
print(c_truncated_reward)

print("\nr_truncated_reward:")
print(r_truncated_reward)

print("\nElement-wise difference (c_truncated_reward - r_truncated_reward):")
print(diff)

print("\nElement-wise log-sigmoid:")
print(logsigmoid_result)

print("\nMean of the log-sigmoid values:")
print(mean_logsigmoid.item())