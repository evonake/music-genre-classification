import torch

# batch_size, channels, feature (y), time (x)
x = torch.tensor([ [[0, 1],
                    [2, 3]],
                   [[4, 5],
                    [6, 7]] ])
# print(x)

# batch_size, channels, time (x)
x = x.view(2, 2, 2)
print(x, x.shape)
print(x[0, :, None, 0])
print(x[0, None, :, 0])
print(torch.transpose(x, 0, 1), torch.transpose(x, 0, 1).shape)

"""
we want the values in x = k for all channels as input:
  input_size: 2, 2, 1
all values in input would be from one time, but different channels
"""

print('EXPECTED: ')
temp = x.data.cpu().numpy()
expected = [
            [round(x[0, 0, 0].item(), 4), round(x[0, 1, 0].item(), 4)],
            [round(x[1, 0, 0].item(), 4), round(x[1, 1, 0].item(), 4)],
           ]
expected_torch = torch.tensor(expected)
print(expected)
print(expected_torch.shape)

for i in range(x.size()[2]):
  print(x[:,:,i,None])
  print(x[:,:,i,None].size())
