import torch


class Model():
   def __init__(self):
       self.conv1 = torch.nn.Conv2d(in_channels = 32, out_channels=128, kernel_size=3)
       self.conv2 = torch.nn.Conv2d(in_channels = 128, out_channels=128, kernel_size=3)
       self.emb = torch.nn.Embedding(10, 32)
       self.out = torch.nn.Linear(128, 1)
       self.sigmoid = torch.nn.Sigmoid()
       components = [self.conv1, self.emb, self.out]
       def parameters():
           for x in components:
               for p in x.parameters():
                   yield p
       self.optim = torch.optim.Adam(lr=0.0001, params=list(parameters()))
   def forward(self, x, labels=None):
       # x (batch, xdim, ydim)
       embedded = self.emb(x).unsqueeze(1).transpose(1,4).squeeze(4)
       layer1 = self.conv1(embedded)
       layer1 = layer1[:, :, range(0,98, 2)]
       layer1 = layer1[:, :, :, range(0,98, 2)]
       print(layer1.size())
       layer2 = self.conv2(layer1)
       layer2 = layer2[:, :, range(0,48, 2)]
       layer2 = layer2[:, :, :, range(0,48, 2)]
       print(layer2.size())
       layer3 = layer2.max(dim=2)[0].max(dim=2)[0]
       out = self.out(layer3)
       print(out.size())
       squashed = 2*self.sigmoid(out)-0.5
       if labels is not None:
           print(labels)
           print(squashed.view(-1))
#           print(squashed.size(), labels.size())
           loss = (squashed.view(-1)-labels).pow(2)
       else:
           loss = None
       return squashed, loss
   def backward(self, loss):
       self.optim.zero_grad()
       loss.backward()
       self.optim.step()

pic = torch.LongTensor(100, 100)
for i in range(50):
    pic[i] = 1
    pic[50+i] = 2
pic[0] = 0
pic[99] = 0
pic[:,0] = 0
pic[:,99] = 0

model = Model()

print(pic[1:-1,][:,1:-1].float().mean())
print(model.forward(pic.unsqueeze(0)))

import random

for iteration in range(1000):
    targets = torch.FloatTensor([random.random() for _ in range(16)])
    pic = (torch.bernoulli(targets.unsqueeze(1).unsqueeze(1).expand(-1, 100, 100))+1)
    pic[:,0] = 0
    pic[:,99] = 0
    pic[:,:,0] = 0
    pic[:,:,99] = 0
    prediction, loss = model.forward(pic.long(), labels=pic[:, 0:99, 0:99].mean(dim=1).mean(dim=1)-1)
    loss = loss.mean()
    model.backward(loss)
    print(loss)


