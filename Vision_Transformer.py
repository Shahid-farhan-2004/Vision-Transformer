import torch
import torch.nn as nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

train_data=datasets.CIFAR10(root="./data",train=True,transform=transform,download=True)
test_data=datasets.CIFAR(root="./data",train=False,transform=transform,download=True)
train_loader=DataLoader(train_data,batch_size=64,shuffle=True)
test_loader=DataLoader(test_data,batch_size=1000)

class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj=nn.Conv2d(3,128,kernel_size=8,stride=8)
    def forward(self,x):
        x=self.proj(x)
        x=x.flatten(2).transpose(1,2)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn=nn.MultiheadAttention(embed_dim=128,num_heads=4,batch_first=True)
        self.ff=nn.Sequential(
            nn.Linear(128,256),
            nn.GELU(),
            nn.Linear(256,128)
        )
        self.norm1=nn.LayerNorm(128)
        self.norm2=nn.LayerNorm(128)
    def forward(self,x):
        x=x+self.attn(self.norm1(x),self.norm1(x),self.norm1(x))[0]
        x=x+self.ff(self.norm2(x))
        return x

class VIT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed=PatchEmbedding()
        num_patches=(32//8)**2
        self.cls_token=nn.Parameter(torch.randn(1,1,128))
        self.pos_embed=nn.Parameter(torch.randn(1,num_patches,128))
        self.encoder=nn.Sequential(*[TransformerEncoder() for _ in range(6) ])
        self.head=nn.Linear(128,10)
    def forward(self,x):
        x=self.patch_embed(x)
        B,N,_=x.shape
        cls_token=self.cls_token.expand(B,-1,-1)
        x=torch.cat((cls_token,x),dim=1)
        x=x+self.pos_embed
        x=self.encode(x)
        return self.head(x[:,0])

model=VIT()
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameter(),lr=0.001)

model.train()
for epoch in range(5):
    for images,labels in train_loader:
        outputs=model(images)
        loss=criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"loss is {loss.item()}")

correct=0
total=0
model.eval()
with torch.no_grad():
    for images,labels in test_loader:
        outputs=model(images)
        _,predictions=torch.max(outputs,1)
        total+=labels.size(0)
        correct+=(labels==predictions).sum().item()
    print(f"correctness is {((correct/total)*100):.2f}")