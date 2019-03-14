from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2

class Net(nn.Module):
    def __init__(self):
         super(Net, self).__init__()
         #Convolutional layer()
         self.con1 = nn.Sequential(
                         nn.Conv2d(
                                 #圖片高度
                                 in_channels = 1,
                                 #提取幾個特徵
                                 out_channels = 16,
                                 #捲積核大小
                                 kernel_size = 5,
                                 #掃描一次跳幾格
                                 stride = 1,
                                 #外圍填充(是'SAME'否'VALID')
                                 #padding =(kernel_size-1)/2
                                 padding = 2
                                 ),
                         #線性整流函數
                         nn.ReLU(),
                         #最大池化層(篩選最大值特徵)
                         nn.MaxPool2d(kernel_size = 2)
                         )
         self.con2 = nn.Sequential(
                         nn.Conv2d(16, 32, 5, 1, 2),
                         nn.ReLU(),
                         nn.MaxPool2d(2)
                         )
         #捨去資料
         #self.conv2_drop = nn.Dropout2d()
         #Fully connected layers
         self.fc1 = nn.Sequential(
                         nn.Linear(32 * 7 * 7, 120),
                         nn.ReLU()
                         )
         self.fc2 = nn.Sequential(
                         nn.Linear(120, 84),
                         nn.ReLU()
                         )
         self.fc3 = nn.Linear(84, 10)
         
    def forward(self, x):
        x = self.con1(x)
        x = self.con2(x)
        #多維度的張量展平成一維
        #x = x.view(x.size(0), -1) #batch(32 * 7 * 7)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
#學習
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        #梯度歸零
        optimizer.zero_grad()
        output , last_layer= model(data)
        #計算誤差
        loss = F.nll_loss(output, target)
        #反向傳播
        loss.backward()
        #單次優化
        optimizer.step()
#評估
def test(args, model, device, test_loader, epoch=10):
#===================================繪製分部圖=========================================
#    import random
#    try: from sklearn.manifold import TSNE; HAS_SK = True
#    except: HAS_SK = False; print('Please install sklearn for layer visualization')
#    k = random.randrange(0,(len(test_loader.dataset.test_labels) / test_loader.batch_size))
#=====================================================================================
#   把module設置為評估模式
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output , last_layer = model(data)
            #所有批量損失
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            #獲取最大對數概率
            pred = output.max(1, keepdim=True)[1]
            #所有正確量
            correct += pred.eq(target.view_as(pred)).sum().item()
#===================================繪製分部圖=========================================
#            if i == k:
#                if HAS_SK:
#                    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
#                    plot_only = 500
#                    low_dim_embs = tsne.fit_transform(last_layer.data.cpu().numpy()[:plot_only, :])
#                    labels = target.data.cpu().numpy()[:plot_only]
#                    plot_with_labels(low_dim_embs, labels)
#=====================================================================================
#   計算損失
    test_loss /= len(test_loader.dataset)
#   顯示訓練過程
    print('Epochs {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
          .format(epoch, test_loss, correct, len(test_loader.dataset),
                  100. * correct / len(test_loader.dataset)))
    return test_loss;
    
def plot_with_labels(lowDWeights, labels):
    from matplotlib import cm
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)
    
def input_image(model, image, device):
#   評估模式
    model.eval()
    k = image.view(1, 1, 28, 28)
    x, y = model(k.to(device) )  
#    print(image)
#    print(k.size())    
#    print(x)
    maxn = x.max(1, keepdim=True)[1]
    print('This image is', int(maxn))
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', type=int, default=640, metavar='N')
    parser.add_argument('--test_batch_size', type=int, default=200, metavar='N')
    parser.add_argument('--epochs', type=int, default=10, metavar='N')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M')
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1, metavar='S')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N')
    #參數
    args = parser.parse_args()
    #為CPU設置種子用於生成隨機數
    torch.manual_seed(args.seed)    
    #使用CPU還是GPU
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    #轉變型態
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device);    
    kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data',                                   #MNIST資料儲存位置
                           train=True,                                  #是要traindata(60000)還是testdata(10000)
                           download=True,                              #是否要下載
                       transform=transforms.Compose([
                           transforms.ToTensor(),                       #將資料改成Tensor型態(將資料壓縮到0,1)
                           transforms.Normalize((0.1307,), (0.3081,))   #正規化
                       ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
            batch_size = args.test_batch_size, shuffle=True, **kwargs)  # 把任意數目的字典數傳入
    test_set = datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
#    設定神經網絡並GPU運算
    model = Net().to(device)
    print(model)
#    讀取訓練過的模塊
#    model = torch.load('CNNnet1.pkl').to(device)   
#    plt.ion()    
#============== Train & Test ==============
#   優化學習
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas = (0.9, 0.99), eps=1e-04, weight_decay=0)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        #如果test_loss<0.04則停止訓練
        if test(args, model, device, test_loader,epoch) < 0.04:
           break
#   存取神經網路
    torch.save(model, 'CNNnet1.pkl')      
#==========================================
#    plt.ioff()
#============== Test image ==============  
##    測試圖
#    img, label = test_set[0]
##    讀圖
#    filename=input("請輸入要測試檔名:")
#    img = cv2.imread(filename)
##    轉向，將(28*28*3)轉成(3*28*28)
#    img = img.transpose(2,0,1)
##    擷取第一層圖片
#    img = img[0].transpose(0, 1)
##    升維
#    img = np.expand_dims(img, axis=0)
#    first_train_img = np.reshape(img, (28, 28))
##    劃出圖片    
#    plt.matshow(first_train_img, cmap = plt.get_cmap('gray'))
#    plt.show()
##    numpy轉成Tensor(張量)
#    img = torch.from_numpy(img).float().to(device)
##    圖片判讀
#    input_image(model, img, device)
#==========================================
if __name__ == '__main__':
    main()


















