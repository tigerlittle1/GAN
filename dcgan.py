import argparse
import os
import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch



#初始權重
def weights_init_normal(m):
    classname = m.__class__.__name__
        
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(
                #一個N維
                m.weight.data,
                # 正態分佈的平均值
                0.0,
                #正態分佈的標準差
                0.02
                )
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(
                #一個N維
                m.bias.data,
                #填充張量的值
                0.0
                )

#生成器
class Generator(nn.Module):
    def __init__(self ,opt):
        super(Generator, self).__init__()
        #??
        self.init_size = opt.img_size // 4
        #??
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128*self.init_size**2))

        self.conv_blocks = nn.Sequential(
            #批量標準化
            nn.BatchNorm2d(
                    #預期的大小輸入
                    128,
                    #為了增加數值計算的穩定性而加到分母裡
                    eps = 1e-05, 
                    #累積移動平均值
                    momentum = 0.1,
                    #該模塊具有可學習的仿射參數
                    affine = True, 
                    #當設置為True，該模塊跟踪運行的均值和方差當設置為False，此模塊不跟踪此類統計信息，並始終在訓練和評估模式下使用批量統計。
                    track_running_stats = True 
                    ),
            #對給定的多通道1D（時間），2D（空間）或3D（體積）數據進行採樣(轉置捲積)
            nn.Upsample(
                    #輸出大小
                    size = None,
                    #圖像高度/寬度/深度的乘數
                    scale_factor = 2,
                    #線性的、雙線性、三線性
                    mode ='nearest', 
                    #當設置為True，則輸入和輸出張量的角點像素對齊，從而保留這些像素的值。當設置為False，這僅當具有效果mode是線性，雙線性或三線性。
                    align_corners = None
                    ),
            #
            nn.Conv2d(
                    #張量（minibatch * in_channels * iH * iW）
                    128, 
                    #過濾器（out_channels * (in_channels/groups) * kH * kW)
                    128,
                    #偏置張量形狀
                    3,
                    #卷積內核的步伐
                    stride=1,
                    #兩側的隱式填充
                    padding=1, 
                    #內核元素之間的間距
                    dilation=1, 
                    #將輸入分組
                    groups=1,
                    ),
            #批量標準化
            nn.BatchNorm2d(128, 0.8),
            #線性整流函數
            nn.LeakyReLU(
                    #控制負斜率的角度
                    negative_slope = 0.2,
                    #選擇就地進行操作
                    inplace=True
                    ),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            #雙曲正切函數
            nn.Tanh()
        )

    def forward(self, z):
        #??
        out = self.l1(z)
        #展平output.view(seq_len, batch, num_directions, hidden_size)
        out = out.view(
                #第一維表示序列長度
                out.shape[0], 
                #第二維表示一批的樣本數
                128,
                #第三個維度的尺寸根據是否為雙向而變化，如果不是雙向，第三個維度等於我們定義的隱藏層大小。
                self.init_size, 
                #隱藏層大小
                self.init_size
                )
        img = self.conv_blocks(out)
        return img
#鑑別器
class Discriminator(nn.Module):
    def __init__(self ,opt):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [   nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                        nn.LeakyReLU(0.2, inplace=True),
                        #捨棄25%特徵點
                        nn.Dropout2d(0.25)]
#            if bn:
#                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        #圖像的高度和寬度
        ds_size = opt.img_size // 2**4
        self.adv_layer = nn.Sequential(
                
                #對傳入數據應用線性轉換
                nn.Linear(
                        #每個輸入樣本的大小
                        128*ds_size**2,
                        #每個輸出樣本的大小
                        1,
                        #如果設置為False，則圖層不會學習附加偏差
                        bias = True
                        ),
                #S形曲線
                nn.Sigmoid(),
                )

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
    parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
    parser.add_argument('--channels', type=int, default=1, help='number of image channels')
    parser.add_argument('--sample_interval', type=int, default=400, help='interval between image sampling')
    opt = parser.parse_args()
    print(opt) 
    
    #建造資料夾
    os.makedirs('images2', exist_ok=True)
    
    #
    cuda = True if torch.cuda.is_available() else False
    
    device = torch.device("cuda" if cuda else "cpu")
    
    
    #BCELoss，自動編碼器中的重建誤差
    adversarial_loss = torch.nn.BCELoss().to(device)
    
    #初始化生成器
    #generator = torch.load('Generator.pkl').to(device)
    #print(generator)
    generator = Generator(opt).to(device)
    
    #初始化鑑別器
    #discriminator = torch.load('Discriminator.pkl').to(device)
    discriminator = Discriminator(opt).to(device)
    
    #啟用CUDA
#    if cuda:
#        generator.cuda()
#        discriminator.cuda()
#        adversarial_loss.cuda()
    
    #初始化權重
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    #配置數據加載
    os.makedirs('../data/mnist', exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
                #MNIST資料儲存位置
                '../data/mnist',
                #是要traindata(60000)還是testdata(10000)
                train=True,
                #數據下載
                download=True,
                transform=transforms.Compose([
                           #將輸入圖像的大小調整為給定大小
                           transforms.Resize(opt.img_size),
                           #將資料改成Tensor型態(將資料壓縮到0,1)
                           transforms.ToTensor(),
                           #正規化
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
                #批量大小
                batch_size = opt.batch_size,
                num_workers = 4,
                #設置為True在每次都重新調整數據
                shuffle=True
        )
    
    #優化器
    #Adam(Adaptive Moment Estimation)相當於 RMSprop + Momentum
    #Adadelta 和 RMSprop 一樣存儲了過去梯度的平方 vt 的指數衰減平均值 ，也像 momentum 一樣保持了過去梯度 mt 的指數衰減平均值
    #生成器優化
    optimizer_G = torch.optim.Adam(
            #可迭代的參數，用於優化或決定參數組
            generator.parameters(), 
            #學習率
            lr=opt.lr,
            #用於計算梯度及其平方的運行平均值的係數
            betas=(opt.b1, opt.b2),
            #添加到分母中以增加數值穩定性
            eps=1e-08, 
            #權重衰減
            weight_decay=0,
            #是否使用該算法AMSGrad，在凸度設置中收斂到最優解。
            amsgrad=True
            )
    #鑑別器優化
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    #訓練
    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            #對抗的基本事實
            valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
    
            #配置輸入
            real_imgs = Variable(imgs.type(Tensor))
    
            # -----------------
            #  訓練生成器
            # -----------------
            
            #歸零
            optimizer_G.zero_grad()
    
            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
    
            #生成一批圖像
            gen_imgs = generator(z)
    
            #計算生成器能力
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            #反向傳播
            g_loss.backward()
            #單次優化
            optimizer_G.step()
    
            # ---------------------
            #  訓練鑑別器
            # ---------------------
            #歸零
            optimizer_D.zero_grad()
    
            #計算鑑別器能力
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            #反向傳播
            d_loss.backward()
            #單次優化
            optimizer_D.step()
            print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item()))
            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                #save image
                save_image(gen_imgs.data[:25], 'images/%d.png' % batches_done, nrow=5, normalize=True)
    #save model
    torch.save(generator, 'Generator2.pkl')  
    torch.save(discriminator, 'Discriminator2.pkl')

if __name__ == '__main__':
    main()
