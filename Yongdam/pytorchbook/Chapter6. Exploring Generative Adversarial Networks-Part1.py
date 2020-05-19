import matplotlib.pyplot as plt # matplotlib 라이브러리
import torch # pytorch
import torch.nn as nn # pytorch에서 neural network 관련 라이브러리
import torch.nn.functional as F # activation function 관련 라이브리러
import torch.optim as optim # optimizer 관련 라이브러리
import torchvision.datasets as dataset # dataset 관련 라이브러리
import torchvision.transforms as transforms # 데이터의 크기를 바꿔주는 라이브러리, 예를 들어 image를 1차원 feature vector로 바꿔주는 것들
import torchvision.utils as vutils # vision task에서 사용하는 utility functions.
from torchsummary import summary # 해당 neural network에 대한 정보를 요약해서 보여주는 함수.
from tqdm import tqdm
import numpy as np # 수치 계산 및 벡터 최적화를 위한 numpy


# 확인해보니, 이번 강의자료에서 사용하는 모델은 original GAN이 아니고, DCGAN(Deep Convolutional GAN) 입니다.

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, )), # transform의 대상을 모든 차원(feature)이 mean 0.5, std 0.5가 되게 정규화를 한다.
])

batch_size = 128 # batch size의 크기는 128
z_dim = 100 # gan의 latent feature의 차원은 100

database = dataset.MNIST('mnist', train = True, download = True, transform = transform) # mnist dataset을 불러와서 정규화.
#Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
train_loader = torch.utils.data.DataLoader(
    #dataset.MNIST('mnist', train = True, download = True, transform = transform),
    database,
    batch_size = batch_size,
    shuffle = True
) # Mnist를 불러오는데 batch_size 마다 불러오고 shuffle을 하여 training에 사용할 수 있게 바꿔준다.

# for i, data in enumerate(train_loader): # 원래는 train_loader에서 불러오는 각 데이터를 출력해주지만, 확인후 주석처리.
#      print ("batch id =" + str(i) )
#      print (data[0])
#      print (data[1])


def weights_init(m): # 각 filter에 적용할 weight값들을 초기화 하는 함수.
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)  # 입력되는 layer가 conv_layer인 경우엔 mean을 0, std를 0.02
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02) # 입력되는 layer가 batch_norm_layer인 경우엔 mean을 1, std를 0.02
        nn.init.constant_(m.bias.data, 0) # bias는 0으로 초기

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 만약에 colab device에 GPU가 있다면 GPU를 사용하고 그렇지 않다면 cpu를 사용.

# GAN(Generative Adversairal Network)의 첫번째 부분 Generator Network의 정의
class Generator_model(nn.Module):
    def __init__(self, z_dim):
        super().__init__() # model 초기화
        self.fc = nn.Linear(z_dim, 256 * 7 * 7) # fully-connected layer는 100 X (256x7x7) 로 구성한다. 여기서의 z-dimensional vector는 임의로 생섣된 random noise 이미지를 의미한다.
        self.gen = nn.Sequential( # nn.Sequential은 아래 나오는 nn module들을 순서대로 쌓아서 NN architecture를 구성해주는 역할을 한다.
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # ConvTranspose2d 함수를 이용한다. ConvTranspose2d는 원래 Conv2d layer에서 원래 값을 복원하는 연산을 수행한다.
            # generator는 이와 같은 ConvTranspose 연산을 학습하는 것으로 원래 이미지에 대한 정보를 random noise로 부터 구현하는 parameter를 찾는 것을 목표로 학습하게 된다.
            nn.BatchNorm2d(128), # regularization을 위한 batchnorm
            nn.LeakyReLU(0.01), # activation function인 LeakyReLU. LeakyReLU는 ReLU와 다르게 x가 0이하인 경우엔 0이 아닌 0.01를 곱한 값을 돌려준다. 즉, f = max(0, x)가 아닌, f = max(0.01x , x)가 된다.
            nn.ConvTranspose2d(128, 64, 3, 1, 1), # convtranspose2d에 대한 설명을 추가하자면, 3x3 filter에 stride, padding은 1로 주고, 128 dim vector를 64 dim vector로 변환해 줍니다.
            nn.BatchNorm2d(64), # 이제 64차원이 된 벡터를 batchnorm.
            nn.LeakyReLU(0.01), # 다시 leakyrelu
            nn.ConvTranspose2d(64, 1, 4, 2, 1), # 이제는 64차원 결과를 1차원으로 줄입니다. 모든 convtranspose 연산은 in_channel, out_channel, kernel_size, stride, padding 계산이 맞아야 작동합니다.
            nn.Tanh() # 마지막은 tanh activation function 값을 줍니다. (결국엔 scalar value가 나옴) 여기서 이 최종 1차원값은 이미지의 한 픽셀을 의미합니다.
        )
    def forward(self, input):
        x = self.fc(input) # input vector를 넣고
        x = x.view(-1, 256, 7, 7) # batch_size x 256 x 7 x 7로 변환하여 출력한다.
        return self.gen(x)

generator = Generator_model(z_dim).to(device) # 위에서 정의한 model을 CPU/GPU device에 할당합니다.
generator.apply(weights_init) # weight initialization을 합니다.
print(summary(generator, (100, ))) # generator model에 z를 100으로 한 뒤에 만들어지는 모델을 보여줍니다.

# 이번엔 Adversarial Network에 해당하는 Discriminator Network에 대해 알아봅니다. Discriminator가 진짜 이미지와 생성된 이미지를 구별못하게 되면 학습은 성공입니다.
# 왜냐면 generator network가 random noise로 부터 원래 이미지의 latent feature(z)를 잘 찾아서, 실제와 굉장히 비슷한 이미지를 생성했다는 뜻이기 때문입니다.
# Discriminator는 이 1차원 value를 가지고 다시 128차원 값으로 만들어서 이미지가 진짜인지 아닌지 분류한다.
class Discriminator_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), # 1 -> 32, conv2d로 다시 계산합니다.
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, 3, 2, 1), # 32 -> 64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 128, 3, 2, 1), # 64 -> 128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01)
        )
        self.fc = nn.Linear(2048, 1) # fully-connected layer는 2048로 만듭니다.
    def forward(self, input):
        x = self.disc(input)
        return F.sigmoid(self.fc(x.view(-1, 2048))) # input(1차원 scalar value들)을 넣어서 최종 fc layer를 거쳐 sigmoid를 계산해서 나온 값을 최종 value로 사용합니다.
    # 이 값을 통해 분류문제를 풀고, 0이면 real, 1이면 fake로 분류합니다.

discriminator = Discriminator_model().to(device) # 역시 GPU device에 세팅하고
discriminator.apply(weights_init)
print(summary(discriminator, (1, 28, 28))) # 학습하여 최종 28*28의 mnist 이미지가 되게 보여줍니다.

criterion = nn.BCELoss() # binary cross-entropy function을 분류모델의 loss function으로 사용합니다.
# create a batch (whose size 64) of fixed noise vectors (z_dim=100)
fixed_noise = torch.randn(128, z_dim, device=device) # 128개의 fixed noise vector(generator network의 input)을 만듭니다.

doptimizer = optim.Adam(discriminator.parameters()) # discriminator의 optimizer로 adam을 사용합니다.
goptimizer = optim.Adam(generator.parameters()) # generator의 optimizer로 adam을 사용합니다.
real_label, fake_label = 1, 0 # label을 지정합니다.

image_list = []
g_losses = []
d_losses = []
iterations = 0
num_epochs = 50 # number of epoch

for epoch in range(num_epochs):
    print(f'Epoch : | {epoch + 1:03} / {num_epochs:03} |')
    for i, data in enumerate(train_loader):

        discriminator.zero_grad() # discriminator의 모든 gradient를 0으로 초기화힙니다.

        real_images = data[0].to(device)  # real_images: size = (128,1,28,28)

        size = real_images.size(0)  # size = 128 = batch size
        label = torch.full((size,), real_label, device=device)  # real_label =1
        d_output = discriminator(real_images).view(-1) # batch_size x 출력결과
        derror_real = criterion(d_output, label) # 정의된 loss function을 통해 결과 확인 및 loss 계산.

        derror_real.backward() # back propagation 수행, gradient update.

        noise = torch.randn(size, z_dim, device=device)  # noise shape = (128, 100)
        fake_images = generator(noise)  # fake_images: shape = (128,1,28,28)
        label.fill_(0)  # _: in-place-operation
        d_output = discriminator(fake_images.detach()).view(-1) # random noise를 넣어서 generator network를 통해 생성된 이미지를 분

        derror_fake = criterion(d_output, label)
        derror_fake.backward()

        derror_total = derror_real + derror_fake # real data와 fake data에서 계산 loss를 합쳐서 update.
        doptimizer.step() # optimizer를 통해서 학습. (adam)

        generator.zero_grad() # generator의 모든 gradient를 0으로 초기화.
        # label.fill_(real_images) #_: in-place-operation; the same as label.fill_(1)
        label.fill_(1)  # why is the label for the fake-image is one rather than zero? 우리가 목표로 하는 학습은 generator의 expected loss가 최소가 되는 것이기 때문에 label 1에 대한 loss가 0이 되도록 하기 위해서 fake label을 1로 세팅합니다.
        d_output = discriminator(fake_images).view(-1)
        gerror = criterion(d_output, label)
        gerror.backward()

        goptimizer.step()

        if i % 50 == 0:  # for every 50th i 50번째 마다 training의 과정을 출력합니다.
            print( # 현재 학습중인 DCGAN의 generator와 discriminator의 loss값을 출력합니다.
                f'| {i:03} / {len(train_loader):03} | G Loss: {gerror.item():.3f} | D Loss: {derror_total.item():.3f} |')
            g_losses.append(gerror.item())
            d_losses.append(derror_total.item())

        if (iterations % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(train_loader) - 1)): # 500 iteration 마다 이미지를 생성합니다.
            with torch.no_grad():  # check if the generator has been improved from the same fixed_noise vector
                fake_images = generator(fixed_noise).detach().cpu()
            image_list.append(vutils.make_grid(fake_images, padding=2, normalize=True))

        iterations += 1

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(g_losses,label="Generator")
plt.plot(d_losses,label="Discriminator")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

for image in image_list: # 500 iter마다 생성된 이미지들은 image_list에 저장되어 있고, 이를 출력하여 학습과정중에 생성된 이미지의 변화를 살펴볼 수 있습니다.
    plt.imshow(np.transpose(image,(1,2,0)))
    plt.show()