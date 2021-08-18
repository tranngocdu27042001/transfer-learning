import torch

from lib import *
from transform import *
from dataset import *
from model_transferlearning import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transfrom_ = transfrom()

train_dataset = Mydataset('../input/100-bird-species/birds/birds.csv',transfrom_,'train')
valid_dataset = Mydataset('../input/100-bird-species/birds/birds.csv',transfrom_,'valid')
test_dataset = Mydataset('../input/100-bird-species/birds/birds.csv',transfrom_,'test')

train_dataloader = DataLoader(train_dataset, batch_size = 512, shuffle = True)
valid_dataloader = DataLoader(valid_dataset, batch_size = 512, shuffle = False)
test_dataloader = DataLoader(test_dataset, batch_size = 256, shuffle = False)


criterion = nn.CrossEntropyLoss()
optimizer = Adam(params=mobilenet_v2.classifier.parameters(),lr= 0.001)
data_size = {'train':train_dataset.__len__(),'valid':valid_dataset.__len__()}
dataloader = {'train':train_dataloader,'valid':valid_dataloader}


def training(model, num_epoch, criterion, optim, data_size, dataloader):
    print('start training')

    loss_list = {'train': [], 'valid': []}
    accuracy_list = {'train': [], 'valid': []}

    for epoch in range(1, num_epoch + 1):

        print('epoch {}/{}'.format(epoch, num_epoch))
        print('-' * 10)

        for phase in ['train', 'valid']:

            mini_batch_loss = 0.0
            mini_batch_accuracy = 0.0

            if phase == 'train':
                model.train()
            else:
                model.eval()

            for inputs, label in tqdm(dataloader[phase]):

                inputs = inputs.to(device)
                label = label.to(device)
                optim.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    out = model(inputs)
                    loss = criterion(out, label)
                    _, pres = torch.max(out, 1)

                    if phase == 'train':
                        loss.backward()
                        optim.step()
                mini_batch_loss += loss.item() * inputs.shape[1]
                mini_batch_accuracy += torch.sum(pres == label.data)

            loss_epoch = mini_batch_loss / data_size[phase]
            accuracy_epoch = mini_batch_accuracy.double() / data_size[phase]

            print('{}: loss = {} ,accuracy = {}'.format(phase, loss_epoch, accuracy_epoch))

            loss_list[phase].append(loss_epoch)
            accuracy_list[phase].append(accuracy_epoch)
        df_loss = pd.DataFrame(loss_list).to_csv('loss_list')
        df_acc = pd.DataFrame(accuracy_list).to_csv('accuracy_list')

        print()
        if epoch % 1 == 0:
            torch.save(model.state_dict(), 'weight.pth')

    return model, loss_list, accuracy_list


if __name__ == '__main__':
    mobilenet_v2.to(device=device)
    modelss, loss_list, acc_list = training(model=mobilenet_v2, num_epoch=10, criterion=criterion, optim=optimizer,
                                            data_size=data_size, dataloader=dataloader)

    #plot loss accuracy

    plt.subplot(1, 2, 1)
    plt.plot(loss_list['train'])
    plt.plot(loss_list['valid'])
    plt.title('loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train', 'valid'])
    plt.subplot(1, 2, 2)
    plt.plot(acc_list['train'])
    plt.plot(acc_list['valid'])
    plt.title('accuracy')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train', 'valid'])
