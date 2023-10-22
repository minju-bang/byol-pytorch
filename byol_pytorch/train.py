import torch
from byol_pytorch import BYOL
from torchvision import models
from torchvision.models import ResNet50_Weights
from torch.utils.tensorboard import SummaryWriter

import datetime

def sample_unlabelled_images():
    return torch.randn(20, 3, 256, 256)

class Trainer():
    def __init__(self, params) -> None:
        self.train_dataloader = params['train_data']
        self.test_dataloader  = params['test_data']
        self.BYOL             = params['byol']
        self.optimizer        = params['optim']

    def train(self, epoch):
        self.BYOL.train()
        epoch_loss = 0
        num_total  = 0
        # for i, data in enumerate(self.train_dataloader):
        images = sample_unlabelled_images() # images, label = data
        loss   = self.BYOL(images)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.BYOL.update_moving_average()
        epoch_loss += loss.item()
        num_total  += images.size(0)

        loss = epoch_loss / num_total
        return loss

    def test(self, epoch):
        self.BYOL.eval()
        epoch_loss = 0
        num_total  = 0
        images = sample_unlabelled_images()
        loss   = self.BYOL(images)      
        epoch_loss += loss.item()
        num_total  += images.size(0)

        loss = epoch_loss / num_total
        return loss  

def main():
    resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    learner = BYOL(
                    resnet,
                    image_size = 256,
                    hidden_layer= 'avgpool')
    opt      = torch.optim.Adam(learner.parameters(), lr=3e-4)
    log_dir  = '.\\logs'
    run_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    pt_dir   = f'.\\experiments\\{run_time}_best.pt'
    # writer  = SummaryWriter(log_dir)
    epochs  = 3
    parameters = {'train_data':None, 
                  'test_data' :None,
                  'byol'      :learner,
                  'optim'     :opt}
    
    trainer   = Trainer(params=parameters)
    best_loss = 10000
    for epoch in range(epochs):
        # train_loss = trainer.train(epoch)
        # writer.add_scalar("Loss/Train", train_loss, epoch)
        # print(f'Train_loss: {train_loss:4f}')

        test_loss = trainer.test(epoch)
        # writer.add_scalar("Loss/Train", test_loss, epoch)
        print(f'Test_loss: {test_loss:4f}')

        if test_loss < best_loss:
            torch.save(trainer.BYOL.state_dict(), pt_dir)

    # writer.close()


main()