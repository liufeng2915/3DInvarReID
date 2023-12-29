
import torch
import wandb
import yaml
import os
import random
import hydra
import time

from lib.body_model import BaseModel
from lib.dataset.datamodule import DataModule, DataProcessor

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 


def configure_optimizers(model, opt):

    grouped_parameters = [
        {"params": list(model.parameters()),
        'lr': opt.model.optim.lr, 
        'betas':(0.9,0.999)},
    ]

    optimizer = torch.optim.Adam(grouped_parameters, lr=opt.model.optim.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.trainer.lr_scheduler_stepsize,
                                         gamma=opt.trainer.lr_schedule_decay_rate)
    return optimizer, scheduler

def to_gpu(tensor):

    for key in tensor.keys():
        if torch.is_tensor(tensor[key]):
            tensor[key] = tensor[key].cuda()

    return tensor


def save_checkpoints(model, epoch):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    path = 'checkpoints/'+str(epoch)+'.pth'
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, path)


@hydra.main(config_path="config", config_name="config")
def main(opt):

    start_time = time.time()
    print(opt.pretty())
    with open('.hydra/config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    wandb.init(config=config, 
               project='3DInvarReID_pretrain',
               name=opt.expname, 
               job_type="training", 
               reinit=True)

    ##
    random.seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    ## data
    datamodule = DataModule(opt.datamodule)
    datamodule.setup(stage='fit')
    meta_info = datamodule.meta_info
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    data_processor = DataProcessor(opt.datamodule)

    ## define model
    model = BaseModel(opt=opt.model, 
                    meta_info=meta_info,
                    data_processor=data_processor,
                    )
    # model loading
    checkpoint_path = opt.starting_path
    if (os.path.exists(checkpoint_path) and opt.resume):
        model_dict = torch.load(checkpoint_path)
        model.load_state_dict(model_dict)

    ## optimizer 
    optimizer, scheduler = configure_optimizers(model, opt)

    ## cuda
    if True:
        model = model.cuda()

    ## Training
    print("training...")
    for epoch in range(opt.trainer.max_epochs):
        ##
        for data_index, train_batch in enumerate(train_dataloader):
            
            train_batch = to_gpu(train_batch)
            ## 
            loss = model.training_step(epoch, train_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.trainer.gradient_clip_val)
            optimizer.step()

            ## disp loss
            print("Epoch: [%4d/%5d] Time: %4.1f, Loss: %.4f"%(epoch, data_index, time.time()-start_time, loss))

        ## visualization
        model.eval()
        if ((epoch+1) % opt.trainer.check_val_every_n_epoch) == 0:
            save_checkpoints(model, epoch)
            for data_index, val_batch in enumerate(val_dataloader):
                if (data_index % 5) == 0:
                    val_batch = to_gpu(val_batch)
                    model.validation_step(epoch, val_batch)
        model.train()

        ## learning rate
        scheduler.step()
        wandb.log({'learning_rate': scheduler.get_last_lr()})

if __name__ == '__main__':
    main()