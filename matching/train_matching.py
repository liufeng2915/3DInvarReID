
import torch
import wandb
import yaml
import os
import random
import hydra
import time
import copy
from lib.matching_model import MatchingModel
from lib.dataset.MatchingDataset import *

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 


def configure_optimizers(model, opt):

    grouped_parameters = [
        {"params": filter(lambda p: p.requires_grad, model.parameters()),
        'lr': opt.model.optim.lr_matching,
        'betas':(0.9,0.999)},
    ]

    optimizer = torch.optim.Adam(grouped_parameters, lr=opt.model.optim.lr_matching)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.trainer.lr_scheduler_stepsize_matching,
                                         gamma=opt.trainer.lr_schedule_decay_rate)
    return optimizer, scheduler

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
               project='3DInvarReID_matching',
               name=opt.expname, 
               job_type="training", 
               reinit=True)

    ##
    random.seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    ## define model
    data_processor = DataProcessor(opt.datamodule)
    model = MatchingModel(opt=opt.model,
                          opt_render=opt.render,
                    data_processor=data_processor,
                    )
    # pretrained 3D shape and texture models loading
    checkpoint_path = 'checkpoints/step1_1999.pth'
    model_dict = model.state_dict()
    pretrained_dict = torch.load(checkpoint_path)
    pretrained_dict = pretrained_dict['model_state_dict']
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and not k.startswith("texture_network")}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    for param in model.naked_shape_network.parameters():
        param.requires_grad = False
    for param in model.clothed_shape_network.parameters():
        param.requires_grad = False
    for param in model.deformer.parameters():
        param.requires_grad = False
    for param in model.generator.parameters():
        param.requires_grad = False
    for param in model.smpl_server.parameters():
        param.requires_grad = False

    ## data
    datamodule = DataModule(opt.datamodule)
    datamodule.setup()
    syn_train_dataloader, real_train_dataloader = datamodule.train_dataloader()
    opt.datamodule.num_classes = real_train_dataloader.dataset.num_classes
    gallery_dataloader, query_dataloader = datamodule.val_dataloader()

    ## optimizer 
    optimizer, scheduler = configure_optimizers(model, opt)

    ## cuda
    if True:
        model = model.cuda()

    ## Training
    print("training...")
    for epoch in range(opt.trainer.max_epochs_matching):

        ## matching valuation
        if ((epoch) % 1) == 0:
            model.eval()
            model.matching_val(gallery_dataloader, query_dataloader)
            model.train()

        ## training
        dataloader_iterator = iter(syn_train_dataloader)
        for data_index, train_real_batch in enumerate(real_train_dataloader):
            try:
                train_syn_batch = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(syn_train_dataloader)
                train_syn_batch = next(dataloader_iterator)
            train_batch = data_processor.combine_batch(train_syn_batch, train_real_batch)
            ## visualization
            if ((data_index) % 50) == 0:
                test_batch = copy.deepcopy(train_batch)
                model.visualization(epoch, test_batch)

            ## 
            loss = model.training_step(epoch, train_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ## disp loss
            print("Epoch: [%4d/%5d] Time: %4.1f, Loss: %.4f"%(epoch, data_index, time.time()-start_time, loss))


        ## save checkpoint
        if ((epoch+1) % 1) == 0:
            save_checkpoints(model, epoch)

        ## learning rate
        scheduler.step()
        wandb.log({'learning_rate': scheduler.get_last_lr()})

if __name__ == '__main__':
    main()