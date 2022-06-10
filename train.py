### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

# --name cityscapes_model --dataroot ./datasets/cityscapes/ --label_nc 35 --loadSize 1024 --resize_or_crop scale_width --batchSize 2
# --name ADE20K_model --dataroot ./datasets/ADE20K/ --label_nc 151 --loadSize 256 --resize_or_crop resize --batchSize 8

import time
from collections import OrderedDict
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import os
import numpy as np
import torch
from torch.autograd import Variable

os.environ["CUDA_LAUNCH_BLOCKING"]="1"

opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
else:    
    start_epoch, epoch_iter = 1, 0

if opt.debug:   
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10 

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset.num_workers = 0
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)

total_steps = (start_epoch-1) * dataset_size + epoch_iter

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(dataset, start=epoch_iter):        
        iter_start_time = time.time()        
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == 0

        losses, real, label, generated, nn_generated_details, compressed, upsampled = model(Variable(data['label']), Variable(data['image']), Variable(data['ds']), infer=save_fake)
        
        # sum per device losses
        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
        loss_dict = dict(zip(model.module.loss_names, losses))

        # calculate final loss scalar
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        if epoch > opt.niter:
            loss_G = loss_dict['G_GAN'] + loss_dict['G_GAN_Feat'] + loss_dict['G_SSIM']
        else:
            loss_G = loss_dict['G_GAN'] + loss_dict['G_GAN_Feat'] + loss_dict['G_VGG'] + loss_dict['G_DIS'] + loss_dict['G_SSIM']

        ############### Backward Pass ####################
        # update generator (and possibly compressor) weights
        model.module.optimizer_G.zero_grad()
        loss_G.backward()        
        model.module.optimizer_G.step()        

        # update discriminator weights
        model.module.optimizer_D.zero_grad()
        loss_D.backward()
        model.module.optimizer_D.step()
                
        ############## Display results and errors ##########
        ### print out errors
        if total_steps % opt.print_freq == 0:

            # errors = {k: v.data[0] if not isinstance(v, int) else v for k, v in loss_dict.items()}
            errors = {k: v.item() if not (isinstance(v, int) or isinstance(v, float)) else v for k, v in loss_dict.items()}

            #errors['psp_loss'] = psp_train_loss.avg
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)

        ### display output images
        if save_fake:
            #for i in range(opt.batchSize):
            i=0
            visuals = OrderedDict([
                                   ('input_label', util.tensor2label(data['label'][i], opt.label_nc)),
                                   ('fine_image', util.tensor2im(nn_generated_details.data[i])),
                                   ('comp_image', util.tensor2im(compressed.data[i])),
                                   ('up_image', util.tensor2im(upsampled.data[i])),
                                   ('synthesized_image', util.tensor2im(generated.data[i])),
                                   ('real_image', util.tensor2im(data['image'][i]))])
            visualizer.display_current_results(visuals, epoch, epoch_iter)

        ### save latest model        
        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.module.save('latest')            
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
                   
    # end of epoch 
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
        model.module.save('latest')
        model.module.save(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

    ### instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        model.module.update_fixed_params()

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.module.update_learning_rate()

print('Finished training')