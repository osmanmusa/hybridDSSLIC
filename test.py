### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
from collections import OrderedDict
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import time
import torch

# os.environ["CUDA_VISIBLE_DEVICES"]="0" # osman added

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset.num_workers = 0 # osman added. Otherwise opt variable is "lost" after enumerate(dataset)
print('Data loaded with %d test samples' % len(data_loader))

dataset_size = len(data_loader)
print('Test set size = %d' % dataset_size)


model = create_model(opt)
visualizer = Visualizer(opt)
# create website
if opt.name=='hybrid_model' or opt.comp_type == 'jp2':
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch), str(opt.comp_ratio))
else:
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))

webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))


print('Model loaded ...')
print(torch.cuda.device_count())
# test
for i, data in enumerate(dataset):    
    if i >= opt.how_many:
        break
             
    start_time = time.time()

    print(data['image'].shape)
    print(data['label'].shape)
    print(data['ds'].shape)

    # fine_details are DNN-generated fine details
    # synthesized = fine_details + upsampled \in [-1,1]^512x1024
    synthesized, fine_details, compressed, upsampled = model.inference(data['image'], data['label'], data['ds'])

    image = torch.tensor(data['image'], device=torch.device('cuda'))
    residual = image - synthesized

    print("--- %s seconds ---" % (time.time() - start_time))
    visuals = OrderedDict([
                            ('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                            ('coarse_ds', util.tensor2im(compressed.data[0])),
                            ('coarse_us', util.tensor2im(upsampled.data[0])),
                            ('nn_generated_fine_details', util.tensor2im(fine_details.data[0])),
                            ('synthesized_(nn_generated_fine_details+upsampled)', util.tensor2im(synthesized.data[0])),
                            ('ground_truth', util.tensor2im(data['image'][0])),
                            ('residual', util.tensor2im(residual.data[0]))
                            ])
    img_path = data['path']
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)

webpage.save()
