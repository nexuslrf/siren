# Enable import from parent package
import sys
import os
import time
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, meta_modules, utils, training, loss_functions, modules
import skimage.io
import torch
from torch.utils.data import DataLoader
import configargparse
from functools import partial

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=1)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-4')
p.add_argument('--num_epochs', type=int, default=10000,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=2000,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=1000,
               help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument('--model_type', type=str, default='sine',
               help='Options currently are "sine" (all sine activations), "relu" (all relu activations,'
                    '"nerf" (relu activations and positional encoding as in NeRF), "rbf" (input rbf layer, rest relu),'
                    'and in the future: "mixed" (first layer sine, other layers tanh)')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--split_mlp', action='store_true')
p.add_argument('--split_train', action='store_true')
p.add_argument('--speed_test', action='store_true')
p.add_argument('--test_dim', type=int, default=512)
p.add_argument('--approx_layers', type=int, default=2)
p.add_argument('--act_scale', type=float, default=1)
p.add_argument('--fusion_operator', type=str, choices=['sum', 'prod'], default='prod')
p.add_argument('--fusion_before_act', action='store_true')
p.add_argument('--image_path', type=str, default='')
opt = p.parse_args()

if opt.split_train:
    assert opt.split_mlp == True

sidelength = 512
image_resolution = (512, 512)
img_src = None
if opt.image_path != "":
    img_src = skimage.io.imread(opt.image_path, as_gray=True)
    image_resolution = img_src.shape
    sidelength = img_src.shape

img_dataset = dataio.Camera(img_src=img_src)
coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=sidelength, compute_diff='all', split_coord=opt.split_train)

dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

# Define the model.
if opt.model_type == 'sine' or opt.model_type == 'relu' or opt.model_type == 'tanh' or opt.model_type == 'selu' or opt.model_type == 'elu'\
        or opt.model_type == 'softplus':
    model = modules.SingleBVPNet(type=opt.model_type, mode='mlp', sidelength=image_resolution, split_mlp=opt.split_mlp, 
        approx_layers=opt.approx_layers, act_scale=opt.act_scale, fusion_operator=opt.fusion_operator, fusion_before_act=opt.fusion_before_act)
elif opt.model_type == 'rbf' or opt.model_type == 'nerf':
    model = modules.SingleBVPNet(type='relu', mode=opt.model_type, sidelength=image_resolution, split_mlp=opt.split_mlp,
        approx_layers=opt.approx_layers, act_scale=opt.act_scale, fusion_operator=opt.fusion_operator, fusion_before_act=opt.fusion_before_act)
else:
    raise NotImplementedError
model.cuda()

root_path = os.path.join(opt.logging_root, opt.experiment_name)

# Define the loss
loss_fn = partial(loss_functions.image_mse, None)
summary_fn = partial(utils.write_image_summary, image_resolution)

if not opt.speed_test:
    training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
                steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
                model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, split_train=opt.split_train)
# # test image
else:
    test_len = 50
    if not opt.split_mlp:
        with torch.no_grad():
            model_input = {'coords': dataio.get_mgrid(opt.test_dim).cuda()}
            t0 = time.time()
            for i in range(test_len):
                model_output = model(model_input)
                f"{model_output['model_out'][...,0]}"
            t1 = time.time()
            print(f"Time consumed: {(t1-t0)/test_len}")
    else:
        with torch.no_grad():
            x = torch.linspace(-1,1,opt.test_dim).unsqueeze(-1).cuda()
            y = torch.linspace(-1,1,opt.test_dim).unsqueeze(-1).cuda()
            t0 = time.time()
            for i in range(test_len):
                x_feat = model.forward_split_channel(x, 0)
                y_feat = model.forward_split_channel(y, 1)
                model_output = model.forward_split_fusion(x_feat.unsqueeze(1) + y_feat.unsqueeze(0))
                f"{model_output[...,0]}"
            t1 = time.time()
            print(f"Time consumed: {(t1-t0)/test_len}")