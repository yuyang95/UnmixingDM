import os
import numpy as np
import diffusionmodel as Model
import argparse
import logging

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import core.logger as Logger
import core.vis as visual
from core.wandb_logger import WandbLogger
from core.loaddata import AbuDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/RS_256_abu_DDPM.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('sampling', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        wandb_logger = WandbLogger(opt)
        val_step = 0
    else:
        wandb_logger = None

    train_path    = ''
    train_set = AbuDataset(image_dir=train_path, augment=False)
    train_loader = DataLoader(train_set, batch_size=8, num_workers=4, shuffle=True)
    
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']
    sample_sum = opt['datasets']['val']['data_len']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                # log                
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                # validation
                if current_step % 25000 == 0:

                    result_path = '{}/{}'.format(opt['path']
                                                 ['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)
                    

                    mat_result_path = '{}/{}'.format(opt['path']
                                                 ['mat_results'], current_epoch)
                    os.makedirs(mat_result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')

                    for idx in range(10):
                        
                        diffusion.sample(continous=False)
                        visuals = diffusion.get_current_visuals(sample=True)
                        sample_img = Metrics.tensor2img(
                            visuals['SAMPLE'])  # uint8
                        rgbsample_img = Metrics.tensor2rgb_band8(
                            visuals['SAMPLE'])  # uint8
                        
                        # generation
                        visual.save_img(
                            rgbsample_img, '{}/{}_{}_sample_preview.png'.format(result_path, current_step, idx))
                        visual.save_mat(
                            sample_img, '{}/{}_{}_sample_abu.mat'.format(mat_result_path, current_step, idx))

                        tb_logger.add_image(
                            'Iter_{}'.format(current_step),
                            np.transpose(rgbsample_img, [2, 0, 1]), idx)

                        if wandb_logger:
                            wandb_logger.log_image(f'validation_{idx}', rgbsample_img)

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')

                # if current_step % opt['train']['save_checkpoint_freq'] == 0:
                if current_step % 25000 == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

        # save model
        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation.')

        result_path = '{}'.format(opt['path']['results'])
        mat_result_path = '{}'.format(opt['path']['mat_results'])

        os.makedirs(result_path, exist_ok=True)
        os.makedirs(mat_result_path, exist_ok=True)

        sample_imgs = []
        for idx in range(40):
            idx += 1
            diffusion.sample(continous=True)
            visuals = diffusion.get_current_visuals(sample=True)

            show_img_mode = 'grid'
            if show_img_mode == 'single':
                # single img series
                sample_img = visuals['SAMPLE']  # uint8
                sample_num = sample_img.shape[0]
                for iter in range(0, sample_num):
                    visual.save_img(
                        visual.tensor2img(sample_img[iter]), '{}/{}_{}_sample_{}.png'.format(result_path, current_step, idx, iter))

            else:
                visual.save_img(
                    visual.tensor2rgb_band8(visuals['SAMPLE'][-1]), '{}/{}_{}_sample_preview.png'.format(result_path, current_step, idx))


                visual.save_mat(
                    visual.tensor2img(visuals['SAMPLE'][-1]), '{}/{}_{}_sample_abu.mat'.format(mat_result_path, current_step, idx))


            sample_imgs.append(visual.tensor2img(visuals['SAMPLE'][-1]))

        if wandb_logger:
            wandb_logger.log_images('eval_images', sample_imgs)
          
    
