import argparse
import logging
import os
import random
import shutil
import time
from pathlib import Path
from warnings import warn

import math
import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import test  # import test.py to get mAP after each epoch
from models.yolo import Model
from utils.datasets import create_dataloader
from utils.general import (
    torch_distributed_zero_first, labels_to_class_weights, plot_labels, check_anchors, labels_to_image_weights,
    compute_loss, plot_images, fitness, strip_optimizer, plot_results, get_latest_run, check_dataset, check_file,
    check_git_status, check_img_size, increment_dir, print_mutation, plot_evolution, set_logging, init_seeds)
from utils.google_utils import attempt_download
from utils.torch_utils import ModelEMA, select_device, intersect_dicts
from utils.ar1_utils import *
from utils.datasets import externalMemory

logger = logging.getLogger(__name__)


def train(hyp, opt, device, tb_writer=None, wandb=None):
    logger.info(f'Hyperparameters {hyp}')
    log_dir = Path(tb_writer.log_dir) if tb_writer else Path(opt.logdir) / 'evolve'  # logging directory
    wdir = log_dir / 'weights'  # weights dfirectory
    os.makedirs(wdir, exist_ok=True)
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = str(log_dir / 'results.txt')
    epochs, batch_size, total_batch_size, weights, rank = \
        opt.epochs_init, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank

    # Save run settings
    with open(log_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(log_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # Configure
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict
    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)  # check
    train_path = data_dict['train']
    test_path = data_dict['val']
    nc, names = (1, ['item']) if opt.single_cls else (int(data_dict['nc']), data_dict['names'])  # number classes, names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # Model
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(rank):
            attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        if hyp.get('anchors'):
            ckpt['model'].yaml['anchors'] = round(hyp['anchors'])  # force autoanchor
        model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc).to(device)  # create
        exclude = ['anchor'] if opt.cfg or hyp.get('anchors') else []  # exclude keys
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
    else:
        model = Model(opt.cfg, ch=3, nc=nc).to(device)  # create

    # Freeze
    freeze = []  # parameter names to freeze (full or partial)
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False

    if opt.reg_lambda != 0:
        # the regularization is based on Synaptic Intelligence as described in the
        # paper. ewcData is a list of two elements (best parametes, importance)
        # while synData is a dictionary with all the trajectory data needed by SI
        model.ewcData, model.synData = create_syn_data(model)

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay


    optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - hyp['lrf']) + hyp['lrf']  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # Logging
    if wandb and wandb.run is None:
        id = ckpt.get('wandb_id') if 'ckpt' in locals() else None
        wandb_run = wandb.init(config=opt, resume="allow", project="YOLOv5", name=os.path.basename(log_dir), id=id)

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # Results
        if ckpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(ckpt['training_results'])  # write results.txt

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if opt.resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
            shutil.copytree(wdir, wdir.parent / f'weights_backup_epoch{start_epoch - 1}')  # save previous weights
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, state_dict

    # Image sizes
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples





    # Model parameters
    hyp['cls'] *= nc / 80.  # scale coco-tuned hyp['cls'] to current dataset
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)

    all_test_dataloader = create_dataloader(test_path, imgsz_test, total_batch_size, gs, opt,
                                            hyp=hyp, augment=False,
                                            cache=opt.cache_images and not opt.notest,
                                            rect=True,
                                            rank=-1, world_size=opt.world_size,
                                            workers=opt.workers,
                                            n_batch=-1)[0]
    
    
    root = 'G:/projects/core50_350_1f/batches/'
    paths = os.listdir(root)
    train_paths = []
    valid_paths = []

    for p in paths:
        if 'train' in p:
            train_paths.append(root + p)
        elif 'val' in p:
            valid_paths.append(root + p)
        else:
            print(p)

           # external_memory = ext_memory()
    extMem = externalMemory()
    for core_batch in range(11):
        # Trainloader

        if opt.reg_lambda != 0:
            init_batch(model, model.ewcData, model.synData)

        print(f'------------CORE50 itertaion №:{core_batch}------------')

        external_files_path = extMem.file
        if core_batch > 0:
            train_path = [train_paths[core_batch], external_files_path]
        else:
            train_path = train_paths[core_batch]

        extMem.update_memory(train_paths[core_batch], update_iters=10 if core_batch == 0 else 1)

        dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                                hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect,
                                                rank=rank, world_size=opt.world_size, workers=opt.workers,
                                                )


        mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
        nb = len(dataloader)  # number of batches
        assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (
            mlc, nc, opt.data, nc - 1)


        testloader = create_dataloader(valid_paths[core_batch], imgsz_test, total_batch_size, gs, opt,
                                       hyp=hyp, augment=False, cache=opt.cache_images and not opt.notest, rect=True,
                                       rank=-1, world_size=opt.world_size, workers=opt.workers
                                       )[0]  # testloader

        #if not opt.resume:
        labels = np.concatenate(dataset.labels, 0)
        c = torch.IntTensor(labels[:, 0])  # classes
        plot_labels(labels, save_dir=log_dir)
        print(torch.bincount(c))
        if tb_writer:
            # tb_writer.add_hparams(hyp, {})  # causes duplicate https://github.com/ultralytics/yolov5/pull/384
            tb_writer.add_histogram('classes', c, core_batch)

            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)

        # model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
        model.names = names

        # Start training
        t0 = time.time()
        nw = max(round(hyp['warmup_epochs'] * nb), 1e3)  # number of warmup iterations, max(3 epochs, 1k iterations)
        # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
        maps = np.zeros(nc)  # mAP per class
        results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
        scheduler.last_epoch = start_epoch - 1  # do not move
        scaler = amp.GradScaler(enabled=cuda)
        logger.info('Image sizes %g train, %g test\n'
                    'Using %g dataloader workers\nLogging results to %s\n'
                    'Starting training for %g epochs...' % (imgsz, imgsz_test, dataloader.num_workers, log_dir, epochs))
        # update number of epochs to iterative training
        if core_batch != 0:
            epochs = opt.epochs_iter

        # x_train, y_train = dataset.get_all_data()



        for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
            model.train()

            mloss = torch.zeros(4, device=device)  # mean losses
            logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'targets', 'img_size'))

           # x_train_splitted = torch.split(x_train, 4)
           # y_train_splitted = torch.split(y_train, 4)
           # pbar = enumerate(zip(x_train_splitted, y_train_splitted))
            pbar = enumerate(dataloader)
            pbar = tqdm(pbar, total=nb)  # progress bar
            optimizer.zero_grad()
            for i, (imgs, targets, _, _) in pbar:  # batch -------------------------------------------------------------

                # imgs = x_train[i * batch_size:(i + 1) * batch_size]
                # targets = y_train[i * batch_size:(i + 1) * batch_size]
                #
                # # preprocess tensor to proper form
                # # img, label = zip(imgs, targets)  # transposed
                # for i, l in enumerate(targets):
                #     l[:, 0] = i  # add target image index for build_targets()
                # 
                # imgs = torch.stack(imgs)
                # targets = torch.cat(targets)


                ni = i + nb * epoch  # number integrated batches (since train start)
                imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0



                if opt.reg_lambda != 0:
                    pre_update(model, model.synData)

                # Warmup
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                    accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                    for j, x in enumerate(optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(ni, xi,
                                            [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

                # # Multi-scale
                # if opt.multi_scale:
                #     sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                #     sf = sz / max(imgs.shape[2:])  # scale factor
                #     if sf != 1:
                #         ns = [math.ceil(x * sf / gs) * gs for x in
                #               imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                #         imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

                # Forward
                with amp.autocast(enabled=cuda):
                    pred = model(imgs)  # forward
                    loss, loss_items = compute_loss(pred, targets.to(device), model)  # loss scaled by batch_size
                # Backward
                scaler.scale(loss).backward()

                # Optimize
                if ni % accumulate == 0:
                    scaler.step(optimizer)  # optimizer.step

                    if opt.reg_lambda != 0:
                        post_update(model, model.synData)

                    scaler.update()
                    optimizer.zero_grad()
                    # if ema:
                    #     ema.update(model)

                # Print
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)


                # end batch ------------------------------------------------------------------------------------------------

            # Scheduler
            lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
            scheduler.step()

            # mAP
            results, maps, times = test.test(opt.data,
                                             batch_size=total_batch_size,
                                             imgsz=imgsz_test,
                                             # model=ema.ema,
                                             model=model,
                                             single_cls=opt.single_cls,
                                             dataloader=testloader,
                                             save_dir=log_dir,
                                             plots=epoch == 0,  # plot first and last
                                             log_imgs=opt.log_imgs)

                # wandb.log({'per class/AP per class': maps})

            # Write
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
            if len(opt.name) and opt.bucket:
                os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))

            # Log
            tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                    'val/giou_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                    'x/lr0', 'x/lr1', 'x/lr2']  # params
            for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                if tb_writer:
                    tb_writer.add_scalar(tag, x, epoch)  # tensorboard
                if wandb:
                    wandb.log({tag: x})  # W&B

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi

            # Save model
            save = not opt.nosave
            if save:
                with open(results_file, 'r') as f:  # create checkpoint
                    ckpt = {'epoch': epoch,
                            'best_fitness': best_fitness,
                            'training_results': f.read(),
                            # 'model': ema.ema,
                            'model': model,
                            'optimizer': optimizer.state_dict(),
                            'wandb_id': wandb_run.id if wandb else None}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                del ckpt
        # end epoch ----------------------------------------------------------------------------------------------------
        # end training

        #consolidate_weights(model, cur_class)
        if opt.reg_lambda != 0:
            update_ewc_data(model, model.ewcData, model.synData, 0.001, 1)

        if rank in [-1, 0]:
            # Strip optimizers
            n = opt.name if opt.name.isnumeric() else ''
            fresults, flast, fbest = log_dir / f'results{n}.txt', wdir / f'last{n}.pt', wdir / f'best{n}.pt'
            for f1, f2 in zip([wdir / 'last.pt', wdir / 'best.pt', results_file], [flast, fbest, fresults]):
                if os.path.exists(f1):
                    os.rename(f1, f2)  # rename
                    if str(f2).endswith('.pt'):  # is *.pt
                        strip_optimizer(f2)  # strip optimizer
                        os.system('gsutil cp %s gs://%s/weights' % (f2, opt.bucket)) if opt.bucket else None  # upload
            # Finish
            plot_results(save_dir=log_dir)  # save as results.png
            logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))

        dist.destroy_process_group() if rank not in [-1, 0] else None
        torch.cuda.empty_cache()
        results, maps, times = test.test(opt.data,
                                         batch_size=total_batch_size,
                                         imgsz=imgsz_test,
                                         #model=ema.ema,
                                         model=model,
                                         single_cls=opt.single_cls,
                                         dataloader=all_test_dataloader,
                                         save_dir=log_dir,
                                         #plots=epoch == 0 or final_epoch,  # plot first and last
                                         log_imgs=opt.log_imgs,
                                         verbose=True)
        
        #wandb.log({'per class/AP per class All': maps[0]})
        #tb_writer.add_scalar('per class/AP per class All', maps[0])

        # Log
        tags = [  # train loss
                'test/precision', 'test/recall', 'test/mAP_0.5', 'test/mAP_0.5:0.95',
                'test/giou_loss', 'test/obj_loss', 'test/cls_loss']  # params
        for x, tag in zip(list(results), tags):
            if tb_writer:
                tb_writer.add_scalar(tag, x, core_batch)  # tensorboard
            if wandb:
                wandb.log({tag: x})  # W&B

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs_init', type=int, default=20)
    parser.add_argument('--epochs_iter', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=4, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--name', default='', help='renames experiment folder exp{N} to exp{N}_{name} if supplied')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--logdir', type=str, default='runs/', help='logging directory')
    parser.add_argument('--log-imgs', type=int, default=10, help='number of images for W&B logging, max 100')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--reg-lambda', type=float, default=0, help='reg lambda for SI for CL')
    opt = parser.parse_args()

    # Set DDP variables
    opt.total_batch_size = opt.batch_size
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)
    if opt.global_rank in [-1, 0]:
        check_git_status()

    # Resume
    if opt.resume:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        log_dir = Path(ckpt).parent.parent  # runs/exp0
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(log_dir / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))  # replace
        opt.cfg, opt.weights, opt.resume = '', ckpt, True
        logger.info('Resuming training from %s' % ckpt)

    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
        log_dir = increment_dir(Path(opt.logdir) / 'exp', opt.name)  # runs/exp1

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps
        if 'box' not in hyp:
            warn('Compatibility: %s missing "box" which was renamed from "giou" in %s' %
                 (opt.hyp, 'https://github.com/ultralytics/yolov5/pull/1120'))
            hyp['box'] = hyp.pop('giou')

    # Train
    logger.info(opt)

    tb_writer, wandb = None, None  # init loggers
    # Tensorboard
    logger.info(f'Start Tensorboard with "tensorboard --logdir {opt.logdir}", view at http://localhost:6006/')
    tb_writer = SummaryWriter(log_dir=log_dir)  # runs/exp0

    # W&B
    try:
        import wandb

        assert os.environ.get('WANDB_DISABLED') != 'true'
        logger.info("Weights & Biases logging enabled, to disable set os.environ['WANDB_DISABLED'] = 'true'")
    except (ImportError, AssertionError):
            opt.log_imgs = 0
            logger.info("Install Weights & Biases for experiment logging via 'pip install wandb' (recommended)")

    train(hyp, opt, device, tb_writer, wandb)

