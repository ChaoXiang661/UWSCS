import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/gencrack_512', #syncrack_dataset
                    help='Name of Experiment')
parser.add_argument('--exp', type=str, default='gencrack_512_1/WS0_01',
                    help='experiment_name')
parser.add_argument('--model', type=str, default='acpa', help='model_name')
parser.add_argument('--noise', type=str, default=True, help='if noise')
parser.add_argument('--noise_degree', type=str, default='noise_erode2*3', help='noise degree')
parser.add_argument('--num_classes', type=int,  default=1, help='output channel of network')
parser.add_argument('--gpu', type=int,  default=3, help='GPU use id')
parser.add_argument('--best_model', type=str, default=True, help='if use best model or not')
parser.add_argument('--epoch_1', type=int, default=30, help='sparse mask train')
parser.add_argument('--epoch_2', type=int, default=40, help='slic')
parser.add_argument('--epoch_3', type=int, default=60, help='crf')
parser.add_argument('--epoch_4', type=int, default=50, help='accurate label train')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.1,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[512, 512],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=33, help='random seed')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=4,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=200,
                    help='labeled data')
parser.add_argument('--total_num', type=int, default=1600,
                    help='total data')
args = parser.parse_args()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
import sys
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import torchvision.transforms.functional as F2
from losses_pytorch.ssim_loss import SSIM
from dataloaders.dataset import *
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume, test_single_volume_ds
from relabel import *

def valid(epoch_num, valloader, model, best_perfor, db_val):
    model.eval()
    metric_list2 = 0.0
    for i_batch, sampled_batch in enumerate(valloader):
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        volume_batch = volume_batch.cuda()
        metric_i = test_single_volume(volume_batch, label_batch, model, args.model)
        metric_list2 += np.array(metric_i)
    metric_list2 = metric_list2 / len(db_val)
    print(metric_list2)
    perfor = np.mean(metric_list2)
    if perfor > best_perfor:
        best_perfor = perfor
        save_mode_path = os.path.join(snapshot_path,
                                      'epoch_{}_dice_{}.pth'.format(
                                          epoch_num, round(best_perfor, 4)))

        save_best = os.path.join(snapshot_path,
                                 '{}_best_epoch_model.pth'.format(args.model))
        torch.save(model.state_dict(), save_mode_path)
        torch.save(model.state_dict(), save_best)
    logging.info('epoch : %d / mean_dice : %f ' % (epoch_num, perfor))
    model.train()
    return best_perfor


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    labeled_slice = args.labeled_num
    labeled_bs = args.labeled_bs
    total_num = args.total_num
    epoch_1, epoch_2, epoch_3, epoch_4 = args.epoch_1 , args.epoch_2 , args.epoch_3, args.epoch_4
    max_epoch = epoch_1 + epoch_2 + epoch_3 + epoch_4
    max_iterations = epoch_1 * (labeled_slice / batch_size) + epoch_2/3 * (labeled_slice / batch_size) \
                     + epoch_2 / 3 *2* (labeled_slice / labeled_bs) + epoch_3 * (total_num / batch_size) \
                     + epoch_4 * (total_num / batch_size)
    print("Total silices is: {}, labeled slices is: {}".format(total_num, labeled_slice))

    model = net_factory(net_type=args.model, in_chns=3, class_num=num_classes)
    ## 阶段训练数据
    db_train1 = DataSet1(base_dir=args.root_path, split="train", num=labeled_slice, noise=args.noise,
                            deg=args.noise_degree,
                            transform=transforms.Compose([RandomGene1(args.patch_size)]))

    db_train2_1 = DataSet2_1(base_dir=args.root_path, split="train", num=labeled_slice, noise=args.noise,
                            deg=args.noise_degree,
                            transform=transforms.Compose([RandomGene2(args.patch_size)]))

    db_train2_2 = DataSet2_2(base_dir=args.root_path, split="train", num=total_num, noise=args.noise,
                            deg=args.noise_degree,
                            transform=transforms.Compose([RandomGene2(args.patch_size)]))

    db_train3 = DataSet3(base_dir=args.root_path, split="train", num=total_num, noise=args.noise,
                            deg=args.noise_degree,
                            transform=transforms.Compose([RandomGene3(args.patch_size)]))
    ## 阶段生成结果
    db_update1_1 = UpDataSet(base_dir=args.root_path, split="train", num=labeled_slice,
                             noise=args.noise, deg=args.noise_degree,
                          transform=transforms.Compose([Gene1(args.patch_size)]), mask='1')

    db_update1_2 = UpDataSet(base_dir=args.root_path, split="train", num=total_num,
                             noise=args.noise, deg=args.noise_degree,
                          transform=transforms.Compose([Gene1(args.patch_size)]), mask='1')

    db_update2 = UpDataSet(base_dir=args.root_path, split="train", num=total_num,
                             noise=args.noise, deg=args.noise_degree,
                          transform=transforms.Compose([Gene1(args.patch_size)]), mask='2')
    ## 验证
    db_val = ValDataSets(base_dir=args.root_path, split="val", deg=args.noise_degree,
                          transform=transforms.Compose([Gene(args.patch_size)]))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_num))

    train1loader = DataLoader(db_train1, batch_size=batch_size, shuffle=True,
                             num_workers=12, pin_memory=True, worker_init_fn=worker_init_fn)

    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)
    train2_1loader = DataLoader(db_train2_1, batch_size=batch_size, shuffle=True, num_workers=12,
                              pin_memory=True, worker_init_fn=worker_init_fn)
    train2_2loader = DataLoader(db_train2_2, batch_sampler=batch_sampler, num_workers=12,
                              pin_memory=True, worker_init_fn=worker_init_fn)
    train3loader = DataLoader(db_train3, batch_size=batch_size, shuffle=True,
                              num_workers=12, pin_memory=True, worker_init_fn=worker_init_fn)
    train_up1_1loader = DataLoader(db_update1_1, batch_size=1, shuffle=False,
                             num_workers=12, pin_memory=True, worker_init_fn=worker_init_fn)
    train_up1_2loader = DataLoader(db_update1_2, batch_size=1, shuffle=False,
                                 num_workers=12, pin_memory=True, worker_init_fn=worker_init_fn)
    train_up2loader = DataLoader(db_update2, batch_size=1, shuffle=False,
                             num_workers=12, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                         momentum=0.9, weight_decay=0.0001)
    ssim_loss = SSIM(window_size=11,size_average=True)
    ce_loss = nn.BCELoss()
    dice_loss = losses.DiceLoss()

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(train1loader)))

    iter_num = 0
    best_perfor = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    m = 0

    for epoch_num in iterator:
        if epoch_num <= epoch_1:
            for i_batch, sampled_batch in enumerate(train1loader):
                volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                # print(volume_batch.shape)
                volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
                outputs = model(volume_batch)
                diceloss = dice_loss(outputs, label_batch.unsqueeze(1))   #losses.dice_loss1
                celoss = ce_loss(outputs, label_batch.unsqueeze(1))
                loss = 0.75 * diceloss + 0.25 * celoss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_ = base_lr * ( 1.0 - iter_num / max_iterations ) ** 3
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
                iter_num = iter_num + 1
                writer.add_scalar('info/lr', lr_, iter_num)
                writer.add_scalar('info/total_loss', loss, iter_num)
                logging.info('iteration : %d / loss : %f / lr : %f' % (iter_num, loss.item(), lr_))
        if epoch_num > epoch_1 and epoch_num < (epoch_1 + epoch_2/2):
            model.eval()
            with torch.no_grad():
                for i_batch, sampled_batch in enumerate(train_up1_1loader):
                    volume_batch, imgpath = sampled_batch['image'], sampled_batch['mask_path']
                    volume_batch = volume_batch.cuda()
                    if args.best_model is True:
                        save_mode_path = os.path.join(snapshot_path,
                                                      '{}_best_epoch_model.pth'.format(args.model))  # _epoch
                        model.load_state_dict(torch.load(save_mode_path))
                    outputs = model(volume_batch)
                    # save
                    tmp = outputs.squeeze(0).cpu().detach().numpy()  # H W, [0,1]
                    tmp[tmp > 0.5] = 1
                    tmp[tmp <= 0.5] = 0
                    out2 = torch.from_numpy(tmp)
                    pred = F2.to_pil_image(out2)
                    pred.save(imgpath[0])
            if epoch_num % 3 ==0 or epoch_num == (epoch_1+1):
                superpixel1(args.root_path, args.noise_degree,labeled_slice)
            model.train()
            for i_batch, sampled_batch in enumerate(train2_1loader):
                volume_batch, label_bath, slic_batch = sampled_batch['image'], sampled_batch['mask1'], \
                sampled_batch['slic']
                volume_batch, label_bath, slic_batch = volume_batch.cuda(), label_bath.cuda(), slic_batch.cuda()
                outputs = model(volume_batch)

                diceloss1 = dice_loss(outputs, label_bath.unsqueeze(1))
                celoss1 = ce_loss(outputs, label_bath.unsqueeze(1))
                loss_label = 0.75 * diceloss1 + 0.25 * celoss1

                diceloss2 = dice_loss(outputs, slic_batch.unsqueeze(1))
                celoss2 = ce_loss(outputs, slic_batch.unsqueeze(1))
                loss_slic = 0.75 * diceloss2 + 0.25 * celoss2

                # loss_a = 0.6 - 0.4 * (epoch_num - epoch_1) / epoch_2 * 2
                # loss = loss_a * loss_label + (1 - loss_a) * loss_slic
                loss = 0.2 * loss_label + 0.8 * loss_slic

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 3
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
                iter_num = iter_num + 1
                writer.add_scalar('info/lr', lr_, iter_num)
                writer.add_scalar('info/total_loss', loss, iter_num)
                logging.info('iteration : %d / loss : %f / lr : %f' % (iter_num, loss.item(), lr_))

        if epoch_num >= (epoch_1 + epoch_2/2) and epoch_num <= (epoch_1 + epoch_2):
            model.eval()
            with torch.no_grad():
                for i_batch, sampled_batch in enumerate(train_up1_2loader):
                    volume_batch, imgpath = sampled_batch['image'], sampled_batch['mask_path']
                    volume_batch = volume_batch.cuda()
                    if args.best_model is True:
                        save_mode_path = os.path.join(snapshot_path,
                                                      '{}_best_epoch_model.pth'.format(args.model))  # _epoch
                        model.load_state_dict(torch.load(save_mode_path))
                    outputs = model(volume_batch)
                    # save
                    tmp = outputs.squeeze(0).cpu().detach().numpy()  # H W, [0,1]
                    tmp[tmp > 0.5] = 1
                    tmp[tmp <= 0.5] = 0
                    out2 = torch.from_numpy(tmp)
                    pred = F2.to_pil_image(out2)
                    pred.save(imgpath[0])
            if (epoch_num == (epoch_1 + epoch_2 / 2) or epoch_num % 3 ==0):# and epoch_num < (epoch_1 + epoch_2):
                superpixel2(args.root_path, args.noise_degree,total_num)
            model.train()
            for i_batch, sampled_batch in enumerate(train2_2loader):
                volume_batch, mask1_bath, slic_batch = sampled_batch['image'], sampled_batch['mask1'], \
                sampled_batch['slic']
                volume_batch, mask1_bath, slic_batch = volume_batch.cuda(), mask1_bath.cuda(), slic_batch.cuda()
                outputs = model(volume_batch)

                diceloss1 = dice_loss(outputs, mask1_bath.unsqueeze(1))
                celoss1 = ce_loss(outputs, mask1_bath.unsqueeze(1))
                loss_mask1 = 0.75 * diceloss1 + 0.25 * celoss1

                diceloss2 = dice_loss(outputs, slic_batch.unsqueeze(1))
                celoss2 = ce_loss(outputs, slic_batch.unsqueeze(1))
                loss_slic = 0.75 * diceloss2 + 0.25 * celoss2

                # loss_a = 0.2 + 0.3 * (epoch_num - epoch_1 - epoch_2/2) / epoch_2 * 2
                # loss = loss_a * loss_mask1 + (1 - loss_a) * loss_slic
                loss = 0.2 * loss_mask1 + 0.8 * loss_slic

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 3
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
                iter_num = iter_num + 1
                writer.add_scalar('info/lr', lr_, iter_num)
                writer.add_scalar('info/total_loss', loss, iter_num)
                logging.info('iteration : %d / loss : %f / lr : %f' % (iter_num, loss.item(), lr_))


        if epoch_num >= (epoch_1 + epoch_2) and epoch_num <= (epoch_1 + epoch_2 + epoch_3):
            model.eval()
            with torch.no_grad():
                for i_batch, sampled_batch in enumerate(train_up2loader):
                    volume_batch, imgpath = sampled_batch['image'], sampled_batch['mask_path']
                    volume_batch = volume_batch.cuda()
                    # if args.best_model is True:
                    #     save_mode_path = os.path.join(snapshot_path,
                    #                                   '{}_best_epoch_model.pth'.format(args.model))
                    #     model.load_state_dict(torch.load(save_mode_path))
                    outputs = model(volume_batch)
                    # save
                    tmp = outputs.squeeze(0).cpu().detach().numpy()  # H W, [0,1]
                    if int(i_batch) < labeled_slice:
                        tmp[tmp > 0.6] = 1
                        tmp[tmp <= 0.6] = 0
                    else:
                        tmp[tmp > 0.6] = 1
                        tmp[tmp <= 0.6] = 0
                    out2 = torch.from_numpy(tmp)
                    pred = F2.to_pil_image(out2)
                    pred.save(imgpath[0])

            relabel(args.root_path, args.noise_degree, total_num)
            model.train()
            for i_batch, sampled_batch in enumerate(train3loader):
                volume_batch, slic_batch, up_bath = sampled_batch['image'], sampled_batch['slic'], sampled_batch['update']
                volume_batch, slic_batch, up_bath = volume_batch.cuda(), slic_batch.cuda(), up_bath.cuda()
                outputs = model(volume_batch)

                diceloss1 = dice_loss(outputs, slic_batch.unsqueeze(1))
                celoss1 = ce_loss(outputs, slic_batch.unsqueeze(1))
                loss_mask2 = 0.75 * diceloss1 + 0.25 * celoss1

                diceloss2 = dice_loss(outputs, up_bath.unsqueeze(1))
                celoss2 = ce_loss(outputs, up_bath.unsqueeze(1))
                loss_up = 0.75 * diceloss2 + 0.25 * celoss2

                loss_a = 0.8 - 0.5 * (epoch_num - (epoch_1 +  epoch_2)) / epoch_3
                loss = loss_a * loss_mask2 + (1 - loss_a) * loss_up

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_ = base_lr * ( 1.0 - iter_num / max_iterations ) ** 3
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
                iter_num = iter_num + 1
                writer.add_scalar('info/lr', lr_, iter_num)
                writer.add_scalar('info/total_loss', loss, iter_num)
                logging.info('iteration : %d / loss : %f / lr : %f' % (iter_num, loss.item(), lr_))

        if epoch_num > (epoch_1 + epoch_2 + epoch_3):
            model.train()
            for i_batch, sampled_batch in enumerate(train3loader):
                volume_batch, up_bath = sampled_batch['image'], sampled_batch['update']
                volume_batch, up_bath = volume_batch.cuda(), up_bath.cuda()
                outputs = model(volume_batch)

                diceloss = dice_loss(outputs, up_bath.unsqueeze(1))
                celoss = ce_loss(outputs, up_bath.unsqueeze(1))
                loss = 0.75 * diceloss + 0.25 * celoss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 3
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
                iter_num = iter_num + 1
                writer.add_scalar('info/lr', lr_, iter_num)
                writer.add_scalar('info/total_loss', loss, iter_num)
                logging.info('iteration : %d / loss : %f / lr : %f' % (iter_num, loss.item(), lr_))
        if epoch_num > 0:
            best_perfor = valid(epoch_num, valloader, model, best_perfor, db_val)
        if epoch_num >= (max_epoch + 1):
            iterator.close()
            break

    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if args.noise:
        snapshot_path = "../model/{}/{}_labeled/{}/{}".format(
        args.exp, args.labeled_num, args.noise_degree, args.model)
    else:
        snapshot_path = "../model/{}/{}_labeled/gt/{}".format(
            args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/weakslic'):
        shutil.rmtree(snapshot_path + '/weakslic')
    shutil.copytree('.', snapshot_path + '/weakslic',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
