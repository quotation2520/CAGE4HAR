import os
import argparse
import time
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter

from dataset.HAR_dataset import HARDataset
from utils.logger import initialize_logger, record_result
from configs import args, dict_to_markdown
from tqdm import tqdm, trange
from models.CAGE import CAGE

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

def set_seed(seed, use_cuda=True):
    # fix seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)

def get_model(n_feat, n_cls):
    if args.seed:
        set_seed(args.seed, use_cuda)

    if args.lambda_ssl > 0:
        proj_dim = args.proj_dim
    else:
        proj_dim = 0
    
    model = CAGE(n_feat // 2, n_cls, proj_dim).to(device)
    if args.load_model != '':
        pre_enc = {k: v for k, v in torch.load(args.load_model).items() if k.split('.')[0] != 'classifier'}
        model.load_state_dict(pre_enc, strict=False)
    
    optimizer = optim.Adam(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    return model, optimizer, scheduler

def train():
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, drop_last=False,
                              num_workers=args.num_workers, pin_memory=True)
    
    model, optimizer, scheduler = get_model(n_feat, n_cls)

    n_device = n_feat // 6 # (accel, gyro) * (x, y, z)

    criterion_cls = nn.CrossEntropyLoss()
    best_f1 = 0.0
    best_epoch = 0

    # logging
    #loss_csv = open(os.path.join(args.save_folder, 'loss.csv'), 'w+')
    result = open(os.path.join(args.save_folder, 'result'), 'w+')

    print("==> training...")
    for epoch in trange(args.epochs, desc='Training_epoch'):
        model.train()
        total_num, total_loss = 0, 0
        ssl_gt, ssl_pred, cls_gt, cls_pred = [], [], [], []
        for idx, (data, label) in enumerate(train_loader):
            data, label = data.to(device).float(), label.to(device).long()
            
            accel_x = data[:, :3 * n_device, :]
            gyro_x = data[:, 3 * n_device:, :]
            ssl_output, cls_output = model(accel_x, gyro_x)
            ssl_label = torch.arange(ssl_output.size(0)).to(device)

            ssl_loss = (criterion_cls(ssl_output, ssl_label) + criterion_cls(ssl_output.T, ssl_label)) / 2
            cls_loss = criterion_cls(cls_output, label)
            loss = ssl_loss * args.lambda_ssl + cls_loss * args.lambda_cls

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss = total_loss + loss.item() * len(label)
            _, ssl_predicted = torch.max(ssl_output, 1)
            _, cls_predicted = torch.max(cls_output, 1)
            ssl_gt.append(ssl_label)
            ssl_pred.append(ssl_predicted)
            cls_gt.append(label)
            cls_pred.append(cls_predicted)
            total_num = total_num + len(label)
            if idx % 20 == 0:
                print('Epoch: [{}/{}], Batch: [{}] - loss:{:.4f}, temperature:{:.4f}'.format(epoch, args.epochs, idx + 1, loss.item(), float(model.temperature.data)))

        if not args.pretrain:                
            scheduler.step()
        total_loss = total_loss / total_num
        label = torch.cat(cls_gt).cpu().detach().numpy()
        predicted = torch.cat(cls_pred).cpu().detach().numpy()
        acc_train = (predicted == label).sum() * 100.0 / total_num
        f1_train = f1_score(label, predicted, average='weighted')
        label2 = torch.cat(ssl_gt).cpu().detach().numpy()
        predicted2 = torch.cat(ssl_pred).cpu().detach().numpy()
        acc_train2 = (predicted2 == label2).sum() * 100.0 / len(label2)
        logger.info('Epoch: [{}/{}] - loss:{:.4f}, train acc: {:.2f}%, train F1: {:.4f} | ssl acc: {:.2f}%'.format(epoch, args.epochs, total_loss / len(train_set), acc_train, f1_train, acc_train2))

        writer.add_scalar('Train/Accuracy_cls', acc_train, epoch)
        writer.add_scalar('Train/F1_cls', f1_train, epoch)
        writer.add_scalar('Train/Accuracy_ssl', acc_train2, epoch)
        writer.add_scalar('Train/Loss', total_loss, epoch)
        acc_test, f1_test = evaluate(model, val_loader, epoch, is_test=False)
        if args.pretrain and epoch % 50 == 0:
            torch.save(model.state_dict(), os.path.join(args.save_folder, f'epoch{epoch}.pth'))

        if f1_test > best_f1:
            best_f1 = f1_test
            best_acc = acc_test
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.save_folder, 'best.pth'))
            # calculate confusion matrix
            c_mat = confusion_matrix(label, predicted)

    # save final model
    torch.save(model.state_dict(), os.path.join(args.save_folder, 'final.pth'))
    print('Done')
    print('Best performance achieved at epoch {}, best acc: {:.2f}%, best F1: {:.4f}'.format(best_epoch, best_acc, best_f1))
    print(c_mat)
    record_result(result, best_epoch, best_acc, best_f1, c_mat)
    writer.close()

def evaluate(model, eval_loader, epoch, is_test=True, mode='best'):
    # Validation
    if is_test:
        model.load_state_dict(torch.load(os.path.join(args.save_folder, mode + '.pth')), strict=False)
    model.eval()

    criterion_cls = nn.CrossEntropyLoss()
    n_device = n_feat // 6 # (accel, gyro) * (x, y, z)

    total_num, ssl_total_loss, cls_total_loss = 0, 0, 0
    ssl_gt, ssl_pred, cls_gt, cls_pred = [], [], [], []
    with torch.no_grad():
        for idx, (data, label) in enumerate(eval_loader):
            data, label = data.to(device).float(), label.to(device).long()
            
            accel_x = data[:, :3 * n_device, :]
            gyro_x = data[:, 3 * n_device:, :]
            ssl_output, cls_output = model(accel_x, gyro_x)
            ssl_label = torch.arange(ssl_output.size(0)).to(device)

            ssl_loss = (criterion_cls(ssl_output, ssl_label) + criterion_cls(ssl_output.T, ssl_label)) / 2
            cls_loss = criterion_cls(cls_output, label)
            ssl_total_loss = ssl_total_loss + ssl_loss * len(label)
            cls_total_loss = cls_total_loss + cls_loss * len(label)
            
            _, ssl_predicted = torch.max(ssl_output, 1)
            _, cls_predicted = torch.max(cls_output, 1)
            ssl_gt.append(ssl_label)
            ssl_pred.append(ssl_predicted)
            cls_gt.append(label)
            cls_pred.append(cls_predicted)
            total_num = total_num + len(label)

    ssl_total_loss = ssl_total_loss / total_num
    cls_total_loss = cls_total_loss / total_num
    label = torch.cat(cls_gt).cpu().detach().numpy()
    predicted = torch.cat(cls_pred).cpu().detach().numpy()
    acc_test = (predicted == label).sum() * 100.0 / total_num
    f1_test = f1_score(label, predicted, average='weighted')
    label2 = torch.cat(ssl_gt).cpu().detach().numpy()
    predicted2 = torch.cat(ssl_pred).cpu().detach().numpy()
    acc_test2 = (predicted2 == label2).sum() * 100.0 / len(label2)

    if is_test:
        print('=> test acc: {:.2f}%, test F1: {:.4f} / ssl acc: {:.2f}%'.format(acc_test, f1_test, acc_test2))
        logger.info('=> test acc: {:.2f}%, test F1: {:.4f} / ssl acc: {:.2f}%'.format(acc_test, f1_test, acc_test2))
        c_mat = confusion_matrix(label, predicted)
        result = open(os.path.join(args.save_folder, 'result'), 'a+')
        record_result(result, epoch, acc_test, f1_test, c_mat)

    else:
        logger.info('=> val acc (cls): {:.2f}%, val F1 (cls): {:.4f} / val acc (ssl): {:.2f}%'.format(acc_test, f1_test, acc_test2))
        logger.info('=> cls_loss: {:.4f} / ssl_loss: {:.4f}'.format(cls_total_loss, ssl_total_loss))
        # save loss
        writer.add_scalar('Validation/Accuracy_cls', acc_test, epoch)
        writer.add_scalar('Validation/F1_cls', f1_test, epoch)
        writer.add_scalar('Validation/Accuracy_ssl', acc_test2, epoch)
        writer.add_scalar('Validation/Loss_cls', cls_total_loss, epoch)
        writer.add_scalar('Validation/Loss_ssl', ssl_total_loss, epoch)

    return acc_test, f1_test
            
if __name__ == "__main__":
    print(dict_to_markdown(vars(args)))

    # get set
    train_set = HARDataset(dataset=args.dataset, split='train', window_width=args.window_width, clean=args.no_clean, include_null=args.no_null, use_portion=args.train_portion)
    val_set = HARDataset(dataset=args.dataset, split='val', window_width=args.window_width, clean=args.no_clean, include_null=args.no_null)
    test_set = HARDataset(dataset=args.dataset, split='test', window_width=args.window_width, clean=args.no_clean, include_null=args.no_null)
    if args.normalize:
        train_set.normalize(train_set.mean, train_set.std)
        val_set.normalize(train_set.mean, train_set.std)
        test_set.normalize(train_set.mean, train_set.std)

    n_feat = train_set.feat_dim
    n_cls = train_set.n_actions

    args.save_folder = os.path.join(args.model_path, args.dataset, 'CAGE', args.trial)
    #if not args.no_clean:
    #    args.save_folder = os.path.join(args.model_path, args.dataset + '_Xclean', 'CAGE', args.trial)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    # logging   
    log_dir = os.path.join(args.save_folder, 'train.log')
    logger = initialize_logger(log_dir)
    writer = SummaryWriter(args.save_folder + '/run')

    #train
    train()

    # test
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, drop_last=False,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, drop_last=False,
                              num_workers=args.num_workers, pin_memory=True)
    model, _, _ = get_model(n_feat, n_cls)
    evaluate(model, test_loader, -1, mode='best')
    evaluate(model, test_loader, -2, mode='final')
    evaluate(model, val_loader, -3, mode='final')
