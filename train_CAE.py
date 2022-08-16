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

    from models.ConvAE import CAE, MLP_Classifier
    model = CAE(n_feat).to(device)
    classifier = MLP_Classifier(n_cls).to(device)
    
    optimizer_model = optim.Adam(params=model.parameters(), lr=args.learning_rate)
    optimizer_classifier = optim.Adam(params=classifier.parameters(), lr=args.learning_rate)
    scheduler_model = optim.lr_scheduler.StepLR(optimizer_model, step_size=25, gamma=0.8)
    scheduler_classifier = optim.lr_scheduler.StepLR(optimizer_classifier, step_size=25, gamma=0.8)
    if args.load_model != '':
        model.load_state_dict(torch.load(args.load_model))
    return [model, classifier], [optimizer_model, optimizer_classifier], [scheduler_model, scheduler_classifier]

def train():
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, drop_last=False,
                              num_workers=args.num_workers, pin_memory=True)
    
    models, optimizers, schedulers = get_model(n_feat, n_cls)

    n_device = n_feat // 6 # (accel, gyro) * (x, y, z)

    criterion_recon = nn.MSELoss()
    criterion_cls = nn.CrossEntropyLoss()
    best_f1 = 0.0
    best_epoch = 0

    # logging
    #loss_csv = open(os.path.join(args.save_folder, 'loss.csv'), 'w+')
    result = open(os.path.join(args.save_folder, 'result'), 'w+')

    print("==> training autoencoder...")
    for epoch in trange(150, desc='Training_epoch'):
        models[0].train()
        total_loss = 0
        total_num = 0

        for idx, (data, _) in enumerate(train_loader):
            data = data.to(device).float()
            output, _ = models[0](data.unsqueeze(1))
            
            loss = criterion_recon(output, data.unsqueeze(1))
            optimizers[0].zero_grad()
            loss.backward()
            optimizers[0].step()

            total_loss = total_loss + loss.item() * len(data)
            total_num = total_num + len(data)
                
        schedulers[0].step()
        total_loss = total_loss / total_num
        logger.info('Epoch: [{}/{}] - loss:{:.4f}'.format(epoch, args.epochs, total_loss / len(train_set)))

        writer.add_scalar('Train/recon_loss', total_loss, epoch)

    print("==> training classifier...")
    models[0].eval()
    for epoch in trange(200, desc='Training_epoch'):
        models[1].train()
        total_loss = 0
        total_num = 0
        label_list = []
        predicted_list = []
        for idx, (data, label) in enumerate(train_loader):
            data, label = data.to(device).float(), label.to(device).long()
            _, hidden = models[0](data.unsqueeze(1))
            output = models[1](hidden)
            loss = criterion_cls(output, label)
            optimizers[1].zero_grad()
            loss.backward()
            optimizers[1].step()

            total_loss = total_loss + loss.item() * len(label)
            _, predicted = torch.max(output, 1)
            label_list.append(label)
            predicted_list.append(predicted)
            total_num = total_num + len(label)
                
        schedulers[1].step()
        total_loss = total_loss / total_num
        label = torch.cat(label_list).cpu().detach().numpy()
        predicted = torch.cat(predicted_list).cpu().detach().numpy()
        acc_train = (predicted == label).sum() * 100.0 / total_num
        f1_train = f1_score(label, predicted, average='weighted')
        logger.info('Epoch: [{}/{}] - loss:{:.4f}, train acc: {:.2f}%, train F1: {:.4f}'.format(epoch, args.epochs, total_loss / len(train_set), acc_train, f1_train))

        writer.add_scalar('Train/Accuracy_cls', acc_train, epoch)
        writer.add_scalar('Train/F1_cls', f1_train, epoch)
        writer.add_scalar('Train/cls_loss', total_loss, epoch)
        acc_test, f1_test = evaluate([models[0], models[1]], val_loader, epoch, is_test=False)

        if f1_test > best_f1:
            best_f1 = f1_test
            best_acc = acc_test
            best_epoch = epoch
            torch.save(models[0].state_dict(), os.path.join(args.save_folder, 'model_best.pth'))
            torch.save(models[1].state_dict(), os.path.join(args.save_folder, 'classifier_best.pth'))
            # calculate confusion matrix
            c_mat = confusion_matrix(label, predicted)

    # save final model
    torch.save(models[0].state_dict(), os.path.join(args.save_folder, 'model_final.pth'))
    torch.save(models[1].state_dict(), os.path.join(args.save_folder, 'classifier_final.pth'))
    print('Done')
    print('Best performance achieved at epoch {}, best acc: {:.2f}%, best F1: {:.4f}'.format(best_epoch, best_acc, best_f1))
    print(c_mat)
    record_result(result, best_epoch, best_acc, best_f1, c_mat)
    writer.close()

def evaluate(models, eval_loader, epoch, is_test=True, mode='best'):
    # Validation
    if is_test:
        models[0].load_state_dict(torch.load(os.path.join(args.save_folder, 'model_' + mode + '.pth')), strict=False)
        models[1].load_state_dict(torch.load(os.path.join(args.save_folder, 'classifier_' + mode + '.pth')), strict=False)
    models[0].eval()
    models[1].eval()
    total_loss = 0
    total_num = 0

    criterion_cls = nn.CrossEntropyLoss()

    with torch.no_grad():
        label_list = []
        predicted_list = []
        for idx, (data, label) in enumerate(eval_loader):
            data, label = data.to(device).float(), label.to(device).long()
            bs = len(label)

            _, hidden = models[0](data.unsqueeze(1))
            output = models[1](hidden)
            total_loss = total_loss + criterion_cls(output, label) * bs
            _, predicted = torch.max(output, 1)
            label_list.append(label)
            predicted_list.append(predicted)
            total_num = total_num + bs

    total_loss = total_loss / total_num
    label = torch.cat(label_list).cpu().detach().numpy()
    predicted = torch.cat(predicted_list).cpu().detach().numpy()
    acc_test = (predicted == label).sum() * 100.0 / total_num
    f1_test = f1_score(label, predicted, average='weighted')

    if is_test:
        print('=> test acc: {:.2f}%, test F1: {:.4f}'.format(acc_test, f1_test))
        logger.info('=> test acc: {:.2f}%, test F1: {:.4f}'.format(acc_test, f1_test))
        c_mat = confusion_matrix(label, predicted)
        result = open(os.path.join(args.save_folder, 'result'), 'a+')
        record_result(result, epoch, acc_test, f1_test, c_mat)


    else:
        logger.info('=> val acc: {:.2f}%, val F1: {:.4f}'.format(acc_test, f1_test))
        logger.info('=> cls_loss: {:.4f}'.format(total_loss))
        # save loss
        writer.add_scalar('Validation/Accuracy_cls', acc_test, epoch)
        writer.add_scalar('Validation/F1_cls', f1_test, epoch)
        writer.add_scalar('Validation/Loss_cls', total_loss, epoch)

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

    args.save_folder = os.path.join(args.model_path, args.dataset, 'ConvAE', args.trial)
    #if not args.no_clean:
    #    args.save_folder = os.path.join(args.model_path, args.dataset + '_Xclean', args.model, args.trial)
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
