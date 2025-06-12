import os
from Backbone.AE_CNN import Ae

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
from torch.utils.data import DataLoader, TensorDataset
import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
from GCD import GCD
from HSI_data.HSI import get_hsi_datasets
from HSI_data.datasets import Datasets
from HSI_data.get_data import Load_my_Dataset
from HSI_data.get_datasets import get_class_splits
from HSI_data.transform import get_transform
from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
from config import exp_root
from model import DINOHead, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, \
    get_params_groups
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime




def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def worker_init_fn(worker_id):
    np.random.seed(42 + worker_id)
    random.seed(42 + worker_id)
    torch.manual_seed(42 + worker_id)


def train(student, train_loader, test_loader, unlabelled_train_loader, args):
    params_groups = get_params_groups(student)
    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 1e-3,
    )

    cluster_criterion = DistillLoss(
        args.warmup_teacher_temp_epochs,
        args.epochs,
        args.n_views,
        args.warmup_teacher_temp,
        args.teacher_temp,
    )

    # innitialize SummaryWriter
    log_dir = os.path.join("runs", f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_experiment")
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        cls_loss_record = AverageMeter()
        cluster_loss_record = AverageMeter()
        cluster_loss_o_record = AverageMeter()
        sup_con_loss_record = AverageMeter()
        contrastive_loss_record = AverageMeter()
        loss_basis_record = AverageMeter()
        res_error_record = AverageMeter()
        balance_record = AverageMeter()

        student.train()
        for batch_idx, batch in enumerate(train_loader):
            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            class_labels = class_labels.cuda(non_blocking=True, device=device)
            mask_lab = mask_lab.cuda(non_blocking=True, device=device).bool()
            images = torch.cat(images[0:2], dim=0).cuda(non_blocking=True, device=device)

            # with torch.cuda.amp.autocast(fp16_scaler is not None):
            student_proj, student_out, res, x_rescon, h = student(images)

            teacher_out = student_out.detach()

            # clustering supervised
            sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
            sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0).long()
            cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

            # clustering unsupervised
            cluster_loss_o = cluster_criterion(student_out, teacher_out, epoch)
            avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
            me_max_loss = - torch.sum(torch.log(avg_probs ** (-avg_probs))) + math.log(float(len(avg_probs)))
            balance_loss = args.memax_weight * me_max_loss
            cluster_loss = balance_loss + cluster_loss_o

            # basis loss
            basis = student.projector.last_layer.weight
            basis_sum = torch.sum(torch.square(basis), dim=-1)

            basis_loss = torch.square(torch.mm(basis, basis.T))

            # basis_loss.fill_diagonal_(0)
            for i in range(0, basis.shape[0], 5):
                basis_loss[i:i + 5, i:i + 5] = 0

            loss_basis = torch.sum(basis_loss)

            res_error = torch.mean(res)

            # unsupervised representation learning
            contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj, device=device)
            contrastive_loss = nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

            # supervised representation learning
            student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
            student_proj = nn.functional.normalize(student_proj, dim=-1)
            sup_con_labels = class_labels[mask_lab]
            sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels, device=device)


            # total loss
            loss = 0
            loss += (1 - args.sup_weight) * (cluster_loss) + args.sup_weight * cls_loss
            loss += (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss
            loss += loss_basis
            loss += res_error

            loss_record.update(loss.item(), class_labels.size(0))
            cls_loss_record.update(cls_loss.item(), class_labels.size(0))
            balance_record.update(balance_loss.item(), class_labels.size(0))
            cluster_loss_o_record.update(cluster_loss_o.item(), class_labels.size(0))
            cluster_loss_record.update(cluster_loss.item(), class_labels.size(0))
            sup_con_loss_record.update(sup_con_loss.item(), class_labels.size(0))
            contrastive_loss_record.update(contrastive_loss.item(), class_labels.size(0))
            loss_basis_record.update(loss_basis.item(), class_labels.size(0))
            res_error_record.update(res_error.item(), class_labels.size(0))

            optimizer.zero_grad()
            if fp16_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            if batch_idx % args.print_freq == 0:
                pstr = (
                    f'cls_loss: {cls_loss.item():.4f} '
                    f'ban_loss: {balance_loss.item():.4f} '
                    f'cluster_loss: {cluster_loss.item():.4f} '
                    f'cluster_loss_o: {cluster_loss_o.item():.4f} '
                    f'sup_con_loss: {sup_con_loss.item():.4f} '
                    f'contrastive_loss: {contrastive_loss.item():.4f} '
                    f'loss_basis: {loss_basis.item():.4f} '
                )
                args.logger.info(f'Epoch: [{epoch}][{batch_idx}/{len(train_loader)}]\t loss {loss.item():.5f}\t {pstr}')

        writer.add_scalar('Loss/Total', loss_record.avg, epoch)
        writer.add_scalar('Loss/Cls_Loss', cls_loss_record.avg, epoch)
        writer.add_scalar('Loss/ban_loss', balance_record.avg, epoch)
        writer.add_scalar('Loss/Cluster_Loss', cluster_loss_record.avg, epoch)
        writer.add_scalar('Loss/Cluster_Loss_o', cluster_loss_o_record.avg, epoch)

        writer.add_scalar('Loss/Sup_Con_Loss', sup_con_loss_record.avg, epoch)
        writer.add_scalar('Loss/Contrastive_Loss', contrastive_loss_record.avg, epoch)
        writer.add_scalar('Loss/Loss_Basis', loss_basis_record.avg, epoch)
        writer.add_scalar('Loss/Res_Error', res_error_record.avg, epoch)

        args.logger.info(f'Train Epoch: {epoch} Avg Loss: {loss_record.avg:.4f} ')

        args.logger.info('Testing on unlabelled examples in the training data...')
        all_acc, old_acc, new_acc = test(student, unlabelled_train_loader, epoch=epoch,
                                         save_name='Train ACC Unlabelled', args=args)
        all_acc_test, old_acc_test, new_acc_test = test(student, test_loader, epoch=epoch, save_name='Test ACC',
                                                        args=args)

        args.logger.info(f'Train Accuracies: All {all_acc:.4f} | Old {old_acc:.4f} | New {new_acc:.4f}')
        args.logger.info(f'Test Accuracies: All {all_acc_test:.4f} | Old {old_acc_test:.4f} | New {new_acc_test:.4f}')

        # 将准确率记录到 TensorBoard
        writer.add_scalar('Accuracy/Train_All', all_acc, epoch)
        writer.add_scalar('Accuracy/Train_Old', old_acc, epoch)
        writer.add_scalar('Accuracy/Train_New', new_acc, epoch)
        writer.add_scalar('Accuracy/Test_All', all_acc_test, epoch)
        writer.add_scalar('Accuracy/Test_Old', old_acc_test, epoch)
        writer.add_scalar('Accuracy/Test_New', new_acc_test, epoch)

        exp_lr_scheduler.step()

        save_dict = {
            'model': student.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
        }

        torch.save(save_dict, args.model_path)
        args.logger.info(f"model saved to {args.model_path}.")
    # 关闭 SummaryWriter
    writer.close()


def test(model, test_loader, epoch, save_name, args):
    model.eval()

    preds, targets = [], []
    mask = np.array([])
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True, device=device)
        with torch.no_grad():
            _, logits, res, x_rescon, h = model(images)
            logits_show = logits.cpu().numpy()
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask,
                             np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    classes, counts = np.unique(preds, return_counts=True)
    args.logger.info(f"Prediction distribution: {classes}, {counts}")
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    return all_acc, old_acc, new_acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='Trento',
                        help='options: PaviaU, Trento, SA')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='hyperspectral')
    parser.add_argument('--sup_weight', type=float, default=0.40)
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--save_path', default='./weight/model.pth', type=str)

    # parser.add_argument('--memax_weight', type=float, default=5)
    parser.add_argument('--memax_weight', type=float, default=60)
    parser.add_argument('--warmup_teacher_temp', default=0.08, type=float,
                        help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.02, type=float,
                        help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int,
                        help='Number of warmup epochs for the teacher temperature.')

    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default="simgcd", type=str)

    parser.add_argument('--nz', default=32, type=int, help="The channel number of latent space")
    parser.add_argument('--seeds', default=42, type=int, help="the random seeds")

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    set_seed(args.seeds)
    device = torch.device('cuda:1')
    # device = torch.device('cpu')
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=['simgcd'])
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')

    torch.backends.cudnn.benchmark = True

    # --------------------
    # DATASETS
    if args.dataset_name == "PaviaU":
        data = Load_my_Dataset("/home/xianlli/dataset/HSI/pavia/PaviaU.mat",
                                  "/home/xianlli/dataset/HSI/pavia/PaviaU_gt.mat", patch_size=args.image_size,
                                  band_number=103, device=device)
        band_number=103
    elif args.dataset_name == "Trento":
        data = Load_my_Dataset("/home/xianlli/dataset/HSI/trento/Trento.mat",
                                  "/home/xianlli/dataset/HSI/trento/Trento_gt.mat", patch_size=args.image_size,
                                  band_number=63, device=device)
        band_number = 63
    elif args.dataset_name == "SA":
        data = Load_my_Dataset("/home/xianlli/dataset/HSI/salinas/Salinas_corrected.mat",
                                  "/home/xianlli/dataset/HSI/salinas/Salinas_gt.mat", patch_size=args.image_size,
                                  band_number=204, device=device)
        band_number = 204

    # ----------------------
    # BASE MODEL
    # ----------------------
    backbone = Ae(in_channels=band_number, n_z=args.nz).to(device)
    for param in backbone.parameters():
        param.requires_grad = True

    args.feat_dim = args.image_size*args.image_size*args.nz
    args.num_mlp_layers = 3
    args.mlp_out_dim = (args.num_labeled_classes + args.num_unlabeled_classes) * 5

    args.logger.info("model build: {}", vars(args))

    args.logger.info('model build')


    index = np.arange(data.train.shape[0])

    Whole_dataset = Datasets(x=data.train, y=data.y, index=index, lo_index=data.index, transform=None)
    train_transform, test_transform = get_transform(transform_type='hyperspectral', mask_ratio=0.10, mask_value=0,
                                                    mask_mode="mix")
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=2)

    train_dataset, test_dataset, unlabelled_train_examples_test, train_dataset_o = get_hsi_datasets(whole_dataset=Whole_dataset,
                                                                                   train_transform=train_transform,
                                                                                   test_transform=test_transform,
                                                                                   train_classes=args.train_classes,
                                                                                   prop_train_labels=args.prop_train_labels,
                                                                                   seed=0)
    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset),
                                                     generator=torch.Generator().manual_seed(42))

    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=256, shuffle=False, sampler=sampler, worker_init_fn=worker_init_fn, drop_last=True, pin_memory=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,worker_init_fn=worker_init_fn, batch_size=256, shuffle=False, pin_memory=False)
    test_loader = DataLoader(test_dataset, num_workers=args.num_workers,worker_init_fn=worker_init_fn, batch_size=256, shuffle=False, pin_memory=False)

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    projector = DINOHead(in_dim=args.feat_dim, bottleneck_dim=256, out_dim=args.mlp_out_dim,
                         nlayers=args.num_mlp_layers)
    model = GCD(backbone=backbone, projector=projector).to(device)

    # model.eval()
    train(model, train_loader, test_loader, test_loader_unlabelled, args)

