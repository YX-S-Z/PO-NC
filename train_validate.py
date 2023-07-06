import sys

import torch
import pickle

import models
from models.res_adapt import ResNet18_adapt
from utils import *
from args import parse_train_args, parse_eval_args
from datasets import make_dataset
from pretrain_finetune import train
import scipy.linalg as scilin
from validate_NC import FCFeatures, compute_Sigma_W, compute_Sigma_B, compute_info, compute_ETF, compute_W_H_relation, compute_Wh_b_relation

def main():

    # Training part
    args = parse_train_args()

    set_seed(manualSeed = args.seed)

    if args.optimizer == 'LBFGS':
        sys.exit('Support for training with 1st order methods!')

    device = torch.device("cuda:"+str(args.gpu_id) if torch.cuda.is_available() else "cpu")
    args.device = device

    trainloader, _, num_classes = make_dataset(args.dataset, args.data_dir, args.batch_size, args.sample_size, SOTA=args.SOTA)
    
    if args.model == "MLP":
        model = models.__dict__[args.model](hidden = args.width, depth = args.depth, fc_bias=args.bias, num_classes=num_classes).to(device)
    elif args.model == "ResNet18_adapt":
        model = ResNet18_adapt(width = args.width, num_classes=num_classes, fc_bias=args.bias).to(device)
    else:
        model = models.__dict__[args.model](num_classes=num_classes, fc_bias=args.bias, ETF_fc=args.ETF_fc, fixdim=args.fixdim, SOTA=args.SOTA).to(device)

    train(args, model, trainloader)

    # Validation part
    # SZ: July 5, Editing here, testing whether we can remove this
    # args = parse_eval_args()

    trainloader, testloader, num_classes = make_dataset(args.dataset, args.data_dir, args.batch_size, args.sample_size)

    fc_features = FCFeatures()
    model.fc.register_forward_pre_hook(fc_features)

    info_dict = {
        'collapse_metric': [],
        'ETF_metric': [],
        'WH_relation_metric': [],
        'Wh_b_relation_metric': [],
        'W': [], 
        'b': [],
        'H': [],
        'mu_G_train': [],
        # 'mu_G_test': [],
        'train_acc1': [],
        'train_acc5': [],
        'train_per_class_acc': [],
        'test_acc1': [],
        'test_acc5': [], 
        'test_per_class_acc': []
    }

    logfile = open('%s/train_log.txt' % (args.save_path), 'a')
    for i in range(args.pretrain_epochs + args.finetune_epochs):
        # SZ: edited, only evaluate the saved model
        if (i + 1) % args.save_freq != 1:
            continue
        model.load_state_dict(torch.load(args.save_path + '/epoch_' + str(i + 1).zfill(3) + '.pth'))
        model.eval()

        for n, p in model.named_parameters():
            if 'fc.weight' in n:
                W = p
            if 'fc.bias' in n:
                b = p

        mu_G_train, mu_c_dict_train, train_acc1, train_acc5, train_per_class_acc = compute_info(args, model, fc_features, trainloader, isTrain=True)
        mu_G_test, mu_c_dict_test, test_acc1, test_acc5, test_per_class_acc = compute_info(args, model, fc_features, testloader, isTrain=False)

        Sigma_W = compute_Sigma_W(args, model, fc_features, mu_c_dict_train, trainloader, isTrain=True)
        # Sigma_W_test_norm = compute_Sigma_W(args, model, fc_features, mu_c_dict_train, testloader, isTrain=False)
        Sigma_B = compute_Sigma_B(mu_c_dict_train, mu_G_train)

        collapse_metric = np.trace(Sigma_W @ scilin.pinv(Sigma_B)) / len(mu_c_dict_train)
        ETF_metric = compute_ETF(W)
        WH_relation_metric, H = compute_W_H_relation(W, mu_c_dict_train, mu_G_train)
        if args.bias:
            Wh_b_relation_metric = compute_Wh_b_relation(W, mu_G_train, b)
        else:
            Wh_b_relation_metric = compute_Wh_b_relation(W, mu_G_train, torch.zeros((W.shape[0], )))

        info_dict['collapse_metric'].append(collapse_metric)
        info_dict['ETF_metric'].append(ETF_metric)
        info_dict['WH_relation_metric'].append(WH_relation_metric)
        info_dict['Wh_b_relation_metric'].append(Wh_b_relation_metric)

        info_dict['W'].append((W.detach().cpu().numpy()))
        if args.bias:
            info_dict['b'].append(b.detach().cpu().numpy())
        info_dict['H'].append(H.detach().cpu().numpy())

        info_dict['mu_G_train'].append(mu_G_train.detach().cpu().numpy())
        # info_dict['mu_G_test'].append(mu_G_test.detach().cpu().numpy())

        info_dict['train_acc1'].append(train_acc1)
        info_dict['train_acc5'].append(train_acc5)
        info_dict['train_per_class_acc'].append(train_per_class_acc)
        info_dict['test_acc1'].append(test_acc1)
        info_dict['test_acc5'].append(test_acc5)
        info_dict['test_per_class_acc'].append(test_per_class_acc)

        print_and_save('[epoch: %d] | train top1: %.4f | train top5: %.4f | test top1: %.4f | test top5: %.4f \n train per class acc: %s \n test per class acc: %s ' %
                       (i + 1, train_acc1, train_acc5, test_acc1, test_acc5, ['{:.2f}'.format(c) for c in train_per_class_acc], ['{:.2f}'.format(c) for c in test_per_class_acc]), logfile)

    with open(args.save_path + '/info.pkl', 'wb') as f:
        pickle.dump(info_dict, f)



if __name__ == "__main__":
    main()