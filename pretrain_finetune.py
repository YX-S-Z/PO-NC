import sys

import torch

import models
from models.res_adapt import ResNet18_adapt
from utils import *
from args import parse_train_args
from datasets import make_dataset

def loss_compute(args, model, criterion, outputs, targets):
    # SZ: Jun 27 writing here, trying to add weights toe the model.
    # Initialize the cross entropy loss
    loss = criterion(outputs[0], targets)

    # Now decide whether to add weight decay on last weights and last features
    if args.sep_decay:
        # Find features and weights
        features = outputs[1]
        w = model.fc.weight
        b = model.fc.bias
        lamb = args.weight_decay / 2
        lamb_feature = args.feature_decay_rate / 2
        loss += lamb * (torch.sum(w ** 2) + torch.sum(b ** 2)) + lamb_feature * torch.sum(features ** 2)

    return loss

def trainer(args, model, trainloader, epoch_id, criterion, optimizer, scheduler, logfile):

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    print_and_save('\nTraining Epoch: [%d | %d] LR: %f' % (epoch_id + 1, args.pretrain_epochs + args.finetune_epochs, scheduler.get_last_lr()[-1]), logfile)
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        inputs, targets = inputs.to(args.device), targets.to(args.device)
        model.train()
        outputs = model(inputs)
        
        if args.sep_decay:
            loss = loss_compute(args, model, criterion, outputs, targets)
        else:
            loss = criterion(outputs[0], targets)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        model.eval()
        outputs = model(inputs)
        prec1, prec5 = compute_accuracy(outputs[0].detach().data, targets.detach().data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        if batch_idx % 10 == 0:
            print_and_save('[epoch: %d] (%d/%d) | Loss: %.4f | top1: %.4f | top5: %.4f ' %
                           (epoch_id + 1, batch_idx + 1, len(trainloader), losses.avg, top1.avg, top5.avg), logfile)

    scheduler.step()

def get_preferemce_weight(args):
    if args.dataset == "mnist" or "cifar10":
        if args.preference_type == "linear":
            # return a linear weight of [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            return torch.tensor(range(10)) + 1
        elif args.preference_type == "quadratic":
            # return a quadratic weight of [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
            return (torch.tensor(range(10))+1)**2
        elif args.preference_type == "cubic":
            # return a quadratic weight of [1, 8, 27, 64, 125, 216, 343, 512, 729, 1000]
            return (torch.tensor(range(10))+1)**2
        elif args.preference_type == "exp":
            # return a exponential weight of [exp(1), exp(2), exp(3), exp(4), exp(5), exp(6), exp(7), exp(8), exp(9), exp(10)]
            return torch.exp(torch.tensor(range(10)))
        elif args.preference_type == "sqrt":
            # return a exponential weight of [sqrt(1), sqrt(2), sqrt(3), sqrt(4), sqrt(5), sqrt(6), sqrt(7), sqrt(8), sqrt(9), sqrt(10)]
            return torch.sqrt(torch.tensor(range(10))+1)
        elif args.preference_type == "log":
            # return a exponential weight of [log(1), log(2), log(3), log(4), log(5), log(6), log(7), log(8), log(9), log(10)] + 1
            return torch.log(torch.tensor(range(10))+1)+1
        else:
            return None
    else:
        sys.exit("Not implemented yet!")

def train(args, model, trainloader):

    criterion = make_criterion(args)
    optimizer = make_optimizer(args, model)
    scheduler = make_scheduler(args, optimizer)

    logfile = open('%s/train_log.txt' % (args.save_path), 'w')

    print_and_save('# of model parameters: ' + str(count_network_parameters(model)), logfile)
    print_and_save('--------------------- Start Pretraining -------------------------------', logfile)
    for epoch_id in range(args.pretrain_epochs):

        trainer(args, model, trainloader, epoch_id, criterion, optimizer, scheduler, logfile)
        torch.save(model.state_dict(), args.save_path + "/epoch_" + str(epoch_id + 1).zfill(3) + ".pth")

    print_and_save('--------------------- Start Finetuning -------------------------------', logfile)
    
    # Set the new critierion for fine-tuning
    criterion = make_criterion(args, preference_weight = get_preferemce_weight(args))
    for epoch_id in range(args.finetune_epochs):

        trainer(args, model, trainloader, epoch_id + args.pretrain_epochs, criterion, optimizer, scheduler, logfile)
        torch.save(model.state_dict(), args.save_path + "/epoch_" + str(epoch_id + 1).zfill(3) + ".pth")
    

    logfile.close()


def main():
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


if __name__ == "__main__":
    main()
