#!/bin/bash

# ft_ratio=("0.1" "0.5" "0.9")
ft_ratio=("1.0")
pt_class_num=("3" "5" "7")
ft_class_num=("2" "4" "6" "8")

for r in "${ft_ratio[@]}"; do
    for pt in "${pt_class_num[@]}"; do
        for ft in "${ft_class_num[@]}"; do
            # python train_validate.py --gpu_id 0 --force --uid pt_test --dataset mnist --optimizer Adam --batch_size 512 --lr 0.001 --ft_ratio $r --pt_class_num $pt --ft_class_num $ft &
            # python train_validate.py --gpu_id 1 --force --uid pt_test --dataset mnist --optimizer Adam --batch_size 512 --lr 0.001 --ft_ratio $r --pt_class_num $pt --ft_class_num $ft &
            python train_validate.py --gpu_id 0 --force --uid pt_test --dataset cifar10 --optimizer Adam --batch_size 512 --lr 0.001 --ft_ratio $r --pt_class_num $pt --ft_class_num $ft &
            python train_validate.py --gpu_id 1 --force --uid pt_test --dataset cifar10 --optimizer Adam --batch_size 512 --lr 0.001 --ft_ratio $r --pt_class_num $pt --ft_class_num $ft &
        done
    done
done

# Test

# python train_validate.py --gpu_id 0 --force --uid pt_test --dataset mnist --optimizer Adam --batch_size 1024 --lr 0.001 --ft_ratio 1.0 --pt_class_num 7 --preference_type exp --ft_class_num 10 &
# python train_validate.py --gpu_id 1 --force --uid pt_test --dataset mnist --optimizer Adam --batch_size 1024 --lr 0.001 --ft_ratio 1.0 --pt_class_num 7 --ft_class_num 5 &
# python train_validate.py --gpu_id 0 --force --uid pt_test --dataset cifar10 --optimizer Adam --batch_size 1024 --lr 0.001 --ft_ratio 1.0 --pt_class_num 7 --preference_type exp --ft_class_num 10 &
# python train_validate.py --gpu_id 1 --force --uid pt_test --dataset cifar10 --optimizer Adam --batch_size 1024 --lr 0.001 --ft_ratio 1.0 --pt_class_num 7 --ft_class_num 5 &

# python train_validate.py --gpu_id 0 --force --uid pt_test --dataset mnist --optimizer Adam --batch_size 1024 --lr 0.001 --ft_ratio 1.0 --pt_class_num 3 --preference_type exp --ft_class_num 10 &
# python train_validate.py --gpu_id 1 --force --uid pt_test --dataset mnist --optimizer Adam --batch_size 1024 --lr 0.001 --ft_ratio 1.0 --pt_class_num 3 --ft_class_num 5 &
# python train_validate.py --gpu_id 0 --force --uid pt_test --dataset cifar10 --optimizer Adam --batch_size 1024 --lr 0.001 --ft_ratio 1.0 --pt_class_num 3 --preference_type exp --ft_class_num 10 &
# python train_validate.py --gpu_id 1 --force --uid pt_test --dataset cifar10 --optimizer Adam --batch_size 1024 --lr 0.001 --ft_ratio 1.0 --pt_class_num 3 --ft_class_num 5 &


# # CIFAR10
# python pretrain_finetune.py --gpu_id 0 --uid 4 --dataset cifar10 --optimizer Adam --batch_size 1024 --lr 0.001 --ft_ratio 0.01 --ft_class_num 10 &
# python pretrain_finetune.py --gpu_id 0 --uid 4 --dataset cifar10 --optimizer Adam --batch_size 1024 --lr 0.001 --ft_ratio 0.01 --ft_class_num 5 &
# python pretrain_finetune.py --gpu_id 0 --uid 4 --dataset cifar10 --optimizer Adam --batch_size 1024 --lr 0.001 --ft_ratio 0.2 --ft_class_num 10 &
# python pretrain_finetune.py --gpu_id 0 --uid 4 --dataset cifar10 --optimizer Adam --batch_size 1024 --lr 0.001 --ft_ratio 0.2 --ft_class_num 5 &

# # MNIST
# python pretrain_finetune.py --gpu_id 1 --uid 5 --dataset mnist --optimizer Adam --batch_size 1024 --lr 0.001 --ETF_fc --ft_ratio 0.01 --ft_class_num 10 &
# python pretrain_finetune.py --gpu_id 1 --uid 5 --dataset mnist --optimizer Adam --batch_size 1024 --lr 0.001 --ETF_fc --ft_ratio 0.01 --ft_class_num 5 &
# python pretrain_finetune.py --gpu_id 1 --uid 5 --dataset mnist --optimizer Adam --batch_size 1024 --lr 0.001 --ETF_fc --ft_ratio 0.2 --ft_class_num 10 &
# python pretrain_finetune.py --gpu_id 1 --uid 5 --dataset mnist --optimizer Adam --batch_size 1024 --lr 0.001 --ETF_fc --ft_ratio 0.2 --ft_class_num 5 &

# # CIFAR10
# python pretrain_finetune.py --gpu_id 1 --uid 5 --dataset cifar10 --optimizer Adam --batch_size 1024 --lr 0.001 --ETF_fc --ft_ratio 0.01 --ft_class_num 10 &
# python pretrain_finetune.py --gpu_id 1 --uid 5 --dataset cifar10 --optimizer Adam --batch_size 1024 --lr 0.001 --ETF_fc --ft_ratio 0.01 --ft_class_num 5 &
# python pretrain_finetune.py --gpu_id 1 --uid 5 --dataset cifar10 --optimizer Adam --batch_size 1024 --lr 0.001 --ETF_fc --ft_ratio 0.2 --ft_class_num 10 &
# python pretrain_finetune.py --gpu_id 1 --uid 5 --dataset cifar10 --optimizer Adam --batch_size 1024 --lr 0.001 --ETF_fc --ft_ratio 0.2 --ft_class_num 5 &


# python pretrain_finetune.py --gpu_id 0 --uid xx --dataset mnist --optimizer Adam --batch_size 2048 --lr 0.001 --ETF_fc --ft_class_num 5