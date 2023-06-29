# MNIST
python pretrain_finetune.py --gpu_id 0 --uid 2 --dataset mnist --optimizer Adam --batch_size 2048 --lr 0.001 --ETF_fc --ft_class_num 10 &
python pretrain_finetune.py --gpu_id 0 --uid 2 --dataset mnist --optimizer Adam --batch_size 2048 --lr 0.001 --ETF_fc --ft_class_num 10 --preference_type exp &
python pretrain_finetune.py --gpu_id 0 --uid 2 --dataset mnist --optimizer Adam --batch_size 2048 --lr 0.001 --ETF_fc --ft_class_num 10 --preference_type log &
python pretrain_finetune.py --gpu_id 0 --uid 2 --dataset mnist --optimizer Adam --batch_size 2048 --lr 0.001 --ETF_fc --ft_class_num 5 &
python pretrain_finetune.py --gpu_id 0 --uid 2 --dataset mnist --optimizer Adam --batch_size 2048 --lr 0.001 --ETF_fc --ft_class_num 5 --preference_type exp &
python pretrain_finetune.py --gpu_id 0 --uid 2 --dataset mnist --optimizer Adam --batch_size 2048 --lr 0.001 --ETF_fc --ft_class_num 5 --preference_type log &

# CIFAR10
python pretrain_finetune.py --gpu_id 1 --uid 2 --dataset cifar10 --optimizer Adam --batch_size 2048 --lr 0.001 --ETF_fc --ft_class_num 10 &
python pretrain_finetune.py --gpu_id 1 --uid 2 --dataset cifar10 --optimizer Adam --batch_size 2048 --lr 0.001 --ETF_fc --ft_class_num 10 --preference_type exp &
python pretrain_finetune.py --gpu_id 1 --uid 2 --dataset cifar10 --optimizer Adam --batch_size 2048 --lr 0.001 --ETF_fc --ft_class_num 10 --preference_type log &
python pretrain_finetune.py --gpu_id 1 --uid 2 --dataset cifar10 --optimizer Adam --batch_size 2048 --lr 0.001 --ETF_fc --ft_class_num 5 &
python pretrain_finetune.py --gpu_id 1 --uid 2 --dataset cifar10 --optimizer Adam --batch_size 2048 --lr 0.001 --ETF_fc --ft_class_num 5 --preference_type exp &
python pretrain_finetune.py --gpu_id 1 --uid 2 --dataset cifar10 --optimizer Adam --batch_size 2048 --lr 0.001 --ETF_fc --ft_class_num 5 --preference_type log d&