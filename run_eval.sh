# MNIST
python validate_NC.py --gpu_id 0 --dataset mnist --batch_size 2048 --load_path /home/simon.zhai/PO-NC/model_weights/5-mnist-uniform-0/ &
python validate_NC.py --gpu_id 0 --dataset mnist --batch_size 2048 --load_path /home/simon.zhai/PO-NC/model_weights/5-mnist-exp-0/ &
python validate_NC.py --gpu_id 0 --dataset mnist --batch_size 2048 --load_path /home/simon.zhai/PO-NC/model_weights/5-mnist-log-0/ &
python validate_NC.py --gpu_id 0 --dataset mnist --batch_size 2048 --load_path /home/simon.zhai/PO-NC/model_weights/10-mnist-uniform-0/ &
python validate_NC.py --gpu_id 0 --dataset mnist --batch_size 2048 --load_path /home/simon.zhai/PO-NC/model_weights/10-mnist-exp-0/ &
python validate_NC.py --gpu_id 0 --dataset mnist --batch_size 2048 --load_path /home/simon.zhai/PO-NC/model_weights/10-mnist-log-0/ &


# CIFAR10
python validate_NC.py --gpu_id 1 --dataset cifar10 --batch_size 2048 --load_path /home/simon.zhai/PO-NC/model_weights/5-cifar10-uniform-0/ &
python validate_NC.py --gpu_id 1 --dataset cifar10 --batch_size 2048 --load_path /home/simon.zhai/PO-NC/model_weights/5-cifar10-exp-0/ &
python validate_NC.py --gpu_id 1 --dataset cifar10 --batch_size 2048 --load_path /home/simon.zhai/PO-NC/model_weights/5-cifar10-log-0/ &
python validate_NC.py --gpu_id 1 --dataset cifar10 --batch_size 2048 --load_path /home/simon.zhai/PO-NC/model_weights/10-cifar10-uniform-0/ &
python validate_NC.py --gpu_id 1 --dataset cifar10 --batch_size 2048 --load_path /home/simon.zhai/PO-NC/model_weights/10-cifar10-exp-0/ &
python validate_NC.py --gpu_id 1 --dataset cifar10 --batch_size 2048 --load_path /home/simon.zhai/PO-NC/model_weights/10-cifar10-log-0/ &