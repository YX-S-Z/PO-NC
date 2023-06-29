# MNIST
python validate_NC.py --gpu_id 0 --dataset mnist --batch_size 2048 --load_path /home/simon.zhai/PO-NC/model_weights/mnist-uniform-0/ &
python validate_NC.py --gpu_id 0 --dataset mnist --batch_size 2048 --load_path /home/simon.zhai/PO-NC/model_weights/mnist-linear-0/ &
python validate_NC.py --gpu_id 0 --dataset mnist --batch_size 2048 --load_path /home/simon.zhai/PO-NC/model_weights/mnist-quadratic-0/ &
python validate_NC.py --gpu_id 0 --dataset mnist --batch_size 2048 --load_path /home/simon.zhai/PO-NC/model_weights/mnist-cubic-0/ &
python validate_NC.py --gpu_id 0 --dataset mnist --batch_size 2048 --load_path /home/simon.zhai/PO-NC/model_weights/mnist-exp-0/ &
python validate_NC.py --gpu_id 0 --dataset mnist --batch_size 2048 --load_path /home/simon.zhai/PO-NC/model_weights/mnist-sqrt-0/ &
python validate_NC.py --gpu_id 0 --dataset mnist --batch_size 2048 --load_path /home/simon.zhai/PO-NC/model_weights/mnist-log-0/ &

# CIFAR10
python validate_NC.py --gpu_id 1 --dataset cifar10 --batch_size 2048 --load_path /home/simon.zhai/PO-NC/model_weights/cifar10-uniform-0/ &
python validate_NC.py --gpu_id 1 --dataset cifar10 --batch_size 2048 --load_path /home/simon.zhai/PO-NC/model_weights/cifar10-linear-0/ &
python validate_NC.py --gpu_id 1 --dataset cifar10 --batch_size 2048 --load_path /home/simon.zhai/PO-NC/model_weights/cifar10-quadratic-0/ &
python validate_NC.py --gpu_id 1 --dataset cifar10 --batch_size 2048 --load_path /home/simon.zhai/PO-NC/model_weights/cifar10-cubic-0/ &
python validate_NC.py --gpu_id 1 --dataset cifar10 --batch_size 2048 --load_path /home/simon.zhai/PO-NC/model_weights/cifar10-exp-0/ &
python validate_NC.py --gpu_id 1 --dataset cifar10 --batch_size 2048 --load_path /home/simon.zhai/PO-NC/model_weights/cifar10-sqrt-0/ &
python validate_NC.py --gpu_id 1 --dataset cifar10 --batch_size 2048 --load_path /home/simon.zhai/PO-NC/model_weights/cifar10-log-0/ &