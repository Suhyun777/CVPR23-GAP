import os
import argparse
import torch
from train import train
from test import test
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model-Agnostic Meta-Learning (MAML)')

    # Argument for train
    parser.add_argument('--dataset', type=str, default='miniImageNet', help='The dataset to train the model.')

    # Argument for test
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--dataset_for_source', type=str, default='miniImageNet', help='The dataset used to train the model.')
    parser.add_argument('--dataset_for_target', type=str, default='miniImageNet', help='The dataset to test the trained model.')
    parser.add_argument('--flags', type=str, default='last', help='best or last')

    # Argument for both train and test
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--folder', type=str, default='./', help='Path to the folder the data is downloaded to.')
    parser.add_argument('--K_shots_for_support', type=int, default=1, help='The number of support set examples per class (i.e., K in "K-shot").')
    parser.add_argument('--K_shots_for_query', type=int, default=15, help='The number of query set examples per class.')
    parser.add_argument('--N_ways', type=int, default=5, help='The number of classes per task (i.e., N in "N-way").')
    parser.add_argument('--train_num_update_step', type=int, default=5, help='The number of update-step in the inner loop process during training.')
    parser.add_argument('--test_num_update_step', type=int, default=10, help='The number of update-step in the inner loop process during validating/testing.')
    parser.add_argument('--first-order', action='store_true', help='Use the first-order approximation of MAML.')
    parser.add_argument('--inner_lr', type=float, default=0.01, help='The step-size for the gradient step for adaptation.')
    parser.add_argument('--hidden_size', type=int, default=128, help='The number of channels for each convolutional layer.')
    parser.add_argument('--batch_size', type=int, default=4, help='The number of tasks in a mini-batch of tasks.')
    parser.add_argument('--iter', type=int, default=60000, help='The number of iteration for training')
    parser.add_argument('--num-workers', type=int, default=0, help='The number of workers for data loading (default: 1).')
    parser.add_argument('--download', action='store_true', help='Download the dataset in the data folder.')
    parser.add_argument('--use-cuda', action='store_true', help='Use CUDA if available.')
    parser.add_argument('--top_k', type=int, default=5, help='The number of models used in the ensemble (if you want).')
    parser.add_argument('--outer_lr1', type=float, default=1e-4, help='The learning rate for initialization in MAML.')
    parser.add_argument('--outer_lr2', type=float, default=1e-4, help='The learning rate for meta-parameters (i.e., for preconditioner).')
    parser.add_argument('--GAP', action='store_true', help='Use the GAP in MAML.')
    parser.add_argument('--approx', action='store_true', help='Use the approximation of GAP')


    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')

    if args.test:
        test(args)
    else:
        train(args)