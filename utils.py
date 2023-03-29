import torch
import torch.nn.functional as F
from inner_loop_optimizer import gradient_update_parameters
import numpy as np
import os
import logging
from collections import OrderedDict

def get_accuracy(logits, targets):
    _, predictions = torch.max(logits, dim=-1)
    return torch.mean(predictions.eq(targets).float())


def evaluate(model, meta_params, meta_test_dataloader, device, inner_lr, first_order, test_num_update_step, GAP, approx):
    accuracy_list = torch.empty(size=[600])
    accuracy = torch.tensor(0., device=device)
    for batch_idx, batch in enumerate(meta_test_dataloader):
        train_inputs, train_targets = batch['train']
        train_inputs = train_inputs.to(device=device)
        train_targets = train_targets.to(device=device)
        # test_task_num = train_inputs.size(0)
        test_inputs, test_targets = batch['test']
        test_inputs = test_inputs.to(device=device)
        test_targets = test_targets.to(device=device)
        for task_idx, (train_input, train_target, test_input, test_target) in enumerate(zip(train_inputs, train_targets, test_inputs, test_targets)):
            ## 1 update step process
            train_logit = model(train_input)
            inner_loss = F.cross_entropy(train_logit, train_target)
            params = gradient_update_parameters(model=model, meta_params=meta_params, loss=inner_loss, inner_lr=inner_lr, first_order=first_order, GAP=GAP, approx=approx)
            ## 2 ~ k update step process
            for i in range(test_num_update_step - 1):
                train_logit = model(train_input, params)
                inner_loss = F.cross_entropy(train_logit, train_target)
                params = gradient_update_parameters(model=model, params=params, meta_params=meta_params, loss=inner_loss, inner_lr=inner_lr, first_order=first_order, GAP=GAP, approx=approx)
            model.eval()
            test_logit = model(test_input, params=params)
            with torch.no_grad():
                accuracy_list[batch_idx] = get_accuracy(test_logit, test_target)
                accuracy += get_accuracy(test_logit, test_target)
            model.train()
        if batch_idx >= 599:
            break
    accuracy.div_(600)
    return accuracy.item(), 1.96 * accuracy_list.std() / np.sqrt(600)

def make_dir(filename):
    if not os.path.exists(filename):
        os.makedirs(filename)

def get_save_path(dataset, train_ways, train_shots, GAP, approx, **kwargs):
    if GAP:
        save_path = './checkpoints/{0}/4conv_{1}way_{2}shot/GAP/approx={3}/'.format(dataset, train_ways, train_shots, approx)
        make_dir(save_path)
        save_path += 'GAP'
    else:
        save_path = './checkpoints/{0}/4conv_{1}way_{2}shot/MAML/approx={3}/'.format(dataset, train_ways, train_shots, approx)
        make_dir(save_path)
        save_path += 'MAML'
    for item in kwargs.items():
        save_path += '.{0}={1}'.format(item[0], item[1])
    save_path += '.th'
    return save_path

def get_log_path(dataset, train_ways, train_shots, GAP, approx, **kwargs):
    if GAP:
        log_path = './logs/{0}/4conv_{1}way_{2}shot/GAP/approx={3}/'.format(dataset, train_ways, train_shots, approx)
        make_dir(log_path)
        log_path += 'GAP'
    else:
        log_path = './logs/{0}/4conv_{1}way_{2}shot/MAML/'.format(dataset, train_ways, train_shots)
        make_dir(log_path)
        log_path += 'MAML'

    for item in kwargs.items():
        log_path += '.{0}={1}'.format(item[0], item[1])
    log_path += '.log'
    return log_path

def get_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def get_meta_parameters(meta_named_parameters, device):
    M = OrderedDict()
    for name, param in meta_named_parameters:
        if len(param.shape) == 4:
            shape = min(list(param.reshape(param.size(0), -1).shape))
            M[name] = torch.nn.Parameter(0.928 * torch.ones(size=[shape], device=device, requires_grad=True))
    return M

def get_test_log(dataset, train_ways, train_shots, GAP, approx, **kwargs):
    if GAP:
        log_path = './test/{0}/4conv_{1}way_{2}shot/GAP/approx={3}/'.format(dataset, train_ways, train_shots, approx)
        make_dir(log_path)
        log_path += 'GAP'
    else:
        log_path = './test/{0}/4conv_{1}way_{2}shot/MAML/'.format(dataset, train_ways, train_shots)
        make_dir(log_path)
        log_path += 'MAML'

    for item in kwargs.items():
        log_path += '.{0}={1}'.format(item[0], item[1])
    log_path += '.log'
    return log_path
