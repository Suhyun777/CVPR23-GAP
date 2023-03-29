from collections import OrderedDict
from torchmeta.modules import MetaModule
import torch

def gradient_update_parameters(model, meta_params, loss, params=None, inner_lr=None, first_order=False, GAP=True, approx =False):
    if not isinstance(model, MetaModule):
        raise ValueError('The model must be an instance of `torchmeta.modules.MetaModule`, got `{0}`'.format(type(model)))
    if params is None:
        params = OrderedDict(model.meta_named_parameters())
    grads = torch.autograd.grad(loss, params.values(), create_graph=not first_order)
    updated_params = OrderedDict()
    for (name, param), grad in zip(params.items(), grads):
        if GAP:
            if len(param.shape) == 4:
                shape = grad.shape
                grad_matrix = grad.reshape(grad.size(0), -1)
                if approx:
                    preconditionr = torch.diag(torch.nn.Softplus(beta=2)(meta_params[name]))
                    if grad_matrix.size(0) <= grad_matrix.size(1):
                        grad = preconditionr @ grad_matrix
                    else:
                        grad = grad_matrix @ preconditionr
                else:
                    grad_matrix_clone = grad_matrix.clone().detach()

                    ### Trick for the stability of backpropagation ###
                    u, _, _ = torch.svd(grad_matrix_clone)
                    preconditionr = u @ (torch.diag(torch.nn.Softplus(beta=2)(meta_params[name])) @ u.T)
                    grad = preconditionr @ grad_matrix

                grad = grad.view(shape)
                updated_params[name] = param - inner_lr * grad
            else:
                updated_params[name] = param - inner_lr * grad
        else:
            updated_params[name] = param - inner_lr * grad
    return updated_params