from dataset import get_dataset
from torchmeta.utils.data import BatchMetaDataLoader
from backbone import ConvolutionalNeuralNetwork
from transform import get_transform
from utils import *
from inner_loop_optimizer import gradient_update_parameters


def train(args):
    print('Dataset: {0} / GAP:{1} / Approx:{2} / Train Begin'.format(args.dataset, args.GAP, args.approx))

    train_transform = get_transform(train=True)
    test_transform = get_transform(train=False)

    meta_train_dataset, meta_val_dataset, meta_test_dataset = get_dataset(args.dataset, args.folder, args.N_ways, args.K_shots_for_support, args.K_shots_for_query, args.download, train_transform, test_transform)

    meta_train_dataloader = BatchMetaDataLoader(meta_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    meta_val_dataloader = BatchMetaDataLoader(meta_val_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers)
    meta_test_dataloader = BatchMetaDataLoader(meta_test_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers)

    model = ConvolutionalNeuralNetwork(3, args.N_ways, hidden_size=args.hidden_size)
    model.to(device=args.device)

    model.train()
    M = get_meta_parameters(model.meta_named_parameters(), args.device)

    meta_optimizer = torch.optim.Adam([{'params': model.meta_parameters(), 'lr': args.outer_lr1}, {'params': [i for _, i in M.items()], 'lr': args.outer_lr1}])

    log_path = get_log_path(dataset=args.dataset, train_ways=args.N_ways, train_shots=args.K_shots_for_support, GAP=args.GAP, approx=args.approx, outer_lr1=args.outer_lr1, outer_lr2=args.outer_lr2, batch_size=args.batch_size)
    logger = get_logger(log_path)

    outer_loss_list = []
    best_val_acc = 0.0
    for batch_idx, batch in enumerate(meta_train_dataloader):
        meta_optimizer.zero_grad()

        train_inputs, train_targets = batch['train']
        train_inputs = train_inputs.to(device=args.device)
        train_targets = train_targets.to(device=args.device)

        test_inputs, test_targets = batch['test']
        test_inputs = test_inputs.to(device=args.device)
        test_targets = test_targets.to(device=args.device)

        outer_loss = torch.tensor(0., device=args.device)
        accuracy = torch.tensor(0., device=args.device)
        for task_idx, (train_input, train_target, test_input, test_target) in enumerate(zip(train_inputs, train_targets, test_inputs, test_targets)):
            acc_list = []

            ## 0 update step accuracy
            test_logit = model(test_input)
            with torch.no_grad():
                acc_list.append(get_accuracy(test_logit, test_target).item())

            ## 1 update step process
            train_logit = model(train_input)
            inner_loss = F.cross_entropy(train_logit, train_target)
            params = gradient_update_parameters(model=model, meta_params=M, loss=inner_loss, inner_lr=args.inner_lr, first_order=args.first_order, GAP=args.GAP, approx=args.approx)
            test_logit = model(test_input, params=params)
            with torch.no_grad():
                acc_list.append(get_accuracy(test_logit, test_target).item())

            ## 2 ~ k update step process
            for i in range(args.train_num_update_step - 1):
                train_logit = model(train_input, params)
                inner_loss = F.cross_entropy(train_logit, train_target)
                params = gradient_update_parameters(model=model, meta_params=M, params=params, loss=inner_loss, inner_lr=args.inner_lr, first_order=args.first_order, GAP=args.GAP, approx=args.approx)
                test_logit = model(test_input, params=params)
                with torch.no_grad():
                    acc_list.append(get_accuracy(test_logit, test_target).item())

            outer_loss += F.cross_entropy(test_logit, test_target)
            with torch.no_grad():
                accuracy += get_accuracy(test_logit, test_target)

        outer_loss.div_(args.batch_size)
        accuracy.div_(args.batch_size)
        outer_loss_list.append(outer_loss.item())
        outer_loss.backward()
        meta_optimizer.step()

        if batch_idx % 10 == 0:
            logger.info('batch_idx: {0} / train avg end-acc: {1:.4f} / loss: {2:.4f}'.format(batch_idx, accuracy.item(), outer_loss.item()))
            logger.info('per step accs: {0}'.format(acc_list))

        if batch_idx % 2000 == 0:
            val_acc = evaluate(model, M, meta_val_dataloader, args.device, args.inner_lr, args.first_order, args.test_num_update_step, args.GAP, args.approx)
            logger.info("Val Acc: {0:.4f} / Val Acc Confidence Intervals (CI): {1:.4f}".format(val_acc[0], val_acc[1]))
            if best_val_acc < val_acc[0]:
                save_path = get_save_path(dataset=args.dataset, train_ways=args.N_ways, train_shots=args.K_shots_for_support, GAP=args.GAP, approx=args.approx, outer_lr1=args.outer_lr1, outer_lr2=args.outer_lr2, batch_size=args.batch_size, flags='best')
                state_dict = model.state_dict()
                dict = {'model': state_dict, 'meta_params': M, 'iter': args.iter, 'test_acc_mean': val_acc[0], 'test_acc_ci': val_acc[1], 'losses': outer_loss_list, 'optimizer': meta_optimizer.state_dict()}
                torch.save(dict, save_path)
                best_val_acc = val_acc[0]

            test_acc = evaluate(model, M, meta_test_dataloader, args.device, args.inner_lr, args.first_order, args.test_num_update_step, args.GAP, args.approx)
            logger.info("Test Acc: {0:.4f} / Test Acc Confidence Intervals (CI): {1:.4f}".format(test_acc[0], test_acc[1]))
        if batch_idx >= args.iter:
            break

    val_acc = evaluate(model, M, meta_val_dataloader, args.device, args.inner_lr, args.first_order, args.test_num_update_step, args.GAP, args.approx)
    logger.info("Val Acc: {0:.4f} / Val Acc Confidence Intervals (CI): {1:.4f}".format(val_acc[0], val_acc[1]))

    # Save last a model
    save_path = get_save_path(dataset=args.dataset, train_ways=args.N_ways, train_shots=args.K_shots_for_support, GAP=args.GAP, approx=args.approx, outer_lr1=args.outer_lr1, outer_lr2=args.outer_lr2, batch_size=args.batch_size, flags='last')
    state_dict = model.state_dict()
    dict = {'model': state_dict, 'meta_params': M, 'iter': args.iter, 'test_acc_mean': val_acc[0], 'test_acc_ci': val_acc[1], 'losses': outer_loss_list, 'optimizer': meta_optimizer.state_dict()}
    torch.save(dict, save_path)

    test_acc = evaluate(model, M, meta_test_dataloader, args.device, args.inner_lr, args.first_order, args.test_num_update_step, args.GAP, args.approx)
    logger.info("Test Acc: {0:.4f} / Test Acc Confidence Intervals (CI): {1:.4f}".format(test_acc[0], test_acc[1]))
