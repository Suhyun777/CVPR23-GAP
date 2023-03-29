from dataset import get_dataset
from torchmeta.utils.data import BatchMetaDataLoader
from backbone import ConvolutionalNeuralNetwork
from transform import get_transform
from utils import *
from inner_loop_optimizer import gradient_update_parameters


def test(args):
    print('Source dataset: {0} / Target dataset: {1} / GAP:{2} / Approx:{3} / {4} / Test Begin'.format(args.dataset_for_source, args.dataset_for_target, args.GAP, args.approx, args.flags))
    train_transform = get_transform(train=True)
    test_transform = get_transform(train=False)

    _, _, meta_test_dataset = get_dataset(args.dataset_for_target, args.folder, args.N_ways, args.K_shots_for_support, args.K_shots_for_query, args.download, train_transform, test_transform)

    meta_test_dataloader = BatchMetaDataLoader(meta_test_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers)


    model = ConvolutionalNeuralNetwork(3, args.N_ways, hidden_size=args.hidden_size)
    model.to(device=args.device)
    model.train()


    save_path = get_save_path(dataset=args.dataset_for_source, train_ways=args.N_ways, train_shots=args.K_shots_for_support, GAP=args.GAP, approx=args.approx, outer_lr1=args.outer_lr1, outer_lr2=args.outer_lr2, batch_size=args.batch_size, flags=args.flags)
    prev_state = torch.load(save_path)
    model.load_state_dict(prev_state['model'])
    M = prev_state['meta_params']


    log_path = get_test_log(dataset=args.dataset_for_source, train_ways=args.N_ways, train_shots=args.K_shots_for_support, GAP=args.GAP, approx=args.approx, outer_lr1=args.outer_lr1, outer_lr2=args.outer_lr2, batch_size=args.batch_size)
    logger = get_logger(log_path)


    test_acc = evaluate(model, M, meta_test_dataloader, args.device, args.inner_lr, args.first_order, args.test_num_update_step, args.GAP, args.approx)
    logger.info("Test Acc: {2:.4f} / Test Acc Confidence Intervals (CI): {3:.4f}".format(args.dataset_for_source, args.dataset_for_target, test_acc[0], test_acc[1]))
