from datasets.helpers import miniimagenet, tieredimagenet, cars, cub
import sys


def get_dataset(dataset, folder, N_ways, K_shots_for_support, K_shots_for_query, download, train_transform, test_transform):
        if dataset == 'miniImageNet':
            meta_train_dataset = miniimagenet(folder, shots=K_shots_for_support, ways=N_ways, shuffle=True, test_shots=K_shots_for_query, meta_train=True, download=download, transform=train_transform)
            meta_val_dataset = miniimagenet(folder, shots=K_shots_for_support, ways=N_ways, shuffle=True, test_shots=K_shots_for_query, meta_val=True, download=download, transform=test_transform)
            meta_test_dataset = miniimagenet(folder, shots=K_shots_for_support, ways=N_ways, shuffle=True, test_shots=K_shots_for_query, meta_test=True, download=download, transform=test_transform)
        elif dataset == 'tieredImageNet':
            meta_train_dataset = tieredimagenet(folder, shots=K_shots_for_support, ways=N_ways, shuffle=True, test_shots=K_shots_for_query, meta_train=True, download=download, transform=train_transform)
            meta_val_dataset = tieredimagenet(folder, shots=K_shots_for_support, ways=N_ways, shuffle=True, test_shots=K_shots_for_query, meta_val=True, download=download, transform=test_transform)
            meta_test_dataset = tieredimagenet(folder, shots=K_shots_for_support, ways=N_ways, shuffle=True, test_shots=K_shots_for_query, meta_test=True, download=download, transform=test_transform)
        elif dataset == 'CARS':
            meta_train_dataset = cars(folder, shots=K_shots_for_support, ways=N_ways, shuffle=True, test_shots=K_shots_for_query, meta_train=True, download=download, transform=train_transform)
            meta_val_dataset = cars(folder, shots=K_shots_for_support, ways=N_ways, shuffle=True, test_shots=K_shots_for_query, meta_val=True, download=download, transform=test_transform)
            meta_test_dataset = cars(folder, shots=K_shots_for_support, ways=N_ways, shuffle=True, test_shots=K_shots_for_query, meta_test=True, download=download, transform=test_transform)
        elif dataset == 'CUB':
            meta_train_dataset = cub(folder, shots=K_shots_for_support, ways=N_ways, shuffle=True, test_shots=K_shots_for_query, meta_train=True, download=download, transform=train_transform)
            meta_val_dataset = cub(folder, shots=K_shots_for_support, ways=N_ways, shuffle=True, test_shots=K_shots_for_query, meta_val=True, download=download, transform=test_transform)
            meta_test_dataset = cub(folder, shots=K_shots_for_support, ways=N_ways, shuffle=True, test_shots=K_shots_for_query, meta_test=True, download=download, transform=test_transform)
        else:
            print('Not Found Dataset..!')
            sys.exit()
        return meta_train_dataset, meta_val_dataset, meta_test_dataset
