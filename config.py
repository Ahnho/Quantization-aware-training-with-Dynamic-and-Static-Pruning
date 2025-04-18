import argparse
import models
from data import valid_datasets as dataset_names


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

r'''learning rate scheduler types
    - step: Decays the learning rate of each parameter group
            by gamma every step_size epochs.
    - multistep: Decays the learning rate of each parameter group
                 by gamma once the number of epoch reaches one of the milestones.
    - exp: Decays the learning rate of each parameter group by gamma every epoch.
    - cosine: Set the learning rate of each parameter group
              using a cosine annealing schedule.
'''
schedule_types = [
    'step', 'multistep', 'exp', 'cosine'
]

def config():
    r"""configuration settings
    """
    parser = argparse.ArgumentParser(description='AI-Challenge Base Code')
    parser.add_argument('dataset', metavar='DATA', default='cifar10',
                        choices=dataset_names,
                        help='dataset: ' +
                             ' | '.join(dataset_names) +
                             ' (default: cifar10)')     
    # for model architecture
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet)')
    parser.add_argument('--layers', default=56, type=int, metavar='N',
                        help='number of layers in ResNet (default: 56)')
    parser.add_argument('--width-mult', default=1.0, type=float, metavar='WM',
                        help='width multiplier to thin a network '
                             'uniformly at each layer (default: 1.0)')
    parser.add_argument('--depth-mult', default=1.0, type=float, metavar='DM',
                         help='depth multiplier network (rexnet)')
    parser.add_argument('--model-mult', default=0, type=int,
                        help="e.g. efficient type (0 : b0, 1 : b1, 2 : b2 ...)")
    # for dataset
    parser.add_argument('--datapath', default='../data', type=str, metavar='PATH',
                        help='where you want to load/save your dataset? (default: ../data)')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    # for learning policy
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run (default: 200)')
    parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate (default: 0.1)',
                        dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--wd', '--weight-decay', dest='weight_decay',
                        default=5e-4, type=float,
                        help='weight decay (default: 5e-4)')
    parser.add_argument('--nest', '--nesterov', dest='nesterov', action='store_true',
                        help='use nesterov momentum?')
    parser.add_argument('--sched', '--scheduler', dest='scheduler', metavar='TYPE',
                        default='step', type=str, choices=schedule_types,
                        help='scheduler: ' +
                             ' | '.join(schedule_types) +
                             ' (default: step)')
    parser.add_argument('--step-size', dest='step_size', default=30,
                        type=int, metavar='STEP',
                        help='period of learning rate decay / '
                             'maximum number of iterations for '
                             'cosine annealing scheduler (default: 30)')
    parser.add_argument('--milestones', metavar='EPOCH', default=[100,150], type=int, nargs='+',
                        help='list of epoch indices for multi step scheduler '
                             '(must be increasing) (default: 100 150)')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='multiplicative factor of learning rate decay (default: 0.1)')
    parser.add_argument('--gammas', default=[0.1, 0.1], type=float, nargs='+',    ########################### yj
                        help='multiplicative factor of learning rate decay (default: 0.1)')
    parser.add_argument('--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    # for gpu configuration
    parser.add_argument('-C', '--cuda', dest='cuda', action='store_true',
                        help='use cuda?')
    parser.add_argument('-g', '--gpuids', metavar='GPU', default=[0],
                        type=int, nargs='+',
                        help='GPU IDs for using (default: 0)')
    # specify run type
    parser.add_argument('--run-type', default='train', type=str, metavar='TYPE',
                        help='type of run the main function e.g. train or evaluate (default: train)')
    # for load and save
    parser.add_argument('--load', default=None, type=str, metavar='FILE.pth',
                        help='name of checkpoint for testing model (default: None)')
    parser.add_argument('--save', default='ckpt.pth', type=str, metavar='FILE.pth',
                        help='name of checkpoint for saving model (default: ckpt.pth)')
    #############
    # for pruning
    parser.add_argument('-P', '--prune', dest='prune', action='store_true',
                         help='Use pruning')
    parser.add_argument('--pruner', default='dpf', type=str,
                        help='method of pruning to apply (default: dpf)')
    parser.add_argument('--prune-type', dest='prune_type', default='unstructured',
                         type=str, help='specify \'unstructured\' or \'structured\'')
    parser.add_argument('--prune-freq', dest='prune_freq', default=16, type=int,
                         help='update frequency')
    parser.add_argument('--prune-rate', dest='prune_rate', default=0.5, type=float,
                         help='pruning rate')
    parser.add_argument('--prune-imp', dest='prune_imp', default='L1', type=str,
                         help='Importance Method : L1, L2, grad, syn')

    parser.add_argument('--prun', dest='prune_imp', default='L1', type=str,
                        help='Cuda_num')

    parser.add_argument('--cu_num', default='1', type=str)

    parser.add_argument('--warmup', default=16, type=int,
                         help='warmup epoch for KD ')

    parser.add_argument('--target_epoch', default=16, type=int,
                         help='target_pruning_epoch ')


    parser.add_argument('--rewind_epoch', default=16, type=int,
                         help='target_pruning_epoch ')


    parser.add_argument('--txt_name_train', default=16, type=str,
                         help='name ')

    parser.add_argument('--txt_name_test', default=16, type=str,
                         help='name ')


    parser.add_argument('--n_bits', default=4, type=int, metavar='N',
                        help='number of total epochs to run (default: 200)')
    
    parser.add_argument('--acti_n_bits', default=4, type=int, metavar='N',
                        help='number of total epochs to run (default: 200)')
    
    parser.add_argument('--acti_quan',default=0, type=int, metavar='N',
                        help=' 1= activation quantization (default: 0)')
    
    parser.add_argument('--method',default='ours_freeze', type=str, metavar='N',
                        help='ours_freeze,ours,update,static_freeze,static_update,dynamic_freeze,dynamic_update ...')
    
    ## activation quant
    # parser.add_argument('--activation_bits', default = 0, type=int)
    # parser.add_argument('--preactivation', default = False, action='store_true')


    parser.add_argument('--interval_factor', default=0.5, type=float,
                         help='pruning rate')


    parser.add_argument('--dpf_epoch', default=0, type=int, metavar='N',
                        help='number of total epochs to run (default: 200)')
    
    parser.add_argument('--txt_time', default=16, type=str,
                         help='name ')

    #args.dpf_epoch

    cfg = parser.parse_args()
    return cfg


# parser.add_argument('--dir', type=str, default=None, required=True, help='training directory (default: None)')

# parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
# parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
#                     help='path to datasets location (default: None)')
# parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
# parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
# parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
#                     help='model name (default: None)')

# parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
#                     help='checkpoint to resume training from (default: None)')

# parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')
# parser.add_argument('--save_freq', type=int, default=25, metavar='N', help='save frequency (default: 25)')
# parser.add_argument('--eval_freq', type=int, default=5, metavar='N', help='evaluation frequency (default: 5)')
# parser.add_argument('--lr_init', type=float, default=0.1, metavar='LR', help='initial learning rate (default: 0.01)')
# parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
# parser.add_argument('--wd', type=float, default=1e-4, help='weight decay (default: 1e-4)')

# parser.add_argument('--swa', action='store_true', help='swa usage flag (default: off)')
# parser.add_argument('--swa_start', type=float, default=161, metavar='N', help='SWA start epoch number (default: 161)')
# parser.add_argument('--swa_lr', type=float, default=0.05, metavar='LR', help='SWA LR (default: 0.05)')
# parser.add_argument('--swa_c_epochs', type=int, default=1, metavar='N',
#                     help='SWA model collection frequency/cycle length in epochs (default: 1)')

# # parser.add_argument('-C', '--cuda', dest='cuda', action='store_true',
# #                         help='use cuda?')

# parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')