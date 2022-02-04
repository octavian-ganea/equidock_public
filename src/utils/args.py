# Ignore future warning
import sys
import warnings
import datetime
import random

warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import os
import torch

print('Parsing args')

parser = argparse.ArgumentParser(description='Docking')


parser.add_argument('-debug', default=False, action='store_true')

parser.add_argument('-log_every', default=100000, type=int, required=False, help='log frequency during training')
parser.add_argument('-random_seed', type=int, required=False, default=8, help='random seed')


# Data
parser.add_argument('-data', type=str, required=False, default='db5', choices=['db5', 'dips'])
parser.add_argument('-data_fraction', type=float, default=1., required=False)
parser.add_argument('-split', type=int, required=False, default=0, help='cross valid split')
parser.add_argument('-worker', type=int, default=5, required=False, help="Number of worker for data loader.")
parser.add_argument('-n_jobs', type=int, default=10, required=False, help="Number of worker for data preprocessing")


# Optim and Scheduler
parser.add_argument('-lr', type=float, default=3e-4, required=False)
parser.add_argument('-w_decay', type=float, default=1e-4, required=False)
parser.add_argument('-scheduler', default='warmup', choices=['ROP', 'warmup', 'cyclic'])
parser.add_argument('-warmup', type=float, default=1., required=False)
parser.add_argument('-patience', type=int, default=50, required=False, help='patience')
parser.add_argument('-num_epochs', type=int, default=10000, required=False, help="Used when splitting data for horovod.")
parser.add_argument('-clip', type=float, default=100., required=False, help="Gradient clip threshold.")
parser.add_argument('-bs', type=int, default=10, required=False)


### GRAPH characteristics and features
parser.add_argument('-graph_nodes', type=str, default='residues', required=False, choices=['residues'])
#################### Only for data caching, inference and to know which data to load.
parser.add_argument('-graph_cutoff', type=float, default=30., required=False, help='Only for data caching and inference.')
parser.add_argument('-graph_max_neighbor', type=int, default=10, required=False, help='Only for data caching and inference.')
parser.add_argument('-graph_residue_loc_is_alphaC', default=False, action='store_true',
                    help='whether to use coordinates of alphaC or avg of atom locations as the representative residue location.'
                         'Only for data caching and inference.')
parser.add_argument('-pocket_cutoff', type=float, default=8., required=False)


# Unbound - bound initial positions
parser.add_argument('-translation_interval', default=5.0, type=float, required=False, help='translation interval')

# Model
parser.add_argument('-rot_model', type=str, default='kb_att', choices=['kb_att'])
parser.add_argument('-num_att_heads', type=int, default=50, required=False)



## Pocket OT:
parser.add_argument('-pocket_ot_loss_weight', type=float, default=1., required=False)


# Intersection loss:
parser.add_argument('-intersection_loss_weight', type=float, default=10., required=False)
parser.add_argument('-intersection_sigma', type=float, default=25., required=False)
parser.add_argument('-intersection_surface_ct', type=float, default=10., required=False)





parser.add_argument('-dropout', type=float, default=0., required=False)
parser.add_argument('-layer_norm', type=str, default='LN', choices=['0', 'BN', 'LN'])
parser.add_argument('-layer_norm_coors', type=str, default='0', choices=['0', 'LN'])
parser.add_argument('-final_h_layer_norm', type=str, default='0', choices=['0', 'GN', 'BN', 'LN'])

parser.add_argument('-nonlin', type=str, default='lkyrelu', choices=['swish', 'lkyrelu'])
parser.add_argument('-iegmn_lay_hid_dim', type=int, default=64, required=False)
parser.add_argument('-iegmn_n_lays', type=int, default=5, required=False)
parser.add_argument('-residue_emb_dim', type=int, default=64, required=False, help='embedding')
parser.add_argument('-shared_layers', default=False, action='store_true')
parser.add_argument('-cross_msgs', default=False, action='store_true')

parser.add_argument('-divide_coors_dist', default=False, action='store_true')


parser.add_argument('-use_dist_in_layers', default=False, action='store_true')
parser.add_argument('-use_edge_features_in_gmn', default=False, action='store_true')

parser.add_argument('-noise_decay_rate', type=float, default=0., required=False)
parser.add_argument('-noise_initial', type=float, default=0., required=False)

parser.add_argument('-use_mean_node_features', default=False, action='store_true')

parser.add_argument('-skip_weight_h', type=float, default=0.5, required=False)

parser.add_argument('-leakyrelu_neg_slope', type=float, default=1e-2, required=False)


parser.add_argument('-x_connection_init', type=float, default=0., required=False)

## Hyper search
parser.add_argument('-hyper_search', default=False, action='store_true')


parser.add_argument('-fine_tune', default=False, action='store_true') ## Some fine-tuning E-GNN model that didn't work, feel free to play with it.


parser.add_argument('-toy', default=False, action='store_true') ## Train only on DB5.5


parser.add_argument('-continue_train_model', type=str, default='')


args = parser.parse_args().__dict__


args['device'] = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f"Available GPUS:{torch.cuda.device_count()}")

if torch.cuda.is_available():
    torch.cuda.set_device(0)




if args['continue_train_model'] != '':
    print('Continue training the DIPS model ', args['continue_train_model'])
    args['continue_train_model'] = 'checkpts/' + args['continue_train_model'] + '/dips_model_best.pth'
    checkpoint = torch.load(args['continue_train_model'], map_location=args['device'])
    for k, v in checkpoint['args'].items():
        if 'continue_train_model' not in k:
            args[k] = v
    args['hyper_search'] = False



########################################
def get_model_name(args):
    params_to_plot = {}
    # params_to_plot['data'] = ''
    # params_to_plot['data_fraction'] = 'f'
    #
    # params_to_plot['graph_nodes'] = 'gn'
    # params_to_plot['pocket_cutoff'] = 'pockC'

    # params_to_plot['graph_cutoff'] = 'cut'
    # params_to_plot['graph_max_neighbor'] = 'mxNE'
    # params_to_plot['graph_residue_loc_is_alphaC'] = 'aC'

    # params_to_plot['lr'] = 'lr'
    # params_to_plot['clip'] = 'cl'
    params_to_plot['dropout'] = 'drp'
    params_to_plot['w_decay'] = 'Wdec'
    #

    params_to_plot['intersection_loss_weight'] = 'ITS_lw'
    # params_to_plot['intersection_sigma'] = 'sigma'
    # params_to_plot['intersection_surface_ct'] = 'surf_ct'


    # params_to_plot['nonlin'] = 'nonlin'
    params_to_plot['iegmn_lay_hid_dim'] = 'Hdim'
    params_to_plot['iegmn_n_lays'] = 'Nlay'
    # params_to_plot['residue_emb_dim'] = 'Rdim'
    params_to_plot['shared_layers'] = 'shrdLay'


    # params_to_plot['cross_msgs'] = 'Xmsgs'
    # params_to_plot['use_dist_in_layers'] = 'DistInLys'
    # params_to_plot['use_edge_features_in_gmn'] = 'EdgeFeas'
    params_to_plot['use_mean_node_features'] = 'SURFfs'

    params_to_plot['layer_norm'] = 'ln'
    params_to_plot['layer_norm_coors'] = 'lnX'
    params_to_plot['final_h_layer_norm'] = 'Hnrm'

    # params_to_plot['rot_model'] = 'rotM'
    params_to_plot['num_att_heads'] = 'NattH'


    params_to_plot['skip_weight_h'] = 'skH'
    params_to_plot['x_connection_init'] = 'xConnI'

    params_to_plot['leakyrelu_neg_slope'] = 'LkySl'

    params_to_plot['pocket_ot_loss_weight'] = 'pokOTw'

    params_to_plot['divide_coors_dist'] = 'divXdist'


    def tostr(v):
        if type(v) is bool and v == True:
            return 'T'
        elif type(v) is bool and v == False:
            return 'F'
        return str(v)

    sss = list(params_to_plot.keys()) # preserves order
    model_name = 'EQUIDOCK__'
    for s in sss:
        assert s in args.keys()
        if len(params_to_plot[s].strip()) > 0:
            model_name += params_to_plot[s].strip() + '_' + tostr(args[s.strip()]) + '#'
        else:
            model_name += tostr(args[s.strip()]) + '#'
    assert len(model_name) <= 255
    return model_name
########################################



## Some hyperaparameter search example
if args['hyper_search']:
    model_was_solved = True
    num_tries = 0
    while model_was_solved:
        num_tries += 1
        if num_tries > 100:
            print('No hyperparams available !! Exiting ... ')
            sys.exit(1)

        args['data'] = 'dips'
        args['data_fraction'] = 1.
        args['split'] = 0


        args['graph_nodes'] = 'residues'
        args['pocket_cutoff'] = 8.
        args['graph_cutoff'] = 30.
        args['graph_max_neighbor'] = 10
        args['graph_residue_loc_is_alphaC'] = True


        args['clip'] = 100. # random.choices([1., 100.], weights=(0.25, 0.75), k=1)[0]
        args['dropout'] = random.choices([0, 0.25], weights=(0.5, 0.5), k=1)[0]
        args['w_decay'] = random.choices([1e-4, 1e-3], weights=(0.2, 0.2), k=1)[0]

        args['intersection_loss_weight'] = random.choices([10., 1.], weights=(0.6, 0.6), k=1)[0]
        ###### ground truth complexes have: sigma_25.0#surf_ct_10.0 --> intersection loss 0.8626
        args['intersection_sigma'] = 25.
        args['intersection_surface_ct'] = 10.


        args['layer_norm'] = 'LN' # random.choices(['0', 'LN'], weights=(0., 0.1), k=1)[0]
        args['layer_norm_coors'] = '0' #random.choices(['0', 'LN'], weights=(0.5, 0.1), k=1)[0]
        args['final_h_layer_norm'] = '0' # random.choices(['0', 'LN'], weights=(0.1, 0.1), k=1)[0]


        args['rot_model'] = 'kb_att'
        args['num_att_heads'] = 50 # random.choices([25, 50, 100], weights=(0.2, 0.2, 0.2), k=1)[0]


        args['pocket_ot_loss_weight'] = random.choices([10., 1.], weights=(0.3, 0.3), k=1)[0]


        args['nonlin'] = 'lkyrelu'
        args['leakyrelu_neg_slope'] = 0.01 # random.choice([0.1, 0.01])

        args['iegmn_lay_hid_dim'] = random.choice([64])
        args['iegmn_n_lays'] = random.choice([5, 8])
        args['residue_emb_dim'] = args['iegmn_lay_hid_dim']

        args['shared_layers'] = random.choices([True, False], weights=(0.1, 0.1), k=1)[0]

        args['divide_coors_dist'] = random.choices([True, False], weights=(0, 1), k=1)[0]

        args['cross_msgs'] = True
        args['use_dist_in_layers'] = True
        args['use_edge_features_in_gmn'] = True
        args['use_mean_node_features'] = True
        args['noise_decay_rate'] = 0.
        args['noise_initial'] = 0.

        args['skip_weight_h'] = random.choice([0.75, 0.5])
        args['x_connection_init'] = random.choices([0., 0.25], weights=(10, 1), k=1)[0]

        assert args['noise_decay_rate'] < 1., 'Noise has to decrease to 0, decay rate cannot be >= 1.'


        bbanner = get_model_name(args)
        model_was_solved = os.path.exists(os.path.join('stdouterr/', bbanner + ".txt"))



assert args['noise_decay_rate'] < 1., 'Noise has to decrease to 0, decay rate cannot be >= 1.'



banner = get_model_name(args)

print(banner)


def pprint(*kargs):
    print('[' + str(datetime.datetime.now()) + '] ', *kargs)

def log(*pargs): # redefined for train
    pprint(*pargs)

log('Model name ===> ', banner)

args['cache_path'] = './cache/' + args['data'] + '_' + args['graph_nodes'] + '_maxneighbor_' + \
                     str(args['graph_max_neighbor']) + '_cutoff_' + str(args['graph_cutoff']) + \
                     '_pocketCut_' + str(args['pocket_cutoff']) + '/'

args['checkpoint_dir'] = './checkpts/' + banner

args['tb_log_dir'] = './tb_logs/'  + banner

