# -*- coding: utf-8 -*-

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system') # https://github.com/facebookresearch/maskrcnn-benchmark/issues/103

import numpy as np

import torch
import random
from torch.utils.data import DataLoader
from src.utils.db5_data import Unbound_Bound_Data
from functools import partial
import dgl
from src.model.rigid_docking_model import *

def set_random_seed(seed=0):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)



def get_dataloader(args, log):
    log('\n\n')
    log(f"# Loading dataset: {args['data']}")
    num_worker = 0  # 0 if args['debug'] else args['worker']
    log(f"# Num_worker:{num_worker}")

    train_set = Unbound_Bound_Data(args, if_swap=True, reload_mode='train', load_from_cache=True, data_fraction=args['data_fraction'])
    val_set = Unbound_Bound_Data(args, if_swap=False, reload_mode='val', load_from_cache=True)
    test_set = Unbound_Bound_Data(args, if_swap=False, reload_mode='test', load_from_cache=True)

    train_loader = DataLoader(dataset=train_set,
                              batch_size=args['bs'],
                              shuffle=True,
                              collate_fn=partial(batchify_and_create_hetero_graphs),
                              num_workers=num_worker,
                              )
    val_loader = DataLoader(dataset=val_set,
                            batch_size=args['bs'],
                            collate_fn=partial(batchify_and_create_hetero_graphs),
                            num_workers=num_worker
                            )
    test_loader = DataLoader(dataset=test_set,
                             batch_size=args['bs'],
                             collate_fn=partial(batchify_and_create_hetero_graphs),
                             num_workers=num_worker)



    log(f" Train:{len(train_loader.dataset)}, Valid:{len(val_loader.dataset)}, Test :{len(test_loader.dataset)}")
    args['input_edge_feats_dim'] = train_set[0][0].edata['he'].shape[1]

    log('input_edge_feats_dim : ', args['input_edge_feats_dim'])
    return train_loader, val_loader, test_loader



def hetero_graph_from_sg_l_r_pair(ligand_graph, receptor_graph):
    ll = [('ligand', 'll', 'ligand'), (ligand_graph.edges()[0], ligand_graph.edges()[1])]
    rr = [('receptor', 'rr', 'receptor'), (receptor_graph.edges()[0], receptor_graph.edges()[1])]
    rl = [('receptor', 'cross', 'ligand'),
          (torch.tensor([], dtype=torch.int32), torch.tensor([], dtype=torch.int32))]
    lr = [('ligand', 'cross', 'receptor'),
          (torch.tensor([], dtype=torch.int32), torch.tensor([], dtype=torch.int32))]
    num_nodes = {'ligand': ligand_graph.num_nodes(), 'receptor': receptor_graph.num_nodes()}
    hetero_graph = dgl.heterograph({ll[0]: ll[1], rr[0]: rr[1], rl[0]: rl[1], lr[0]: lr[1]}, num_nodes_dict=num_nodes)

    hetero_graph.nodes['ligand'].data['res_feat'] = ligand_graph.ndata['res_feat']
    hetero_graph.nodes['ligand'].data['x'] = ligand_graph.ndata['x']
    hetero_graph.nodes['ligand'].data['new_x'] = ligand_graph.ndata['new_x']
    hetero_graph.nodes['ligand'].data['mu_r_norm'] = ligand_graph.ndata['mu_r_norm']

    hetero_graph.edges['ll'].data['he'] = ligand_graph.edata['he']

    hetero_graph.nodes['receptor'].data['res_feat'] = receptor_graph.ndata['res_feat']
    hetero_graph.nodes['receptor'].data['x'] = receptor_graph.ndata['x']
    hetero_graph.nodes['receptor'].data['mu_r_norm'] = receptor_graph.ndata['mu_r_norm']

    hetero_graph.edges['rr'].data['he'] = receptor_graph.edata['he']
    return hetero_graph



def batchify_and_create_hetero_graphs(data):
    ligand_graph_list, receptor_graph_list, \
    bound_ligand_repres_nodes_loc_array_list, bound_receptor_repres_nodes_loc_array_list,\
    pocket_coors_ligand_list, pocket_coors_receptor_list = map(list, zip(*data))

    hetero_graph_list = []
    for i, ligand_graph in enumerate(ligand_graph_list):
        receptor_graph = receptor_graph_list[i]
        hetero_graph = hetero_graph_from_sg_l_r_pair(ligand_graph, receptor_graph)
        hetero_graph_list.append(hetero_graph)

    batch_hetero_graph = dgl.batch(hetero_graph_list)
    return batch_hetero_graph, bound_ligand_repres_nodes_loc_array_list, bound_receptor_repres_nodes_loc_array_list,\
           pocket_coors_ligand_list, pocket_coors_receptor_list


def batchify_and_create_hetero_graphs_inference(ligand_graph, receptor_graph):
    hetero_graph_list = []
    hetero_graph = hetero_graph_from_sg_l_r_pair(ligand_graph, receptor_graph)
    hetero_graph_list.append(hetero_graph)
    batch_hetero_graph = dgl.batch(hetero_graph_list)
    return batch_hetero_graph


def create_model(args, log):
    assert 'input_edge_feats_dim' in args.keys(), 'get_loader has to be called before create_model.'

    return Rigid_Body_Docking_Net(args=args, log=log)


def param_count(model, log, print_model=False):
    if print_model:
        log(model)
    param_count = sum(param.numel() for param in model.parameters() if param.requires_grad)
    log(f'Number of parameters = {param_count:,}')


def lr_lambda(step, warmup=10.):
    # return min(1, step / warmup) * (warmup / step) ** 0.5
    return min(1., ((step+1) / warmup)**3) #  * max(0.2, (warmup / step) ** 0.5)

def get_scheduler(optimizer, args):
    if args['scheduler'] == 'warmup':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=partial(lr_lambda, warmup=args['warmup']))
    elif args['scheduler'] == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args['lr'], step_size_up=args['warmup'],
                                                      max_lr=args['lr'] * 10,
                                                      cycle_momentum=False)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               factor=0.5,
                                                               patience=100,
                                                               verbose=True,
                                                               threshold_mode='rel',
                                                               cooldown=0,
                                                               min_lr=1e-12,
                                                               eps=1e-08)
    return scheduler


def pretty_print_stats(split_type, epoch, total_num_epochs,
                       complex_rmsd_mean, complex_rmsd_median,
                       ligand_rmsd_mean, ligand_rmsd_median,
                       receptor_rmsd_mean, receptor_rmsd_median,
                       avg_loss, avg_loss_ligand_coors,
                       avg_loss_receptor_coors, avg_loss_ot, avg_loss_intersection, log):

    log('[{:s}] --> epoch {:d}/{:d} '
          '|| mean/median complex rmsd {:.4f} / {:.4f} '
          '|| mean/median ligand rmsd {:.4f} / {:.4f} '
          '|| mean/median sqrt pocket OT loss {:.4f} '
          '|| intersection loss {:.4f} '
          '|| mean/median receptor rmsd {:.4f} / {:.4f} '.
          format(split_type,
                 epoch, total_num_epochs,
                 complex_rmsd_mean, complex_rmsd_median,
                 ligand_rmsd_mean, ligand_rmsd_median,
                 math.sqrt(avg_loss_ot),
                 avg_loss_intersection,
                 receptor_rmsd_mean, receptor_rmsd_median))