import random

from dgl import save_graphs, load_graphs
import numpy as np
import os
from torch.utils.data import Dataset
from src.utils.io import pmap_multi

from src.utils.protein_utils import preprocess_unbound_bound, protein_to_graph_unbound_bound, UniformRotation_Translation
import pickle
from biopandas.pdb import PandasPdb
import pandas as pd
from src.utils.zero_copy_from_numpy import *

def get_residues_db5(pdb_filename):
    df = PandasPdb().read_pdb(pdb_filename).df['ATOM']
    df.rename(columns={'chain_id': 'chain', 'residue_number': 'residue', 'residue_name': 'resname',
                       'x_coord': 'x', 'y_coord': 'y', 'z_coord': 'z', 'element_symbol': 'element'}, inplace=True)
    residues = list(df.groupby(['chain', 'residue', 'resname']))  ## Not the same as sequence order !
    return residues



def get_residues_DIPS(dill_filename):
    x = pd.read_pickle(dill_filename)
    df0 = x.df0
    df0.rename(columns={'chain_id': 'chain', 'residue_number': 'residue', 'residue_name': 'resname',
                       'x_coord': 'x', 'y_coord': 'y', 'z_coord': 'z', 'element_symbol': 'element'}, inplace=True)
    residues0 = list(df0.groupby(['chain', 'residue', 'resname']))  ## Not the same as sequence order !
    df1 = x.df1
    df1.rename(columns={'chain_id': 'chain', 'residue_number': 'residue', 'residue_name': 'resname',
                       'x_coord': 'x', 'y_coord': 'y', 'z_coord': 'z', 'element_symbol': 'element'}, inplace=True)
    residues1 = list(df1.groupby(['chain', 'residue', 'resname']))  ## Not the same as sequence order !
    return residues0, residues1



__all__ = ['Unbound_Bound_Data']
class Unbound_Bound_Data(Dataset):

    def __init__(self, args, if_swap=True, reload_mode='train',
                 load_from_cache=True, raw_data_path=None, split_files_path=None, data_fraction=1.):
        self.args = args
        self.reload_mode = reload_mode
        self.if_swap = if_swap

        frac_str = ''
        if reload_mode == 'train' and args['data'] == 'dips':
            frac_str = 'frac_' + str(data_fraction) + '_'

        label_filename = os.path.join(args['cache_path'], 'label_' + frac_str + reload_mode + '.pkl')
        ligand_graph_filename = os.path.join(args['cache_path'], 'ligand_graph_' + frac_str + reload_mode + '.bin')
        receptor_graph_filename = os.path.join(args['cache_path'], 'receptor_graph_' + frac_str + reload_mode + '.bin')

        if load_from_cache:
            with open(label_filename, 'rb') as infile:
                label = pickle.load(infile)

            self.pocket_coors_list = label['pocket_coors_list']
            self.bound_ligand_repres_nodes_loc_array_list = label['bound_ligand_repres_nodes_loc_array_list']
            self.bound_receptor_repres_nodes_loc_array_list = label['bound_receptor_repres_nodes_loc_array_list']
            self.ligand_graph_list, _ = load_graphs(ligand_graph_filename)
            self.receptor_graph_list, _ = load_graphs(receptor_graph_filename)

        else:
            print('Processing ', label_filename)
            assert raw_data_path is not None and split_files_path is not None

            if os.path.exists(label_filename):
                print('\n\n\nNot recreating ', label_filename, ' because data already exists\n\n\n\n\n')
                return

            if args['data'] == 'db5':
                assert data_fraction == 1.
                onlyfiles = [f for f in os.listdir(raw_data_path) if os.path.isfile(os.path.join(raw_data_path, f))]
                code_set = set([file.split('_')[0] for file in onlyfiles])
                split_code_set = set()
                with open(os.path.join(split_files_path, reload_mode + '.txt'), 'r') as f:
                    for line in f.readlines():
                        split_code_set.add(line.rstrip())

                code_set = code_set & split_code_set
                code_list = list(code_set)

                bound_ligand_residues_list = [get_residues_db5(os.path.join(raw_data_path, code + '_l_b.pdb'))
                                              for code in code_list]
                bound_receptor_residues_list = [get_residues_db5(os.path.join(raw_data_path, code + '_r_b.pdb'))
                                                for code in code_list]

                input_residues_lists = [(bound_ligand_residues_list[i], bound_receptor_residues_list[i])
                                        for i in range(len(bound_ligand_residues_list))]

            else:
                assert args['data'] == 'dips'
                if reload_mode != 'train':
                    data_fraction = 1.

                dill_filenames_list = []
                with open(os.path.join(split_files_path, 'pairs-postprocessed-' + reload_mode + '.txt'), 'r') as f:
                    for line in f.readlines():
                        dill_filenames_list.append(line.rstrip())

                random.shuffle(dill_filenames_list)
                dill_filenames_list = dill_filenames_list[: int(data_fraction * len(dill_filenames_list))]

                print('Num of pairs in ', reload_mode, ' = ', len(dill_filenames_list))
                input_residues_lists = [get_residues_DIPS(os.path.join(raw_data_path, f)) for f in dill_filenames_list]


            print('Start preprocess_unbound_bound')
            preprocess_result = pmap_multi(preprocess_unbound_bound,
                                           input_residues_lists,
                                           n_jobs=args['n_jobs'],
                                           graph_nodes=args['graph_nodes'],
                                           pos_cutoff=args['pocket_cutoff'],
                                           inference=False)
            print('Done preprocess_unbound_bound\n\n')

            unbound_predic_ligand_list, unbound_predic_receptor_list = [], []
            bound_ligand_repres_nodes_loc_array_list, bound_receptor_repres_nodes_loc_array_list = [], []
            pocket_coors_list = []
            for result in preprocess_result:
                unbound_predic_ligand, unbound_predic_receptor,\
                bound_ligand_repres_nodes_loc_array, bound_receptor_repres_nodes_loc_array, pocket_coors = result
                if pocket_coors is not None:
                    unbound_predic_ligand_list.append(unbound_predic_ligand)
                    unbound_predic_receptor_list.append(unbound_predic_receptor)
                    bound_ligand_repres_nodes_loc_array_list.append(bound_ligand_repres_nodes_loc_array)
                    bound_receptor_repres_nodes_loc_array_list.append(bound_receptor_repres_nodes_loc_array)
                    pocket_coors_list.append(pocket_coors)


            label = {'pocket_coors_list': pocket_coors_list,
                     'bound_ligand_repres_nodes_loc_array_list': bound_ligand_repres_nodes_loc_array_list,
                     'bound_receptor_repres_nodes_loc_array_list': bound_receptor_repres_nodes_loc_array_list}

            with open(label_filename, 'wb') as outfile:
                pickle.dump(label, outfile, pickle.HIGHEST_PROTOCOL)

            protein_to_graph_input = [ (unbound_predic_ligand_list[i],
                                        unbound_predic_receptor_list[i],
                                        bound_ligand_repres_nodes_loc_array_list[i],
                                        bound_receptor_repres_nodes_loc_array_list[i]) for i in range(len(unbound_predic_ligand_list))]
            print('Start protein_to_graph_unbound_bound')

            both_proteins_to_graph_pair_list = pmap_multi(protein_to_graph_unbound_bound,
                                                          protein_to_graph_input,
                                                          n_jobs=args['n_jobs'],
                                                          graph_nodes=args['graph_nodes'],
                                                          cutoff=args['graph_cutoff'],
                                                          max_neighbor=args['graph_max_neighbor'],
                                                          one_hot=False,
                                                          residue_loc_is_alphaC=args['graph_residue_loc_is_alphaC']
                                                          )
            print('Done protein_to_graph_unbound_bound')

            ligand_graph_list, receptor_graph_list = [], []
            for result in both_proteins_to_graph_pair_list:
                ligand_graph, receptor_graph = result
                ligand_graph_list.append(ligand_graph)
                receptor_graph_list.append(receptor_graph)

            save_graphs(ligand_graph_filename, ligand_graph_list)
            save_graphs(receptor_graph_filename, receptor_graph_list)




    def __len__(self):
        return len(self.ligand_graph_list)

    def __getitem__(self, idx):
        swap = False
        if self.if_swap:
            rnd = np.random.uniform(low=0.0, high=1.0)
            if rnd > 0.5:
                swap = True

        if swap: # Just as a sanity check, but our model anyway is invariant to such swapping.
            bound_ligand_repres_nodes_loc_array = self.bound_receptor_repres_nodes_loc_array_list[idx]
            bound_receptor_repres_nodes_loc_array = self.bound_ligand_repres_nodes_loc_array_list[idx]
            ligand_graph = self.receptor_graph_list[idx]
            receptor_graph = self.ligand_graph_list[idx]
        else:
            bound_ligand_repres_nodes_loc_array = self.bound_ligand_repres_nodes_loc_array_list[idx]
            bound_receptor_repres_nodes_loc_array = self.bound_receptor_repres_nodes_loc_array_list[idx]
            ligand_graph = self.ligand_graph_list[idx]
            receptor_graph = self.receptor_graph_list[idx]


        pocket_coors_ligand = self.pocket_coors_list[idx]
        pocket_coors_receptor = self.pocket_coors_list[idx]

        # Randomly rotate and translate the ligand.
        rot_T, rot_b = UniformRotation_Translation(translation_interval=self.args['translation_interval'])

        ligand_original_loc = ligand_graph.ndata['x'].detach().numpy()

        mean_to_remove = ligand_original_loc.mean(axis=0, keepdims=True)

        pocket_coors_ligand = (rot_T @ (pocket_coors_ligand - mean_to_remove).T).T + rot_b
        ligand_new_loc = (rot_T @ (ligand_original_loc - mean_to_remove).T).T + rot_b

        ligand_graph.ndata['new_x'] = zerocopy_from_numpy(ligand_new_loc.astype(np.float32))

        return ligand_graph, receptor_graph,\
               zerocopy_from_numpy(bound_ligand_repres_nodes_loc_array.astype(np.float32)), \
               zerocopy_from_numpy(bound_receptor_repres_nodes_loc_array.astype(np.float32)), \
               zerocopy_from_numpy(pocket_coors_ligand.astype(np.float32)),\
               zerocopy_from_numpy(pocket_coors_receptor.astype(np.float32))


