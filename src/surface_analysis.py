import os
from src.utils.protein_utils import preprocess_unbound_bound, protein_to_graph_unbound_bound, UniformRotation_Translation
from biopandas.pdb import PandasPdb
from src.utils.args import *
from src.utils.zero_copy_from_numpy import *
import numpy as np
from Bio.PDB.ResidueDepth import ResidueDepth
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.ResidueDepth import get_surface
from Bio.PDB.ResidueDepth import residue_depth
from scipy import stats


def get_residues(pdb_filename):
    df = PandasPdb().read_pdb(pdb_filename).df['ATOM']
    df.rename(columns={'chain_id': 'chain', 'residue_number': 'residue', 'residue_name': 'resname',
                       'x_coord': 'x', 'y_coord': 'y', 'z_coord': 'z', 'element_symbol': 'element'}, inplace=True)
    residues = list(df.groupby(['chain', 'residue', 'resname']))  ## Not the same as sequence order !
    return residues


def main(args):
    dataset = 'dips'

    args['debug'] = False
    args['device'] = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    args['n_jobs'] = 1
    args['worker'] = 0


    input_dir = './test_sets_pdb/' + dataset + '_test_random_transformed/complexes/'

    pdb_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.endswith('.pdb')]
    for file in pdb_files:
        if not file.endswith('_l_b_COMPLEX.pdb'):
            continue
        ll = len('_l_b_COMPLEX.pdb')
        ligand_filename = os.path.join(input_dir, file[:-ll] + '_l_b' + '_COMPLEX.pdb')
        receptor_filename = os.path.join(input_dir, file[:-ll] + '_r_b' + '_COMPLEX.pdb')

        ppdb_ligand = PandasPdb().read_pdb(ligand_filename)
        unbound_ligand_all_atoms_pre_pos = ppdb_ligand.df['ATOM'][['x_coord', 'y_coord', 'z_coord']].to_numpy().squeeze().astype(np.float32)



        unbound_predic_ligand, \
        unbound_predic_receptor, \
        bound_ligand_repres_nodes_loc_clean_array,\
        bound_receptor_repres_nodes_loc_clean_array = preprocess_unbound_bound(
            get_residues(ligand_filename), get_residues(receptor_filename),
            graph_nodes=args['graph_nodes'], pos_cutoff=args['pocket_cutoff'], inference=True)


        ligand_graph, receptor_graph = protein_to_graph_unbound_bound(unbound_predic_ligand,
                                                                      unbound_predic_receptor,
                                                                      bound_ligand_repres_nodes_loc_clean_array,
                                                                      bound_receptor_repres_nodes_loc_clean_array,
                                                                      graph_nodes=args['graph_nodes'],
                                                                      cutoff=args['graph_cutoff'],
                                                                      max_neighbor=args['graph_max_neighbor'],
                                                                      one_hot=False,
                                                                      residue_loc_is_alphaC=args['graph_residue_loc_is_alphaC']
                                                                      )

        ligand_graph.ndata['new_x'] = ligand_graph.ndata['x']


# See https://biopython.org/docs/1.75/api/Bio.PDB.ResidueDepth.html

        parser = PDBParser()
        structure = parser.get_structure("aaa", ligand_filename)
        model = structure[0]
        chain = list(model.get_chains())[0]
        surface = get_surface(model)
        d = []
        surf_fs = []

        if len(chain) == ligand_graph.ndata['mu_r_norm'].shape[0]:
            for i in range(len(chain)):
                d.append(residue_depth(chain[i+1], surface))
                # print('depth = ', d[i], '; surf mu = ', ligand_graph.ndata['mu_r_norm'][i][4])
                surf_fs.append(- ligand_graph.ndata['mu_r_norm'][i][4].item())

            print('SPEARMAN for file ', ligand_filename, ' = ', stats.spearmanr(d, surf_fs))
            print('\n')

if __name__ == "__main__":
    main(args)