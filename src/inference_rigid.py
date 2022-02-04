import os
os.environ['DGLBACKEND'] = 'pytorch'

from datetime import datetime as dt
from src.utils.protein_utils import preprocess_unbound_bound, protein_to_graph_unbound_bound
from biopandas.pdb import PandasPdb
from src.utils.train_utils import *
from src.utils.args import *
from src.utils.ot_utils import *
from src.utils.zero_copy_from_numpy import *
from src.utils.io import create_dir


dataset = 'db5'
method_name = 'equidock'



def get_residues(pdb_filename):
    df = PandasPdb().read_pdb(pdb_filename).df['ATOM']
    df.rename(columns={'chain_id': 'chain', 'residue_number': 'residue', 'residue_name': 'resname',
                       'x_coord': 'x', 'y_coord': 'y', 'z_coord': 'z', 'element_symbol': 'element'}, inplace=True)
    residues = list(df.groupby(['chain', 'residue', 'resname']))  ## Not the same as sequence order !
    return residues



def main(args):

    ## Pre-trained models.
    if dataset == 'dips':
        checkpoint_filename = 'oct20_Wdec_0.0001#ITS_lw_10.0#Hdim_64#Nlay_8#shrdLay_F#ln_LN#lnX_0#Hnrm_0#NattH_50#skH_0.75#xConnI_0.0#LkySl_0.01#pokOTw_1.0#fine_F#/'
        checkpoint_filename = 'checkpts/' + checkpoint_filename + '/dips_model_best.pth'
    elif dataset == 'db5':
        checkpoint_filename = 'oct20_Wdec_0.001#ITS_lw_10.0#Hdim_64#Nlay_5#shrdLay_T#ln_LN#lnX_0#Hnrm_0#NattH_50#skH_0.5#xConnI_0.0#LkySl_0.01#pokOTw_1.0#fine_F#'
        checkpoint_filename = 'checkpts/' + checkpoint_filename + '/db5_model_best.pth'

    print('checkpoint_filename = ', checkpoint_filename)

    checkpoint = torch.load(checkpoint_filename, map_location=args['device'])

    for k,v in checkpoint['args'].items():
        args[k] = v
    args['debug'] = False
    args['device'] = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    args['n_jobs'] = 1
    args['worker'] = 0


    model = create_model(args, log)
    model.load_state_dict(checkpoint['state_dict'])
    param_count(model, log)
    model = model.to(args['device'])
    model.eval()

    print(args['layer_norm'], args['layer_norm_coors'], args['final_h_layer_norm'], args['intersection_loss_weight'])
    print('divide_coors_dist = ', args['divide_coors_dist'])



    time_list = []

    input_dir = './test_sets_pdb/' + dataset + '_test_random_transformed/random_transformed/'
    ground_truth_dir = './test_sets_pdb/' + dataset + '_test_random_transformed/complexes/'

    output_dir = './test_sets_pdb/' + dataset + '_' + method_name + '_results/'
    create_dir(output_dir)

    pdb_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.endswith('.pdb')]
    for file in pdb_files:
        if not file.endswith('_l_b.pdb'):
            continue
        ll = len('_l_b.pdb')
        ligand_filename = os.path.join(input_dir, file[:-ll] + '_l_b' + '.pdb')
        receptor_filename = os.path.join(ground_truth_dir, file[:-ll] + '_r_b' + '_COMPLEX.pdb')
        out_filename = file[:-ll] + '_l_b' + '_' + method_name.upper() + '.pdb'

        print(' inference on file = ', ligand_filename)


        start = dt.now()

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

        if args['input_edge_feats_dim'] < 0:
            args['input_edge_feats_dim'] = ligand_graph.edata['he'].shape[1]


        ligand_graph.ndata['new_x'] = ligand_graph.ndata['x']

        assert np.linalg.norm(bound_ligand_repres_nodes_loc_clean_array - ligand_graph.ndata['x'].detach().cpu().numpy()) < 1e-1

        # Create a batch of a single DGL graph
        batch_hetero_graph = batchify_and_create_hetero_graphs_inference(ligand_graph, receptor_graph)

        with torch.no_grad():
            batch_hetero_graph = batch_hetero_graph.to(args['device'])
            model_ligand_coors_deform_list, \
            model_keypts_ligand_list, model_keypts_receptor_list, \
            all_rotation_list, all_translation_list = model(batch_hetero_graph, epoch=0)


            rotation = all_rotation_list[0].detach().cpu().numpy()
            translation = all_translation_list[0].detach().cpu().numpy()

            new_residues = (rotation @ bound_ligand_repres_nodes_loc_clean_array.T).T+translation
            assert np.linalg.norm(new_residues - model_ligand_coors_deform_list[0].detach().cpu().numpy()) < 1e-1

            unbound_ligand_new_pos = (rotation @ unbound_ligand_all_atoms_pre_pos.T).T+translation

            ppdb_ligand.df['ATOM'][['x_coord', 'y_coord', 'z_coord']] = unbound_ligand_new_pos
            unbound_ligand_save_filename = os.path.join(output_dir, out_filename)
            ppdb_ligand.to_pdb(path=unbound_ligand_save_filename, records=['ATOM'], gz=False)

        end = dt.now()
        time_list.append((end-start).total_seconds())

    time_array = np.array(time_list)
    log(f"Mean runtime: {np.mean(time_array)}, std runtime: {np.std(time_array)}")
    log('Time list = ', time_list)


if __name__ == "__main__":
    main(args)