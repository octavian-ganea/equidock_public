import os

import torch

os.environ['DGLBACKEND'] = 'pytorch'

from datetime import datetime as dt
from src.utils.protein_utils import preprocess_unbound_bound, protein_to_graph_unbound_bound
from biopandas.pdb import PandasPdb
from src.utils.train_utils import *
from src.utils.args import *
from src.utils.ot_utils import *
from src.utils.zero_copy_from_numpy import *
from src.utils.io import create_dir


dataset = 'dips'
method_name = 'equidock'
remove_clashes = False  # Set to true if you want to remove (most of the) steric clashes. Will increase run time.
if remove_clashes:
    method_name = method_name + '_no_clashes'
    print('Inference with postprocessing to remove clashes')
else:
    print('Inference without any postprocessing to remove clashes')


# Ligand residue locations: a_i in R^3. Receptor: b_j in R^3
# Ligand: G_l(x) = -sigma * ln( \sum_i  exp(- ||x - a_i||^2 / sigma)  ), same for G_r(x)
# Ligand surface: x such that G_l(x) = surface_ct
# Other properties: G_l(a_i) < 0, G_l(x) = infinity if x is far from all a_i
# Intersection of ligand and receptor: points x such that G_l(x) < surface_ct && G_r(x) < surface_ct
# Intersection loss: IL = \avg_i max(0, surface_ct - G_r(a_i)) + \avg_j max(0, surface_ct - G_l(b_j))
def G_fn(protein_coords, x, sigma):
    # protein_coords: (n,3) ,  x: (m,3), output: (m,)
    e = torch.exp(- torch.sum((protein_coords.view(1, -1, 3) - x.view(-1,1,3)) ** 2, dim=2) / float(sigma) )  # (m, n)
    return - sigma * torch.log(1e-3 +  e.sum(dim=1) )


def compute_body_intersection_loss(model_ligand_coors_deform, bound_receptor_repres_nodes_loc_array, sigma = 25., surface_ct=10.):
    assert model_ligand_coors_deform.shape[1] == 3
    loss = torch.mean( torch.clamp(surface_ct - G_fn(bound_receptor_repres_nodes_loc_array, model_ligand_coors_deform, sigma), min=0) ) + \
           torch.mean( torch.clamp(surface_ct - G_fn(model_ligand_coors_deform, bound_receptor_repres_nodes_loc_array, sigma), min=0) )
    return loss


def get_rot_mat(euler_angles):
    roll = euler_angles[0]
    yaw = euler_angles[1]
    pitch = euler_angles[2]

    tensor_0 = torch.zeros([])
    tensor_1 = torch.ones([])
    cos = torch.cos
    sin = torch.sin

    RX = torch.stack([
        torch.stack([tensor_1, tensor_0, tensor_0]),
        torch.stack([tensor_0, cos(roll), -sin(roll)]),
        torch.stack([tensor_0, sin(roll), cos(roll)])]).reshape(3, 3)

    RY = torch.stack([
        torch.stack([cos(pitch), tensor_0, sin(pitch)]),
        torch.stack([tensor_0, tensor_1, tensor_0]),
        torch.stack([-sin(pitch), tensor_0, cos(pitch)])]).reshape(3, 3)

    RZ = torch.stack([
        torch.stack([cos(yaw), -sin(yaw), tensor_0]),
        torch.stack([sin(yaw), cos(yaw), tensor_0]),
        torch.stack([tensor_0, tensor_0, tensor_1])]).reshape(3, 3)

    R = torch.mm(RZ, RY)
    R = torch.mm(R, RX)
    return R



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

    input_dir = './test_sets_pdb/jean/'
    ground_truth_dir = './test_sets_pdb/jean/'
    output_dir = './test_sets_pdb/jean_out/'
    # create_dir(output_dir)

    pdb_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.endswith('.pdb')]
    for file in pdb_files:

        if not file.endswith('_l_b.pdb'):
            continue
        ll = len('_l_b.pdb')
        ligand_filename = os.path.join(input_dir, file[:-ll] + '_l_b' + '.pdb')
        receptor_filename = os.path.join(ground_truth_dir, file[:-ll] + '_r_b' + '_COMPLEX.pdb')
        gt_ligand_filename = os.path.join(ground_truth_dir, file[:-ll] + '_l_b' + '_COMPLEX.pdb')
        out_filename = file[:-ll] + '_l_b' + '_' + method_name.upper() + '.pdb'

        print(' inference on file = ', ligand_filename)


        start = dt.now()

        ppdb_ligand = PandasPdb().read_pdb(ligand_filename)

        unbound_ligand_all_atoms_pre_pos = ppdb_ligand.df['ATOM'][['x_coord', 'y_coord', 'z_coord']].to_numpy().squeeze().astype(np.float32)


        def get_nodes_coors_numpy(filename, all_atoms=False):
            df = PandasPdb().read_pdb(filename).df['ATOM']
            if not all_atoms:
                return torch.from_numpy(df[df['atom_name'] == 'CA'][['x_coord', 'y_coord', 'z_coord']].to_numpy().squeeze().astype(np.float32))
            return torch.from_numpy(df[['x_coord', 'y_coord', 'z_coord']].to_numpy().squeeze().astype(np.float32))

        gt_ligand_nodes_coors = get_nodes_coors_numpy(gt_ligand_filename, all_atoms=True)
        gt_receptor_nodes_coors = get_nodes_coors_numpy(receptor_filename, all_atoms=True)
        initial_ligand_nodes_coors = get_nodes_coors_numpy(ligand_filename, all_atoms=True)



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

        batch_hetero_graph = batch_hetero_graph.to(args['device'])
        model_ligand_coors_deform_list, \
        model_keypts_ligand_list, model_keypts_receptor_list, \
        all_rotation_list, all_translation_list = model(batch_hetero_graph, epoch=0)


        rotation = all_rotation_list[0].detach().cpu().numpy()
        translation = all_translation_list[0].detach().cpu().numpy()

        new_residues = (rotation @ bound_ligand_repres_nodes_loc_clean_array.T).T+translation
        assert np.linalg.norm(new_residues - model_ligand_coors_deform_list[0].detach().cpu().numpy()) < 1e-1

        unbound_ligand_new_pos = (rotation @ unbound_ligand_all_atoms_pre_pos.T).T+translation

        euler_angles_finetune = torch.zeros([3], requires_grad=True)
        translation_finetune = torch.zeros([3], requires_grad=True)
        ligand_th = (get_rot_mat(euler_angles_finetune) @ torch.from_numpy(unbound_ligand_new_pos).T).T + translation_finetune

        ## Optimize the non-intersection loss:
        if remove_clashes:
            non_int_loss_item = 100.
            it = 0
            while non_int_loss_item > 0.5 and it < 2000:
                non_int_loss = compute_body_intersection_loss(ligand_th, gt_receptor_nodes_coors, sigma=8, surface_ct=8)
                non_int_loss_item = non_int_loss.item()
                eta = 1e-3
                if non_int_loss < 2.:
                    eta = 1e-4
                if it > 1500:
                    eta = 1e-2
                if it % 100 == 0:
                    print(it, ' ' , non_int_loss_item)
                non_int_loss.backward()
                translation_finetune = translation_finetune - eta * translation_finetune.grad.detach()
                translation_finetune = torch.tensor(translation_finetune, requires_grad=True)

                euler_angles_finetune = euler_angles_finetune - eta * euler_angles_finetune.grad.detach()
                euler_angles_finetune = torch.tensor(euler_angles_finetune, requires_grad=True)

                ligand_th = (get_rot_mat(euler_angles_finetune) @ torch.from_numpy(unbound_ligand_new_pos).T).T + translation_finetune

                it += 1


        ppdb_ligand.df['ATOM'][['x_coord', 'y_coord', 'z_coord']] = ligand_th.detach().numpy() # unbound_ligand_new_pos
        unbound_ligand_save_filename = os.path.join(output_dir, out_filename)
        ppdb_ligand.to_pdb(path=unbound_ligand_save_filename, records=['ATOM'], gz=False)

        end = dt.now()
        time_list.append((end-start).total_seconds())

    time_array = np.array(time_list)
    log(f"Mean runtime: {np.mean(time_array)}, std runtime: {np.std(time_array)}")
    log('Time list = ', time_list)


if __name__ == "__main__":
    main(args)