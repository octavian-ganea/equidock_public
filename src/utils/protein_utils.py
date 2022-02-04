import scipy.spatial as spa
import numpy as np
from numpy import linalg as LA
import dgl
from src.utils.protein_featurizers import residue_type_one_hot_dips_not_one_hot,  residue_type_one_hot_dips

import math

from scipy.special import softmax

from scipy.spatial.transform import Rotation
from src.utils.zero_copy_from_numpy import *


def UniformRotation_Translation(translation_interval):
    rotation = Rotation.random(num=1)
    rotation_matrix = rotation.as_matrix().squeeze()

    t = np.random.randn(1, 3)
    t = t / np.sqrt( np.sum(t * t))
    length = np.random.uniform(low=0, high=translation_interval)
    t = t * length
    return rotation_matrix.astype(np.float32), t.astype(np.float32)


# Input: expects 3xN matrix of points
# Returns R,t as of Kabsch algorithm (see prop 4.4 from the thesis)
# R = 3x3 rotation matrix
# t = 3x1 column vector
# This already takes residue identity into account.
def rigid_transform_Kabsch_3D(A, B):
    assert A.shape[1] == B.shape[1]
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")


    # find mean column wise: 3 x 1
    centroid_A = np.mean(A, axis=1, keepdims=True)
    centroid_B = np.mean(B, axis=1, keepdims=True)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ Bm.T

    # find rotation
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        SS = np.diag([1.,1.,-1.])
        R = (Vt.T @ SS) @ U.T
    assert math.fabs(np.linalg.det(R) - 1) < 1e-5

    t = -R @ centroid_A + centroid_B
    return R, t






def distance_list_featurizer(dist_list):
    length_scale_list = [1.5 ** x for x in range(15)]
    center_list = [0. for _ in range(15)]

    num_edge = len(dist_list)
    dist_list = np.array(dist_list)

    transformed_dist = [ np.exp( - ((dist_list - center) ** 2) / float(length_scale))
                         for length_scale, center in zip(length_scale_list, center_list)]

    transformed_dist = np.array(transformed_dist).T
    transformed_dist = transformed_dist.reshape((num_edge, -1))

    processed_features = dict()
    processed_features['he'] = zerocopy_from_numpy(transformed_dist.astype(np.float32))
    return processed_features


def residue_list_featurizer_dips_one_hot(predic):
    residue_list = [ term[1]['resname'].iloc[0] for term in predic ]
    feature_list = [residue_type_one_hot_dips(residue) for residue in residue_list]
    feature_list = np.stack(feature_list)
    processed_features = dict()
    processed_features['res_feat'] = zerocopy_from_numpy(feature_list.astype(np.float32))
    return processed_features

def residue_list_featurizer_dips_NOT_one_hot(predic):
    residue_list = [term[1]['resname'].iloc[0] for term in predic]
    feature_list = [[residue_type_one_hot_dips_not_one_hot(residue)] for residue in residue_list]
    feature_list = np.array(feature_list) ####
    processed_features = dict()
    processed_features['res_feat'] = zerocopy_from_numpy(feature_list.astype(np.float32))  # (N_res, 1)
    return processed_features



def preprocess_unbound_bound(bound_ligand_residues, bound_receptor_residues, graph_nodes, pos_cutoff=8.0, inference=False):
    #######################
    def filter_residues(residues):
        residues_filtered = []
        for residue in residues:
            df = residue[1]
            Natom = df[df['atom_name'] == 'N']
            alphaCatom = df[df['atom_name'] == 'CA']
            Catom = df[df['atom_name'] == 'C']

            if Natom.shape[0] == 1 and alphaCatom.shape[0] == 1 and Catom.shape[0] == 1:
                residues_filtered.append(residue)
        return residues_filtered
   ##########################

    bound_predic_ligand_filtered = filter_residues(bound_ligand_residues)
    unbound_predic_ligand_filtered = bound_predic_ligand_filtered

    bound_predic_receptor_filtered = filter_residues(bound_receptor_residues)
    unbound_predic_receptor_filtered = bound_predic_receptor_filtered

    bound_predic_ligand_clean_list = bound_predic_ligand_filtered
    unbound_predic_ligand_clean_list = unbound_predic_ligand_filtered

    bound_predic_receptor_clean_list= bound_predic_receptor_filtered
    unbound_predic_receptor_clean_list = unbound_predic_receptor_filtered

    ###################
    def get_alphaC_loc_array(bound_predic_clean_list):
        bound_alphaC_loc_clean_list = []
        for residue in bound_predic_clean_list:
            df = residue[1]
            alphaCatom = df[df['atom_name'] == 'CA']
            alphaC_loc = alphaCatom[['x', 'y', 'z']].to_numpy().squeeze().astype(np.float32)
            assert alphaC_loc.shape == (3,), \
                f"alphac loc shape problem, shape: {alphaC_loc.shape} residue {df} resid {df['residue']}"
            bound_alphaC_loc_clean_list.append(alphaC_loc)
        if len(bound_alphaC_loc_clean_list) <= 1:
            bound_alphaC_loc_clean_list.append(np.zeros(3))
        return np.stack(bound_alphaC_loc_clean_list, axis=0)  # (N_res,3)

   ####################

    assert graph_nodes == 'residues'
    bound_receptor_repres_nodes_loc_array = get_alphaC_loc_array(bound_predic_receptor_clean_list)
    bound_ligand_repres_nodes_loc_array = get_alphaC_loc_array(bound_predic_ligand_clean_list)

    if not inference:

        # Keep pairs of ligand and receptor residues/atoms that have pairwise distances < threshold
        ligand_receptor_distance = spa.distance.cdist(bound_ligand_repres_nodes_loc_array, bound_receptor_repres_nodes_loc_array)
        positive_tuple = np.where(ligand_receptor_distance < pos_cutoff)
        active_ligand = positive_tuple[0]
        active_receptor = positive_tuple[1]
        if active_ligand.size <= 3: # We need: active_ligand.size > 0 '
            pocket_coors = None ## Will be filtered out later
        else:
            ligand_pocket_coors = bound_ligand_repres_nodes_loc_array[active_ligand, :]
            receptor_pocket_coors = bound_receptor_repres_nodes_loc_array[active_receptor, :]
            assert np.max(np.linalg.norm(ligand_pocket_coors - receptor_pocket_coors, axis=1)) <= pos_cutoff
            pocket_coors = 0.5 * (ligand_pocket_coors + receptor_pocket_coors)
            print('Num pocket nodes = ', len(active_ligand), ' total nodes = ', bound_ligand_repres_nodes_loc_array.shape[0], ' graph_nodes = ', graph_nodes)

        return unbound_predic_ligand_clean_list, unbound_predic_receptor_clean_list, \
               bound_ligand_repres_nodes_loc_array, bound_receptor_repres_nodes_loc_array, \
               pocket_coors

    return unbound_predic_ligand_clean_list, unbound_predic_receptor_clean_list, \
           bound_ligand_repres_nodes_loc_array, bound_receptor_repres_nodes_loc_array




def protein_to_graph_unbound_bound(unbound_ligand_predic,
                                   unbound_receptor_predic,
                                   bound_ligand_repres_nodes_loc_clean_array,  # (N_res, 3) np array of coordinates
                                   bound_receptor_repres_nodes_loc_clean_array,  # (N_res, 3) np array
                                   graph_nodes,
                                   cutoff=20,
                                   max_neighbor=None,
                                   one_hot=False,
                                   residue_loc_is_alphaC=True):

    return protein_to_graph_unbound_bound_residuesonly(unbound_ligand_predic,
                                                       unbound_receptor_predic,
                                                       bound_ligand_repres_nodes_loc_clean_array,
                                                       bound_receptor_repres_nodes_loc_clean_array,
                                                       cutoff,
                                                       max_neighbor,
                                                       one_hot,
                                                       residue_loc_is_alphaC)



def protein_to_graph_unbound_bound_residuesonly(unbound_ligand_predic,
                                                unbound_receptor_predic,
                                                bound_ligand_repres_nodes_loc_clean_array,  # (N_res, 3) np array of coordinates
                                                bound_receptor_repres_nodes_loc_clean_array,  # (N_res, 3) np array
                                                cutoff=20,
                                                max_neighbor=None,
                                                one_hot=False,
                                                residue_loc_is_alphaC=True):


    ################## Extract 3D coordinates and n_i,u_i,v_i vectors of representative residues ################
    def l_or_r_extract_3d_coord_and_n_u_v_vecs(l_or_r_predic):
        l_or_r_all_atom_coords_in_residue_list = []
        l_or_r_residue_representatives_loc_list = []
        l_or_r_n_i_list = []
        l_or_r_u_i_list = []
        l_or_r_v_i_list = []

        for residue in l_or_r_predic:
            df = residue[1]
            coord = df[['x', 'y', 'z']].to_numpy().astype(np.float32)  # (N_atoms, 3)
            l_or_r_all_atom_coords_in_residue_list.append(coord)

            Natom = df[df['atom_name'] == 'N']
            alphaCatom = df[df['atom_name'] == 'CA']
            Catom = df[df['atom_name'] == 'C']

            if Natom.shape[0] != 1 or alphaCatom.shape[0] != 1 or Catom.shape[0] != 1:
                print(df.iloc[0, :])
                raise ValueError("protein utils protein_to_graph_unbound_bound, no N/CA/C exists")

            N_loc = Natom[['x', 'y', 'z']].to_numpy().squeeze().astype(np.float32)
            alphaC_loc = alphaCatom[['x', 'y', 'z']].to_numpy().squeeze().astype(np.float32)
            C_loc = Catom[['x', 'y', 'z']].to_numpy().squeeze().astype(np.float32)

            u_i = (N_loc - alphaC_loc) / LA.norm(N_loc - alphaC_loc)
            t_i = (C_loc - alphaC_loc) / LA.norm(C_loc - alphaC_loc)
            n_i = np.cross(u_i, t_i) / LA.norm(np.cross(u_i, t_i))
            v_i = np.cross(n_i, u_i)
            assert (math.fabs(LA.norm(v_i) - 1.) < 1e-5), "protein utils protein_to_graph_dips, v_i norm larger than 1"

            l_or_r_n_i_list.append(n_i)
            l_or_r_u_i_list.append(u_i)
            l_or_r_v_i_list.append(v_i)

            if residue_loc_is_alphaC:
                l_or_r_residue_representatives_loc_list.append(alphaC_loc)
            else:
                heavy_df = df[df['element'] != 'H']
                residue_loc = heavy_df[['x', 'y', 'z']].mean(axis=0).to_numpy().astype(np.float32) # average of all atom coordinates
                l_or_r_residue_representatives_loc_list.append(residue_loc)

        l_or_r_residue_representatives_loc_feat = np.stack(l_or_r_residue_representatives_loc_list, axis=0)  # (N_res, 3)
        l_or_r_n_i_feat = np.stack(l_or_r_n_i_list, axis=0)
        l_or_r_u_i_feat = np.stack(l_or_r_u_i_list, axis=0)
        l_or_r_v_i_feat = np.stack(l_or_r_v_i_list, axis=0)

        l_or_r_num_residues = len(l_or_r_predic)
        if l_or_r_num_residues <= 1:
            raise ValueError(f"l_or_r contains only 1 residue!")
        return l_or_r_all_atom_coords_in_residue_list, \
               l_or_r_residue_representatives_loc_feat, \
               l_or_r_n_i_feat, l_or_r_u_i_feat, l_or_r_v_i_feat, l_or_r_num_residues


    (ligand_all_atom_coords_in_residue_list, # list of (N_atoms,3) arrays, for each residue
    ligand_residue_representatives_loc_feat, # (N_res, 3)
    ligand_n_i_feat, # (N_res, 3)
    ligand_u_i_feat, # (N_res, 3)
    ligand_v_i_feat, # (N_res, 3)
    ligand_num_residues) = l_or_r_extract_3d_coord_and_n_u_v_vecs(unbound_ligand_predic)


    (receptor_all_atom_coords_in_residue_list,
    receptor_residue_representatives_loc_feat,
    receptor_n_i_feat,
    receptor_u_i_feat,
    receptor_v_i_feat,
    receptor_num_residues) = l_or_r_extract_3d_coord_and_n_u_v_vecs(unbound_receptor_predic)

    ################# Align unbound and bound structures, if needed ################################
    def l_or_r_align_unbound_and_bound(l_or_r_residue_representatives_loc_feat,
                                       l_or_r_n_i_feat,
                                       l_or_r_u_i_feat,
                                       l_or_r_v_i_feat,
                                       bound_l_or_r_alphaC_loc_clean_array):

        ret_R_l_or_r, ret_t_l_or_r = rigid_transform_Kabsch_3D(l_or_r_residue_representatives_loc_feat.T,
                                                               bound_l_or_r_alphaC_loc_clean_array.T)
        l_or_r_residue_representatives_loc_feat = ((ret_R_l_or_r @ (l_or_r_residue_representatives_loc_feat).T) +
                                                   ret_t_l_or_r).T
        l_or_r_n_i_feat = ((ret_R_l_or_r @ (l_or_r_n_i_feat).T)).T
        l_or_r_u_i_feat = ((ret_R_l_or_r @ (l_or_r_u_i_feat).T)).T
        l_or_r_v_i_feat = ((ret_R_l_or_r @ (l_or_r_v_i_feat).T)).T
        return l_or_r_residue_representatives_loc_feat, l_or_r_n_i_feat, l_or_r_u_i_feat, l_or_r_v_i_feat

    (ligand_residue_representatives_loc_feat,
     ligand_n_i_feat,
     ligand_u_i_feat,
     ligand_v_i_feat) = l_or_r_align_unbound_and_bound(ligand_residue_representatives_loc_feat,
                                                       ligand_n_i_feat, ligand_u_i_feat, ligand_v_i_feat,
                                                       bound_ligand_repres_nodes_loc_clean_array)
    (receptor_residue_representatives_loc_feat,
     receptor_n_i_feat,
     receptor_u_i_feat,
     receptor_v_i_feat) = l_or_r_align_unbound_and_bound(receptor_residue_representatives_loc_feat,
                                                         receptor_n_i_feat, receptor_u_i_feat, receptor_v_i_feat,
                                                         bound_receptor_repres_nodes_loc_clean_array)

    ################### Build the k-NN graph ##############################
    def compute_dig_kNN_graph(l_or_r_num_residues,
                              l_or_r_all_atom_coords_in_residue_list,
                              unbound_l_or_r_predic,
                              l_or_r_residue_representatives_loc_feat,
                              l_or_r_n_i_feat,
                              l_or_r_u_i_feat,
                              l_or_r_v_i_feat):

        assert l_or_r_num_residues == l_or_r_residue_representatives_loc_feat.shape[0]
        assert l_or_r_residue_representatives_loc_feat.shape[1] == 3

        l_or_r_distance = np.full((l_or_r_num_residues, l_or_r_num_residues), np.inf)

        for i in range(l_or_r_num_residues - 1):
            for j in range((i + 1), l_or_r_num_residues):
                l_or_r_pairwise_dis = spa.distance.cdist(l_or_r_all_atom_coords_in_residue_list[i],
                                                         l_or_r_all_atom_coords_in_residue_list[j])
                l_or_r_distance[i, j] = np.mean(l_or_r_pairwise_dis)
                l_or_r_distance[j, i] = np.mean(l_or_r_pairwise_dis)

        l_or_r_protein_graph = dgl.graph(([], []), idtype=torch.int32)
        l_or_r_protein_graph.add_nodes(l_or_r_num_residues)

        l_or_r_src_list = []
        l_or_r_dst_list = []
        l_or_r_dist_list = []
        l_or_r_mean_norm_list = []

        for i in range(l_or_r_num_residues):
            valid_src = list(np.where(l_or_r_distance[i, :] < cutoff)[0])
            assert i not in valid_src
            if len(valid_src) > max_neighbor:
                valid_src = list(np.argsort(l_or_r_distance[i, :]))[0 : max_neighbor]
            valid_dst = [i] * len(valid_src)
            l_or_r_dst_list.extend(valid_dst)
            l_or_r_src_list.extend(valid_src)

            valid_dist = list(l_or_r_distance[i, valid_src])
            l_or_r_dist_list.extend(valid_dist)

            valid_dist_np = l_or_r_distance[i, valid_src]
            sigma = np.array([1., 2., 5., 10., 30.]).reshape((-1, 1))
            weights = softmax(- valid_dist_np.reshape((1, -1)) ** 2 / sigma, axis=1)  # (sigma_num, neigh_num)
            assert weights[0].sum() > 1 - 1e-2 and weights[0].sum() < 1.01
            diff_vecs = l_or_r_residue_representatives_loc_feat[valid_dst, :] - l_or_r_residue_representatives_loc_feat[valid_src, :]  # (neigh_num, 3)
            mean_vec = weights.dot(diff_vecs)  # (sigma_num, 3)
            denominator = weights.dot(np.linalg.norm(diff_vecs, axis=1))  # (sigma_num,)
            mean_vec_ratio_norm = np.linalg.norm(mean_vec, axis=1) / denominator  # (sigma_num,)
            l_or_r_mean_norm_list.append(mean_vec_ratio_norm)

        assert len(l_or_r_src_list) == len(l_or_r_dst_list)
        assert len(l_or_r_dist_list) == len(l_or_r_dst_list)
        l_or_r_protein_graph.add_edges(torch.IntTensor(l_or_r_src_list), torch.IntTensor(l_or_r_dst_list))

        if one_hot:
            l_or_r_protein_graph.ndata.update(residue_list_featurizer_dips_one_hot(unbound_l_or_r_predic))
        else:
            l_or_r_protein_graph.ndata.update(residue_list_featurizer_dips_NOT_one_hot(unbound_l_or_r_predic))

        l_or_r_protein_graph.edata.update(distance_list_featurizer(l_or_r_dist_list))


        # Loop over all edges of the graph and build the various p_ij, q_ij, k_ij, t_ij pairs
        l_or_r_edge_feat_ori_list = []
        for i in range(len(l_or_r_dist_list)):
            src = l_or_r_src_list[i]
            dst = l_or_r_dst_list[i]

            # place n_i, u_i, v_i as lines in a 3x3 basis matrix
            basis_matrix = np.stack((l_or_r_n_i_feat[dst, :], l_or_r_u_i_feat[dst, :], l_or_r_v_i_feat[dst, :]), axis=0)
            p_ij = np.matmul(basis_matrix,
                             l_or_r_residue_representatives_loc_feat[src, :] - l_or_r_residue_representatives_loc_feat[dst, :])
            q_ij = np.matmul(basis_matrix, l_or_r_n_i_feat[src, :])  # shape (3,)
            k_ij = np.matmul(basis_matrix, l_or_r_u_i_feat[src, :])
            t_ij = np.matmul(basis_matrix, l_or_r_v_i_feat[src, :])
            s_ij = np.concatenate((p_ij, q_ij, k_ij, t_ij), axis=0)   # shape (12,)
            l_or_r_edge_feat_ori_list.append(s_ij)
        l_or_r_edge_feat_ori_feat = np.stack(l_or_r_edge_feat_ori_list, axis=0) # shape (num_edges, 4, 3)
        l_or_r_edge_feat_ori_feat = zerocopy_from_numpy(l_or_r_edge_feat_ori_feat.astype(np.float32))
        l_or_r_protein_graph.edata['he'] = torch.cat((l_or_r_protein_graph.edata['he'], l_or_r_edge_feat_ori_feat), axis=1)  # (num_edges, 17)


        l_or_r_residue_representatives_loc_feat = zerocopy_from_numpy(l_or_r_residue_representatives_loc_feat.astype(np.float32))
        l_or_r_protein_graph.ndata['x'] = l_or_r_residue_representatives_loc_feat
        l_or_r_protein_graph.ndata['mu_r_norm'] = zerocopy_from_numpy(np.array(l_or_r_mean_norm_list).astype(np.float32))

        return l_or_r_protein_graph


    ligand_protein_graph = compute_dig_kNN_graph(ligand_num_residues,
                                                 ligand_all_atom_coords_in_residue_list,
                                                 unbound_ligand_predic,
                                                 ligand_residue_representatives_loc_feat,
                                                 ligand_n_i_feat,
                                                 ligand_u_i_feat,
                                                 ligand_v_i_feat)

    receptor_protein_graph = compute_dig_kNN_graph(receptor_num_residues,
                                                   receptor_all_atom_coords_in_residue_list,
                                                   unbound_receptor_predic,
                                                   receptor_residue_representatives_loc_feat,
                                                   receptor_n_i_feat,
                                                   receptor_u_i_feat,
                                                   receptor_v_i_feat)

    return ligand_protein_graph, receptor_protein_graph


