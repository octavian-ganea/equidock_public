import math

import dgl
import torch
from torch import nn
from dgl import function as fn
from src.utils.graph_norm import GraphNorm
import sys

def get_non_lin(type, negative_slope):
    if type == 'swish':
        return nn.SiLU()
    else:
        assert type == 'lkyrelu'
        return nn.LeakyReLU(negative_slope=negative_slope)


def get_layer_norm(layer_norm_type, dim):
    if layer_norm_type == 'BN':
        return nn.BatchNorm1d(dim)
    elif layer_norm_type == 'LN':
        return nn.LayerNorm(dim)
    else:
        return nn.Identity()


def get_final_h_layer_norm(layer_norm_type, dim):
    if layer_norm_type == 'BN':
        return nn.BatchNorm1d(dim)
    elif layer_norm_type == 'LN':
        return nn.LayerNorm(dim)
    elif layer_norm_type == 'GN':
        return GraphNorm(dim)
    else:
        assert layer_norm_type == '0'
        return nn.Identity()


def apply_final_h_layer_norm(g, h, node_type, norm_type, norm_layer):
    if norm_type == 'GN':
        return norm_layer(g, h, node_type)
    return norm_layer(h)



def compute_cross_attention(queries, keys, values, mask, cross_msgs):
    """Compute cross attention.
    x_i attend to y_j:
    a_{i->j} = exp(sim(x_i, y_j)) / sum_j exp(sim(x_i, y_j))
    attention_x = sum_j a_{i->j} y_j
    Args:
      queries: NxD float tensor --> queries
      keys: MxD float tensor --> keys
      values: Mxd
      mask: NxM
    Returns:
      attention_x: Nxd float tensor.
    """
    if not cross_msgs:
        return queries * 0.
    a = mask * torch.mm(queries, torch.transpose(keys, 1, 0)) - 1000. * (1. - mask)
    a_x = torch.softmax(a, dim=1)  # i->j, NxM, a_x.sum(dim=1) = torch.ones(N)
    attention_x = torch.mm(a_x, values)  # (N,d)
    return attention_x



def get_mask(ligand_batch_num_nodes, receptor_batch_num_nodes, device):
    rows = sum(ligand_batch_num_nodes)
    cols = sum(receptor_batch_num_nodes)
    mask = torch.zeros(rows, cols).to(device)
    partial_l = 0
    partial_r = 0
    for l_n, r_n in zip(ligand_batch_num_nodes, receptor_batch_num_nodes):
        mask[partial_l: partial_l + l_n, partial_r: partial_r + r_n] = 1
        partial_l = partial_l + l_n
        partial_r = partial_r + r_n
    return mask



class IEGMN_Layer(nn.Module):
    def __init__(
            self,
            orig_h_feats_dim,
            h_feats_dim,  # in dim of h
            out_feats_dim,  # out dim of h
            fine_tune,
            args,
            log=None
    ):

        super(IEGMN_Layer, self).__init__()

        input_edge_feats_dim = args['input_edge_feats_dim']
        dropout = args['dropout']
        nonlin = args['nonlin']
        self.cross_msgs = args['cross_msgs']
        layer_norm = args['layer_norm']
        layer_norm_coors = args['layer_norm_coors']
        self.final_h_layer_norm = args['final_h_layer_norm']
        self.use_dist_in_layers = args['use_dist_in_layers']
        self.skip_weight_h = args['skip_weight_h']
        self.x_connection_init = args['x_connection_init']
        leakyrelu_neg_slope = args['leakyrelu_neg_slope']

        self.fine_tune = fine_tune

        self.debug = args['debug']
        self.device = args['device']
        self.log = log

        self.h_feats_dim = h_feats_dim
        self.out_feats_dim = out_feats_dim

        self.all_sigmas_dist = [1.5 ** x for x in range(15)]

        # EDGES
        self.edge_mlp = nn.Sequential(
            nn.Linear((h_feats_dim * 2) + input_edge_feats_dim + len(self.all_sigmas_dist), self.out_feats_dim),
            nn.Dropout(dropout),
            get_non_lin(nonlin, leakyrelu_neg_slope),
            get_layer_norm(layer_norm, self.out_feats_dim),
            nn.Linear(self.out_feats_dim, self.out_feats_dim),
        )

        # NODES
        self.node_norm = nn.Identity() # nn.LayerNorm(h_feats_dim)

        self.att_mlp_Q = nn.Sequential(
            nn.Linear(h_feats_dim, h_feats_dim, bias=False),
            get_non_lin(nonlin, leakyrelu_neg_slope),
        )
        self.att_mlp_K = nn.Sequential(
            nn.Linear(h_feats_dim, h_feats_dim, bias=False),
            get_non_lin(nonlin, leakyrelu_neg_slope),
        )
        self.att_mlp_V = nn.Sequential(
            nn.Linear(h_feats_dim, h_feats_dim, bias=False),
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(orig_h_feats_dim + 2 * h_feats_dim + self.out_feats_dim, h_feats_dim),
            nn.Dropout(dropout),
            get_non_lin(nonlin, leakyrelu_neg_slope),
            get_layer_norm(layer_norm, h_feats_dim),
            nn.Linear(h_feats_dim, out_feats_dim),
        )

        self.final_h_layernorm_layer = get_final_h_layer_norm(self.final_h_layer_norm, out_feats_dim)

        ## The scalar weight to be multiplied by (x_i - x_j)
        self.coors_mlp = nn.Sequential(
            nn.Linear(self.out_feats_dim, self.out_feats_dim),
            nn.Dropout(dropout),
            get_non_lin(nonlin, leakyrelu_neg_slope),
            get_layer_norm(layer_norm_coors, self.out_feats_dim),
            nn.Linear(self.out_feats_dim, 1)
        )

        if self.fine_tune:
            self.att_mlp_cross_coors_Q = nn.Sequential(
                nn.Linear(h_feats_dim, h_feats_dim, bias=False),
                get_non_lin(nonlin, leakyrelu_neg_slope),
            )
            self.att_mlp_cross_coors_K = nn.Sequential(
                nn.Linear(h_feats_dim, h_feats_dim, bias=False),
                get_non_lin(nonlin, leakyrelu_neg_slope),
            )
            self.att_mlp_cross_coors_V = nn.Sequential(
                nn.Linear(h_feats_dim, h_feats_dim),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                nn.Linear(h_feats_dim, 1),
            )
        # self.reset_parameters()


    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p, gain=1.)
            else:
                torch.nn.init.zeros_(p)

    def apply_edges1(self, edges):
        return {'cat_feat': torch.cat([edges.src['feat'], edges.dst['feat']], dim=1)}


    def forward(self, hetero_graph,
                coors_ligand, h_feats_ligand, original_ligand_node_features, original_edge_feats_ligand, orig_coors_ligand,
                coors_receptor, h_feats_receptor, original_receptor_node_features, original_edge_feats_receptor, orig_coors_receptor):

        with hetero_graph.local_scope():
            hetero_graph.nodes['ligand'].data['x_now'] = coors_ligand
            hetero_graph.nodes['receptor'].data['x_now'] = coors_receptor
            hetero_graph.nodes['ligand'].data['feat'] = h_feats_ligand  # first time set here
            hetero_graph.nodes['receptor'].data['feat'] = h_feats_receptor

            if self.debug:
                self.log(torch.max(hetero_graph.nodes['ligand'].data['x_now']), 'x_now : x_i at layer entrance')
                self.log(torch.max(hetero_graph.nodes['ligand'].data['feat']), 'data[feat] = h_i at layer entrance')


            hetero_graph.apply_edges(fn.u_sub_v('x_now', 'x_now', 'x_rel'), etype=('ligand', 'll', 'ligand')) ## x_i - x_j
            hetero_graph.apply_edges(fn.u_sub_v('x_now', 'x_now', 'x_rel'), etype=('receptor', 'rr', 'receptor'))


            x_rel_mag_ligand = hetero_graph.edges[('ligand', 'll', 'ligand')].data['x_rel'] ** 2
            x_rel_mag_ligand = torch.sum(x_rel_mag_ligand, dim=1, keepdim=True)   ## ||x_i - x_j||^2 : (N_res, 1)
            x_rel_mag_ligand = torch.cat([torch.exp(-x_rel_mag_ligand / sigma) for sigma in self.all_sigmas_dist], dim=-1)

            x_rel_mag_receptor = hetero_graph.edges[('receptor', 'rr', 'receptor')].data['x_rel'] ** 2
            x_rel_mag_receptor = torch.sum(x_rel_mag_receptor, dim=1, keepdim=True)
            x_rel_mag_receptor = torch.cat([torch.exp(-x_rel_mag_receptor / sigma) for sigma in self.all_sigmas_dist], dim=-1)

            if not self.use_dist_in_layers:
                x_rel_mag_ligand = x_rel_mag_ligand * 0.
                x_rel_mag_receptor = x_rel_mag_receptor * 0.

            if self.debug:
                self.log(torch.max(hetero_graph.edges[('ligand', 'll', 'ligand')].data['x_rel']), 'x_rel : x_i - x_j')
                self.log(torch.max(x_rel_mag_ligand,dim=0).values, 'x_rel_mag_ligand = [exp(-||x_i - x_j||^2 / sigma) for sigma = 1.5 ** x, x = [0, 15]]')



            hetero_graph.apply_edges(self.apply_edges1, etype=('ligand', 'll', 'ligand'))  ## i->j edge:  [h_i h_j]
            hetero_graph.apply_edges(self.apply_edges1, etype=('receptor', 'rr', 'receptor'))

            cat_input_for_msg_ligand = torch.cat((hetero_graph.edges['ll'].data['cat_feat'], # [h_i h_j]
                                                  original_edge_feats_ligand,
                                                  x_rel_mag_ligand), dim=-1)
            cat_input_for_msg_receptor = torch.cat((hetero_graph.edges['rr'].data['cat_feat'],
                                                    original_edge_feats_receptor,
                                                    x_rel_mag_receptor), dim=-1)

            hetero_graph.edges['ll'].data['msg'] = self.edge_mlp(cat_input_for_msg_ligand)  # m_{i->j}
            hetero_graph.edges['rr'].data['msg'] = self.edge_mlp(cat_input_for_msg_receptor)

            if self.debug:
                self.log(torch.max(hetero_graph.edges['ll'].data['msg']), 'data[msg] = m_{i->j} = phi^e(h_i, h_j, f_{i,j}, x_rel_mag_ligand)')



            mask = get_mask(hetero_graph.batch_num_nodes('ligand'), hetero_graph.batch_num_nodes('receptor'), self.device)

            # \mu_i
            hetero_graph.nodes['ligand'].data['aggr_cross_msg'] = compute_cross_attention(self.att_mlp_Q(h_feats_ligand),
                                                                                          self.att_mlp_K(h_feats_receptor),
                                                                                          self.att_mlp_V(h_feats_receptor),
                                                                                          mask,
                                                                                          self.cross_msgs)
            hetero_graph.nodes['receptor'].data['aggr_cross_msg'] = compute_cross_attention(self.att_mlp_Q(h_feats_receptor),
                                                                                            self.att_mlp_K(h_feats_ligand),
                                                                                            self.att_mlp_V(h_feats_ligand),
                                                                                            mask.transpose(0,1),
                                                                                            self.cross_msgs)

            if self.debug:
                self.log(torch.max(hetero_graph.nodes['ligand'].data['aggr_cross_msg']), 'aggr_cross_msg(i) = sum_j a_{i,j} * h_j')



            edge_coef_ligand = self.coors_mlp(hetero_graph.edges['ll'].data['msg'])  # \phi^x(m_{i->j})
            hetero_graph.edges['ll'].data['x_moment'] = hetero_graph.edges['ll'].data['x_rel'] * edge_coef_ligand # (x_i - x_j) * \phi^x(m_{i->j})
            edge_coef_receptor = self.coors_mlp(hetero_graph.edges['rr'].data['msg'])
            hetero_graph.edges['rr'].data['x_moment'] = hetero_graph.edges['rr'].data['x_rel'] * edge_coef_receptor


            if self.debug:
                self.log(torch.max(edge_coef_ligand), 'edge_coef_ligand : \phi^x(m_{i->j})')
                self.log(torch.max(hetero_graph.edges['ll'].data['x_moment']), 'data[x_moment] = (x_i - x_j) * \phi^x(m_{i->j})')


            hetero_graph.update_all(fn.copy_e('x_moment', 'm'), fn.mean('m', 'x_update'),
                                    etype=('ligand', 'll', 'ligand'))
            hetero_graph.update_all(fn.copy_e('x_moment', 'm'), fn.mean('m', 'x_update'),
                                    etype=('receptor', 'rr', 'receptor'))


            hetero_graph.update_all(fn.copy_e('msg', 'm'), fn.mean('m', 'aggr_msg'),
                                    etype=('ligand', 'll', 'ligand'))
            hetero_graph.update_all(fn.copy_e('msg', 'm'), fn.mean('m', 'aggr_msg'),
                                    etype=('receptor', 'rr', 'receptor'))


            x_final_ligand = self.x_connection_init * orig_coors_ligand + \
                             (1. - self.x_connection_init) * hetero_graph.nodes['ligand'].data['x_now'] + \
                             hetero_graph.nodes['ligand'].data['x_update']

            x_final_receptor = self.x_connection_init * orig_coors_receptor + \
                               (1. - self.x_connection_init) * hetero_graph.nodes['receptor'].data['x_now'] + \
                               hetero_graph.nodes['receptor'].data['x_update']

            if self.fine_tune:
                x_final_ligand = x_final_ligand + \
                                 self.att_mlp_cross_coors_V(h_feats_ligand) * (
                                         hetero_graph.nodes['ligand'].data['x_now'] -
                                         compute_cross_attention(self.att_mlp_cross_coors_Q(h_feats_ligand),
                                                                 self.att_mlp_cross_coors_K(h_feats_receptor),
                                                                 hetero_graph.nodes['receptor'].data['x_now'],
                                                                 mask,
                                                                 self.cross_msgs))
                x_final_receptor = x_final_receptor + \
                                   self.att_mlp_cross_coors_V(h_feats_receptor) * (
                                           hetero_graph.nodes['receptor'].data['x_now'] -
                                           compute_cross_attention(self.att_mlp_cross_coors_Q(h_feats_receptor),
                                                                   self.att_mlp_cross_coors_K(h_feats_ligand),
                                                                   hetero_graph.nodes['ligand'].data['x_now'],
                                                                   mask.transpose(0,1),
                                                                   self.cross_msgs))


            if self.debug:
                self.log(torch.max(hetero_graph.nodes['ligand'].data['aggr_msg']), 'data[aggr_msg]: \sum_j m_{i->j} ')
                self.log(torch.max(hetero_graph.nodes['ligand'].data['x_update']), 'data[x_update] : \sum_j (x_i - x_j) * \phi^x(m_{i->j})')
                self.log(torch.max(x_final_ligand), 'x_i new = x_final_ligand : x_i + data[x_update]')


            input_node_upd_ligand = torch.cat((self.node_norm(hetero_graph.nodes['ligand'].data['feat']),
                                               hetero_graph.nodes['ligand'].data['aggr_msg'],
                                               hetero_graph.nodes['ligand'].data['aggr_cross_msg'],
                                               original_ligand_node_features),
                                              dim=-1)

            input_node_upd_receptor = torch.cat((self.node_norm(hetero_graph.nodes['receptor'].data['feat']),
                                                 hetero_graph.nodes['receptor'].data['aggr_msg'],
                                                 hetero_graph.nodes['receptor'].data['aggr_cross_msg'],
                                                 original_receptor_node_features),
                                                dim=-1)

            # Skip connections
            if self.h_feats_dim == self.out_feats_dim:
                node_upd_ligand = self.skip_weight_h * self.node_mlp(input_node_upd_ligand) + (1. - self.skip_weight_h) * h_feats_ligand
                node_upd_receptor = self.skip_weight_h * self.node_mlp(input_node_upd_receptor) + (1. - self.skip_weight_h) * h_feats_receptor
            else:
                node_upd_ligand = self.node_mlp(input_node_upd_ligand)
                node_upd_receptor = self.node_mlp(input_node_upd_receptor)

            if self.debug:
                self.log('node_mlp params')
                for p in self.node_mlp.parameters():
                    print(p)
                self.log(torch.max(input_node_upd_ligand), 'concat(h_i, aggr_msg, aggr_cross_msg)')
                self.log(torch.max(node_upd_ligand), 'h_i new = h_i + MLP(h_i, aggr_msg, aggr_cross_msg)')



            node_upd_ligand = apply_final_h_layer_norm(hetero_graph, node_upd_ligand, 'ligand', self.final_h_layer_norm, self.final_h_layernorm_layer)
            node_upd_receptor = apply_final_h_layer_norm(hetero_graph, node_upd_receptor, 'receptor', self.final_h_layer_norm, self.final_h_layernorm_layer)


            return x_final_ligand, node_upd_ligand, x_final_receptor, node_upd_receptor


    def __repr__(self):
        return "IEGMN Layer " + str(self.__dict__)


# =================================================================================================================
class IEGMN(nn.Module):

    def __init__(self, args, n_lays, fine_tune, log=None):

        super(IEGMN, self).__init__()

        self.debug = args['debug']
        self.log=log

        self.device = args['device']
        self.graph_nodes = args['graph_nodes']

        self.rot_model = args['rot_model']

        self.noise_decay_rate = args['noise_decay_rate']
        self.noise_initial = args['noise_initial']

        self.use_edge_features_in_gmn = args['use_edge_features_in_gmn']

        self.use_mean_node_features = args['use_mean_node_features']

        # 21 types of amino-acid types
        self.residue_emb_layer = nn.Embedding(num_embeddings=21, embedding_dim=args['residue_emb_dim'])

        assert self.graph_nodes == 'residues'
        input_node_feats_dim = args['residue_emb_dim'] ## One residue type

        if self.use_mean_node_features:
            input_node_feats_dim += 5 ### Additional features from mu_r_norm

        self.iegmn_layers = nn.ModuleList()

        self.iegmn_layers.append(
            IEGMN_Layer(orig_h_feats_dim=input_node_feats_dim,
                        h_feats_dim=input_node_feats_dim,
                        out_feats_dim=args['iegmn_lay_hid_dim'],
                        fine_tune=fine_tune,
                        args=args,
                        log=self.log))

        if args['shared_layers']:
            interm_lay = IEGMN_Layer(orig_h_feats_dim=input_node_feats_dim,
                                     h_feats_dim=args['iegmn_lay_hid_dim'],
                                     out_feats_dim=args['iegmn_lay_hid_dim'],
                                     args=args,
                                     fine_tune=fine_tune,
                                     log=self.log)
            for layer_idx in range(1, n_lays):
                self.iegmn_layers.append(interm_lay)

        else:
            for layer_idx in range(1, n_lays):
                self.iegmn_layers.append(
                    IEGMN_Layer(orig_h_feats_dim=input_node_feats_dim,
                                h_feats_dim=args['iegmn_lay_hid_dim'],
                                out_feats_dim=args['iegmn_lay_hid_dim'],
                                args=args,
                                fine_tune=fine_tune,
                                log=self.log))


        assert args['rot_model'] == 'kb_att'

        # Attention layers
        self.num_att_heads = args['num_att_heads']
        self.out_feats_dim = args['iegmn_lay_hid_dim']

        self.att_mlp_key_ROT = nn.Sequential(
            nn.Linear(self.out_feats_dim, self.num_att_heads * self.out_feats_dim, bias=False),
        )
        self.att_mlp_query_ROT = nn.Sequential(
            nn.Linear(self.out_feats_dim, self.num_att_heads * self.out_feats_dim, bias=False),
        )

        self.mlp_h_mean_ROT = nn.Sequential(
            nn.Linear(self.out_feats_dim, self.out_feats_dim),
            nn.Dropout(args['dropout']),
            get_non_lin(args['nonlin'], args['leakyrelu_neg_slope']),
        )

        # self.reset_parameters()


    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p, gain=1.)
            else:
                torch.nn.init.zeros_(p)


    def forward(self, batch_hetero_graph, epoch):
        orig_coors_ligand = batch_hetero_graph.nodes['ligand'].data['new_x']
        orig_coors_receptor = batch_hetero_graph.nodes['receptor'].data['x']

        coors_ligand = batch_hetero_graph.nodes['ligand'].data['new_x']
        coors_receptor = batch_hetero_graph.nodes['receptor'].data['x']

        ## Embed residue types with a lookup table.
        h_feats_ligand = self.residue_emb_layer(
            batch_hetero_graph.nodes['ligand'].data['res_feat'].view(-1).long())  # (N_res, emb_dim)
        h_feats_receptor = self.residue_emb_layer(
            batch_hetero_graph.nodes['receptor'].data['res_feat'].view(-1).long())  # (N_res, emb_dim)

        if self.debug:
            self.log(torch.max(h_feats_ligand), 'h_feats_ligand before layers ')

        if self.use_mean_node_features:
            h_feats_ligand = torch.cat([h_feats_ligand,
                                        torch.log(batch_hetero_graph.nodes['ligand'].data['mu_r_norm'])], dim=1)
            h_feats_receptor = torch.cat([h_feats_receptor,
                                          torch.log(batch_hetero_graph.nodes['receptor'].data['mu_r_norm'])], dim=1)

        if self.debug:
            self.log(torch.max(h_feats_ligand), torch.norm(h_feats_ligand),
                     'h_feats_ligand before layers but after mu_r_norm')

        original_ligand_node_features = h_feats_ligand
        original_receptor_node_features = h_feats_receptor

        original_edge_feats_ligand = batch_hetero_graph.edges['ll'].data['he'] * self.use_edge_features_in_gmn
        original_edge_feats_receptor = batch_hetero_graph.edges['rr'].data['he'] * self.use_edge_features_in_gmn

        for i, layer in enumerate(self.iegmn_layers):
            if self.debug:
                self.log('layer ', i)

            coors_ligand, \
            h_feats_ligand, \
            coors_receptor, \
            h_feats_receptor = layer(hetero_graph=batch_hetero_graph,
                                     coors_ligand=coors_ligand,
                                     h_feats_ligand=h_feats_ligand,
                                     original_ligand_node_features=original_ligand_node_features,
                                     original_edge_feats_ligand=original_edge_feats_ligand,
                                     orig_coors_ligand=orig_coors_ligand,
                                     coors_receptor=coors_receptor,
                                     h_feats_receptor=h_feats_receptor,
                                     original_receptor_node_features=original_receptor_node_features,
                                     original_edge_feats_receptor=original_edge_feats_receptor,
                                     orig_coors_receptor=orig_coors_receptor
                                     )

        if self.debug:
            self.log(torch.max(h_feats_ligand), 'h_feats_ligand after MPNN')
            self.log(torch.max(coors_ligand), 'coors_ligand before after MPNN')

        batch_hetero_graph.nodes['ligand'].data['x_iegmn_out'] = coors_ligand
        batch_hetero_graph.nodes['receptor'].data['x_iegmn_out'] = coors_receptor
        batch_hetero_graph.nodes['ligand'].data['hv_iegmn_out'] = h_feats_ligand
        batch_hetero_graph.nodes['receptor'].data['hv_iegmn_out'] = h_feats_receptor

        list_hetero_graph = dgl.unbatch(batch_hetero_graph)

        all_T_align_list = []
        all_b_align_list = []
        all_Y_receptor_att_ROT_list = []
        all_Y_ligand_att_ROT_list = []


        ### TODO: run SVD in batches, if possible
        for the_idx, hetero_graph in enumerate(list_hetero_graph):

            # Get H vectors
            H_receptor_feats = hetero_graph.nodes['receptor'].data['hv_iegmn_out'] # (m, d)
            H_receptor_feats_att_mean_ROT = torch.mean(self.mlp_h_mean_ROT(H_receptor_feats), dim=0, keepdim=True) # (1, d)


            H_ligand_feats = hetero_graph.nodes['ligand'].data['hv_iegmn_out'] # (n, d)
            H_ligand_feats_att_mean_ROT = torch.mean(self.mlp_h_mean_ROT(H_ligand_feats), dim=0, keepdim=True) # (1, d)

            d = H_ligand_feats.shape[1]
            assert d == self.out_feats_dim

            # Z coordinates
            Z_receptor_coors = hetero_graph.nodes['receptor'].data['x_iegmn_out']

            Z_ligand_coors = hetero_graph.nodes['ligand'].data['x_iegmn_out']


            #################### AP 1: compute two point clouds of K_heads points each, then do Kabsch  #########################
            # Att weights to compute the receptor centroid. They query is the average_h_ligand. Keys are each h_receptor_j
            att_weights_receptor_ROT = torch.softmax(
                self.att_mlp_key_ROT(H_receptor_feats).view(-1, self.num_att_heads, d).transpose(0, 1) @  # (K_heads, m_rec, d)
                self.att_mlp_query_ROT(H_ligand_feats_att_mean_ROT).view(1, self.num_att_heads, d).transpose(0, 1).transpose(1, 2) /  # (K_heads, d, 1)
                math.sqrt(d), # (K_heads, m_receptor, 1)
                dim=1).view(self.num_att_heads, -1)

            Y_receptor_att_ROT = att_weights_receptor_ROT @ Z_receptor_coors  # K_heads, 3
            all_Y_receptor_att_ROT_list.append(Y_receptor_att_ROT)


            # Att weights to compute the ligand centroid. They query is the average_h_receptor. Keys are each h_ligand_i
            att_weights_ligand_ROT = torch.softmax(
                self.att_mlp_key_ROT(H_ligand_feats).view(-1, self.num_att_heads, d).transpose(0, 1) @
                self.att_mlp_query_ROT(H_receptor_feats_att_mean_ROT).view(1, self.num_att_heads, d).transpose(0, 1).transpose(1, 2) /
                math.sqrt(d),  # (K_heads, n_ligand, 1)
                dim=1).view(self.num_att_heads, -1)

            Y_ligand_att_ROT = att_weights_ligand_ROT @ Z_ligand_coors  # K_heads, 3
            all_Y_ligand_att_ROT_list.append(Y_ligand_att_ROT)

            ## Apply Kabsch algorithm
            Y_receptor_att_ROT_mean = Y_receptor_att_ROT.mean(dim=0, keepdim=True) # (1,3)
            Y_ligand_att_ROT_mean = Y_ligand_att_ROT.mean(dim=0, keepdim=True) # (1,3)


            A = (Y_receptor_att_ROT - Y_receptor_att_ROT_mean).transpose(0,1) @ (Y_ligand_att_ROT - Y_ligand_att_ROT_mean) # 3, 3


            assert not torch.isnan(A).any()
            U, S, Vt = torch.linalg.svd(A)

            num_it = 0
            while torch.min(S) < 1e-3 or torch.min(torch.abs((S**2).view(1,3) - (S**2).view(3,1) + torch.eye(3).to(self.device))) < 1e-2:
                if self.debug:
                    self.log('S inside loop ', num_it, ' is ', S, ' and A = ', A)

                A = A + torch.rand(3,3).to(self.device) * torch.eye(3).to(self.device)
                U, S, Vt = torch.linalg.svd(A)
                num_it += 1

                if num_it > 10:
                    self.log('SVD consistently numerically unstable! Exitting ... ')
                    sys.exit(1)

            corr_mat = torch.diag(torch.Tensor([1,1,torch.sign(torch.det(A))])).to(self.device)
            T_align = (U @  corr_mat) @ Vt

            b_align = Y_receptor_att_ROT_mean - torch.t(T_align @ Y_ligand_att_ROT_mean.t())  # (1,3)


        #################### end AP 1 #########################

            if self.debug:
                self.log('Y_receptor_att_ROT_mean', Y_receptor_att_ROT_mean)
                self.log('Y_ligand_att_ROT_mean', Y_ligand_att_ROT_mean)


            all_T_align_list.append(T_align)
            all_b_align_list.append(b_align)

        return [all_T_align_list, all_b_align_list, all_Y_ligand_att_ROT_list, all_Y_receptor_att_ROT_list]


    def __repr__(self):
        return "IEGMN " + str(self.__dict__)

# =================================================================================================================


class Rigid_Body_Docking_Net(nn.Module):

    def __init__(self, args, log=None):

        super(Rigid_Body_Docking_Net, self).__init__()

        self.debug = args['debug']
        self.log=log

        self.device = args['device']

        self.iegmn_original = IEGMN(args, n_lays=args['iegmn_n_lays'], fine_tune=False, log=log)
        if args['fine_tune']:
            self.iegmn_fine_tune = IEGMN(args, n_lays=2, fine_tune=True, log=log)
            self.list_iegmns = [('original', self.iegmn_original), ('finetune', self.iegmn_fine_tune)]
        else:
            self.list_iegmns = [('finetune', self.iegmn_original)] ## just original

        # self.reset_parameters()


    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p, gain=1.)
            else:
                torch.nn.init.zeros_(p)



    ####### FORWARD for Rigid_Body_Docking_Net
    def forward(self, batch_hetero_graph, epoch):
        last_outputs = None
        all_ligand_coors_deform_list = []

        for stage, iegmn in self.list_iegmns:
            outputs = iegmn(batch_hetero_graph, epoch)
            assert len(outputs) == 4

            if stage == 'finetune':
                last_outputs = outputs

            list_hetero_graph = dgl.unbatch(batch_hetero_graph)
            if stage == 'original':
                new_list_hetero_graph = []

            for the_idx, hetero_graph in enumerate(list_hetero_graph):
                orig_coors_ligand = hetero_graph.nodes['ligand'].data['new_x']
                # orig_coors_receptor = hetero_graph.nodes['receptor'].data['x']

                T_align = outputs[0][the_idx]
                b_align = outputs[1][the_idx]
                assert b_align.shape[0] == 1 and b_align.shape[1] == 3

                inner_coors_ligand = ( T_align @ orig_coors_ligand.t() ).t() + b_align  # (n,3)

                if stage == 'original':
                    hetero_graph.nodes['ligand'].data['new_x'] = inner_coors_ligand
                    new_list_hetero_graph.append(hetero_graph)

                if self.debug:
                    self.log('T_align', T_align)
                    self.log('T_align @ T_align.t() - eye(3)', T_align @ T_align.t() - torch.eye(3).to(self.device))
                    self.log('b_align', b_align)
                    self.log('\n ---> inner_coors_ligand mean - true ligand mean ',
                             inner_coors_ligand.mean(dim=0) - hetero_graph.nodes['ligand'].data['x'].mean(dim=0), '\n')

                if stage == 'finetune':
                    all_ligand_coors_deform_list.append(inner_coors_ligand)

            if stage == 'original':
                batch_hetero_graph = dgl.batch(new_list_hetero_graph)


        all_keypts_ligand_list = last_outputs[2]
        all_keypts_receptor_list = last_outputs[3]
        all_rotation_list = last_outputs[0]
        all_translation_list = last_outputs[1]

        return all_ligand_coors_deform_list, \
               all_keypts_ligand_list, all_keypts_receptor_list, \
               all_rotation_list, all_translation_list


    def __repr__(self):
        return "Rigid_Body_Docking_Net " + str(self.__dict__)
