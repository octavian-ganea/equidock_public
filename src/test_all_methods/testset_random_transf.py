# File used to create the final test sets of 100 complexes (DIPS) and 25 complexes (DB5.5). These test sets created
# using this script can be found in test_sets_pbd/, so you don't have to run this script again.

import os
import random

from biopandas.pdb import PandasPdb
import numpy as np
import pandas as pd

from scipy.spatial.transform import Rotation

def UniformRotation_Translation(translation_interval):
    rotation = Rotation.random(num=1)
    rotation_matrix = rotation.as_matrix().squeeze()

    t = np.random.randn(1, 3)
    t = t / np.sqrt( np.sum(t * t))
    length = np.random.uniform(low=0, high=translation_interval)
    t = t * length
    return rotation_matrix.astype(np.float32), t.astype(np.float32)



def regen_ids_from_zero(ppdb, field):
    field_numbers = ppdb._df['ATOM'][field].to_numpy().copy()
    cur_id = 1
    cur_field_num = field_numbers[0]
    for v in range(field_numbers.shape[0]):
        if field_numbers[v] == cur_field_num:
            field_numbers[v] = cur_id
        else:
            cur_field_num = field_numbers[v]
            cur_id += 1
            field_numbers[v] = cur_id
    ppdb._df['ATOM'][field] = field_numbers
    return ppdb



def random_transf_pdb(ppdb, pdb_out_file, unchanged=False):
    if not unchanged:
        atom_loc = ppdb._df['ATOM'][['x_coord', 'y_coord', 'z_coord']].to_numpy().squeeze().astype(np.float32)
        assert np.mean(atom_loc, axis=0).shape[0] == 3
        # Randomly rotate and translate the ligand.
        rot_T, rot_b = UniformRotation_Translation(translation_interval=20.)
        atom_loc = atom_loc - np.mean(atom_loc, axis=0, keepdims=True)
        new_atom_loc = (rot_T @ atom_loc.T).T + rot_b

        ppdb._df['ATOM']['x_coord'] = new_atom_loc[:, 0]
        ppdb._df['ATOM']['y_coord'] = new_atom_loc[:, 1]
        ppdb._df['ATOM']['z_coord'] = new_atom_loc[:, 2]

        assert np.linalg.norm(new_atom_loc - atom_loc) > 0.1
        assert np.linalg.norm(ppdb._df['ATOM'][['x_coord', 'y_coord', 'z_coord']].to_numpy() - atom_loc) > 0.1

    ppdb = regen_ids_from_zero(ppdb, 'residue_number')
    ppdb = regen_ids_from_zero(ppdb, 'atom_number')

    ppdb.to_pdb(path=pdb_out_file,
                records=['ATOM'],
                gz=False,
                append_newline=True)



def main_dips():
    raw_data_path = './DIPS/data/DIPS/interim/pairs-pruned/'
    split_files_path = './DIPS/data/DIPS/interim/pairs-pruned/pairs-postprocessed-test.txt'
    final_dir = './DIPS/data/DIPS/test_pdb_random_transformed/'

    dill_filenames = {}
    with open(split_files_path, 'r') as f:
        for line in f.readlines():
            cat = line.rstrip().split('/')[0]
            if cat not in dill_filenames.keys():
                dill_filenames[cat] = []
            dill_filenames[cat].append(line.rstrip())

    all_cat = list(dill_filenames.keys())

    selected_files = set()
    idx = -1
    for i in range(100):
        file_to_add = ''
        while file_to_add == '':
            idx = (idx + 1) % len(all_cat)
            cat = all_cat[idx]
            choices = dill_filenames[cat]
            random.shuffle(choices)
            for file in choices:
                if file not in selected_files:
                    file_to_add = file
                    break
        selected_files.add(file_to_add)


    for dill_filename in selected_files:
        print(dill_filename)
        x = pd.read_pickle(os.path.join(raw_data_path, dill_filename))
        df0 = x.df0
        df1 = x.df1

        def dips_update_df(df):
            df.rename(columns={'aid': 'atom_number', 'atom_name': 'atom_name', 'chain': 'chain_id',
                                'residue': 'residue_number', 'resname': 'residue_name',
                                'x': 'x_coord', 'y': 'y_coord', 'z': 'z_coord', 'element': 'element_symbol'},
                       inplace=True)
            assert len(list(df['atom_number'])) == len(set(list(df['atom_number'])))
            df['record_name'] = 'ATOM'
            df['blank_1'] = ''
            df['alt_loc'] = ''
            df['blank_2'] = ''
            df['insertion'] = ''
            df['blank_3'] = ''
            df['blank_4'] = ''
            df['segment_id'] = ''
            df['charge'] = 0
            df['occupancy'] = 1.0
            df['b_factor'] = 0.0
            df['line_idx'] = list(range(len(list(df['residue_number']))))
            return df

        cols = ['record_name', 'atom_number', 'blank_1', 'atom_name', 'alt_loc', 'residue_name', 'blank_2',
                'chain_id', 'residue_number', 'insertion', 'blank_3', 'x_coord', 'y_coord', 'z_coord',
                'occupancy', 'b_factor', 'blank_4', 'segment_id', 'element_symbol', 'charge', 'line_idx']

        ppdb0 = PandasPdb()
        ppdb0._df['ATOM'] = dips_update_df(df0)
        ppdb0._df['ATOM'] = ppdb0._df['ATOM'][cols]

        ppdb1 = PandasPdb()
        ppdb1._df['ATOM'] = dips_update_df(df1)
        ppdb1._df['ATOM'] = ppdb1._df['ATOM'][cols]

        random_transf_pdb(ppdb0, os.path.join(final_dir, dill_filename.replace('/', '_') + '_l_b_COMPLEX.pdb'), unchanged=True)
        random_transf_pdb(ppdb1, os.path.join(final_dir, dill_filename.replace('/', '_') + '_r_b_COMPLEX.pdb'), unchanged=True)

        random_transf_pdb(ppdb0, os.path.join(final_dir, dill_filename.replace('/', '_') + '_l_b.pdb'))
        random_transf_pdb(ppdb1, os.path.join(final_dir, dill_filename.replace('/', '_') + '_r_b.pdb'))




def main_db5():
    raw_data_path = './data/benchmark5.5/structures'
    split_files_path = './data/benchmark5.5/cv/cv_0/test.txt'
    final_dir = './data/benchmark5.5/cv/cv_0/test_pdb_random_transformed/'


    onlyfiles = [f for f in os.listdir(raw_data_path) if os.path.isfile(os.path.join(raw_data_path, f))]
    code_set = set([file.split('_')[0] for file in onlyfiles])
    split_code_set = set()
    with open(os.path.join(split_files_path), 'r') as f:
        for line in f.readlines():
            split_code_set.add(line.rstrip())

    code_set = code_set & split_code_set
    code_list = list(code_set)
    print('Num pairs = ', len(code_set))

    for code in code_list:
        print(' code = ', code)

        random_transf_pdb(PandasPdb().read_pdb(os.path.join(raw_data_path, code + '_l_b.pdb')),
                          os.path.join(final_dir, code + '_l_b_COMPLEX.pdb'), unchanged=True)
        random_transf_pdb(PandasPdb().read_pdb(os.path.join(raw_data_path, code + '_r_b.pdb')),
                          os.path.join(final_dir, code + '_r_b_COMPLEX.pdb'), unchanged=True)

        random_transf_pdb(PandasPdb().read_pdb(os.path.join(raw_data_path, code + '_l_b.pdb')),
                          os.path.join(final_dir, code + '_l_b.pdb'))
        random_transf_pdb(PandasPdb().read_pdb(os.path.join(raw_data_path, code + '_r_b.pdb')),
                          os.path.join(final_dir, code + '_r_b.pdb'))



if __name__ == "__main__":
    main_dips()
    main_db5()
