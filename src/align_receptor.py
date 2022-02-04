from biopandas.pdb import PandasPdb
from src.utils.protein_utils import rigid_transform_Kabsch_3D
import numpy as np
import os

## Align receptor files for HDOCK or ATTRACT.

input_dir = '/path/to/hdock_or_attract/pdb_files/'

method = 'HDOCK'
file = 'aq_4aqa.pdb1_0.dill_'

ligand_filename = os.path.join(input_dir, file + 'l_b_' + method + '.pdb')
receptor_filename = os.path.join(input_dir, file + 'r_b_' + method + '.pdb')
receptor_to_align_filename = os.path.join(input_dir, file + 'r_b_' + 'COMPLEX.pdb')
out_filename = os.path.join(input_dir, file + 'l_b_' + method + '_ALIGNED.pdb')

ppdb_ligand = PandasPdb().read_pdb(ligand_filename)
ligand_atoms = ppdb_ligand.df['ATOM'][['x_coord', 'y_coord', 'z_coord']].to_numpy().squeeze().astype(np.float32)
receptor_atoms = PandasPdb().read_pdb(receptor_filename).df['ATOM'][['x_coord', 'y_coord', 'z_coord']].to_numpy().squeeze().astype(np.float32)
receptor_to_align = PandasPdb().read_pdb(receptor_to_align_filename).df['ATOM'][['x_coord', 'y_coord', 'z_coord']].to_numpy().squeeze().astype(np.float32)

R,b = rigid_transform_Kabsch_3D(receptor_atoms.T, receptor_to_align.T)
assert  np.linalg.norm( ((R @ receptor_atoms.T) + b ).T - receptor_to_align) < 1e-1

ppdb_ligand.df['ATOM'][['x_coord', 'y_coord', 'z_coord']] = ((R @ ligand_atoms.T) + b ).T
ppdb_ligand.to_pdb(path=out_filename, records=['ATOM'], gz=False)



