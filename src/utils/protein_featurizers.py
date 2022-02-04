# -*- coding: utf-8 -*-
# Node and edge featurization for molecular graphs.
# pylint: disable= no-member, arguments-differ, invalid-name

from dgllife.utils import one_hot_encoding


def residue_type_one_hot_dips(residue):

    dit = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
           'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
           'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
           'HIP': 'H', 'HIE': 'H', 'TPO': 'T', 'HID': 'H', 'LEV': 'L', 'MEU': 'M', 'PTR': 'Y',
           'GLV': 'E', 'CYT': 'C', 'SEP': 'S', 'HIZ': 'H', 'CYM': 'C', 'GLM': 'E', 'ASQ': 'D',
           'TYS': 'Y', 'CYX': 'C', 'GLZ': 'G'}
    allowable_set = ['Y', 'R', 'F', 'G', 'I', 'V', 'A', 'W', 'E', 'H', 'C', 'N', 'M', 'D', 'T', 'S', 'K', 'L', 'Q', 'P']
    res_name = residue
    if res_name not in dit.keys():
        res_name = None
    else:
        res_name = dit[res_name]
    return one_hot_encoding(res_name, allowable_set, encode_unknown=True)


def residue_type_one_hot_dips_not_one_hot(residue):

    dit = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
           'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
           'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',

           'HIP': 'H', 'HIE': 'H', 'TPO': 'T', 'HID': 'H', 'LEV': 'L', 'MEU': 'M', 'PTR': 'Y',
           'GLV': 'E', 'CYT': 'C', 'SEP': 'S', 'HIZ': 'H', 'CYM': 'C', 'GLM': 'E', 'ASQ': 'D',
           'TYS': 'Y', 'CYX': 'C', 'GLZ': 'G'}

    rare_residues = {'HIP': 'H', 'HIE': 'H', 'TPO': 'T', 'HID': 'H', 'LEV': 'L', 'MEU': 'M', 'PTR': 'Y',
           'GLV': 'E', 'CYT': 'C', 'SEP': 'S', 'HIZ': 'H', 'CYM': 'C', 'GLM': 'E', 'ASQ': 'D',
           'TYS': 'Y', 'CYX': 'C', 'GLZ': 'G'}

    if residue in rare_residues.keys():
        print('Some rare residue: ', residue)

    indicator = {'Y': 0, 'R': 1, 'F': 2, 'G': 3, 'I': 4, 'V': 5,
                 'A': 6, 'W': 7, 'E': 8, 'H': 9, 'C': 10, 'N': 11,
                 'M': 12, 'D': 13, 'T': 14, 'S': 15, 'K': 16, 'L': 17, 'Q': 18, 'P': 19}
    res_name = residue
    if res_name not in dit.keys():
        return 20
    else:
        res_name = dit[res_name]
        return indicator[res_name]



