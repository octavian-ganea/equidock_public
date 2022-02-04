## Run:
# python preprocess_raw_data.py -n_jobs 40 -data db5 -graph_nodes residues -graph_cutoff 30 -graph_max_neighbor 10 -graph_residue_loc_is_alphaC -pocket_cutoff 8
# If you want to use less DIPS data, then: python preprocess_raw_data.py -n_jobs 60 -data dips -graph_nodes residues -graph_cutoff 30 -graph_max_neighbor 10 -graph_residue_loc_is_alphaC -pocket_cutoff 8 -data_fraction 0.25

from src.utils.args import *

if __name__ == "__main__":
    from src.utils.db5_data import Unbound_Bound_Data

    the_path = args['cache_path']

    if args['data'] == 'db5':
        raw_data_path= './data/benchmark5.5/structures'
        split_files_path = './data/benchmark5.5/cv/'
    else:
        assert args['data'] == 'dips'
        raw_data_path= './DIPS/data/DIPS/interim/pairs-pruned/' ## See utils/partition_dips.py on how to get this data preprocessed.
        split_files_path = './DIPS/data/DIPS/interim/pairs-pruned/'


    os.makedirs(the_path, exist_ok=True)  ## Directory may exist!

    num_splits = 1
    if args['data'] == 'db5':
        num_splits = 3


    for i in range(num_splits):
        print('\n\nProcessing split ', i)

        args['cache_path'] = os.path.join(the_path, 'cv_' + str(i))
        os.makedirs(args['cache_path'], exist_ok=True)

        if args['data'] == 'db5':
            split_files_path = os.path.join(split_files_path, 'cv_' + str(i))

        Unbound_Bound_Data(args, reload_mode='val', load_from_cache=False, raw_data_path=raw_data_path,
                           split_files_path=split_files_path, data_fraction=args['data_fraction'])
        Unbound_Bound_Data(args, reload_mode='test', load_from_cache=False, raw_data_path=raw_data_path,
                           split_files_path=split_files_path, data_fraction=args['data_fraction'])
        Unbound_Bound_Data(args, reload_mode='train', load_from_cache=False, raw_data_path=raw_data_path,
                           split_files_path=split_files_path, data_fraction=args['data_fraction']) # Writes data into the cache folder.
