# Place this file in src/ of the DIPS folder preprocessed as in
# https://github.com/amorehead/DIPS-Plus up to and including the step prune_pairs.py (but not beyond that step).

import logging
import os
import random
from pathlib import Path

import atom3.pair as pa
import click
import pandas as pd
from atom3 import database as db
from tqdm import tqdm

from project.utils.constants import DB5_TEST_PDB_CODES, ATOM_COUNT_LIMIT


@click.command()
@click.argument('output_dir', default='../DIPS/final/raw', type=click.Path())
def main(output_dir: str):
    """Partition dataset filenames."""
    filter_by_atom_count = True
    max_atom_count = 10000
    logger = logging.getLogger(__name__)
    logger.info(f'Writing filename DataFrames to their respective text files')

    # Make sure the output_dir exists
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print('pairs-postprocessed.txt start')

    pairs_postprocessed_txt = os.path.join(output_dir, 'pairs-postprocessed.txt')
    if os.path.exists(pairs_postprocessed_txt):
        print('pairs-postprocessed.txt exists, skipping ...')
    else:
        open(pairs_postprocessed_txt, 'w').close()  # Create pairs-postprocessed.txt from scratch each run

        # Record dataset filenames conditionally by sequence length (if requested - otherwise, record all)
        pair_filenames = [pair_filename for pair_filename in Path(output_dir).rglob('*.dill')]
        for pair_filename in tqdm(pair_filenames):
            struct_id = pair_filename.as_posix().split(os.sep)[-2]
            if filter_by_atom_count:
                postprocessed_pair: pa.Pair = pd.read_pickle(pair_filename)
                if len(postprocessed_pair.df0) < max_atom_count and len(postprocessed_pair.df1) < max_atom_count:
                    with open(pairs_postprocessed_txt, 'a') as f:
                        path, filename = os.path.split(pair_filename.as_posix())
                        filename = os.path.join(struct_id, filename)
                        f.write(filename + '\n')  # Pair file was copied
            else:
                with open(pairs_postprocessed_txt, 'a') as f:
                    path, filename = os.path.split(pair_filename.as_posix())
                    filename = os.path.join(struct_id, filename)
                    f.write(filename + '\n')  # Pair file was copied

    print('pairs-postprocessed.txt done')

    # Prepare files
    pairs_postprocessed_train_txt = os.path.join(output_dir, 'pairs-postprocessed-train.txt')
    if not os.path.exists(pairs_postprocessed_train_txt):  # Create train data list if not already existent
        open(pairs_postprocessed_train_txt, 'w+').close()

    pairs_postprocessed_val_txt = os.path.join(output_dir, 'pairs-postprocessed-val.txt')
    if not os.path.exists(pairs_postprocessed_val_txt):  # Create val data list if not already existent
        open(pairs_postprocessed_val_txt, 'w+').close()

    pairs_postprocessed_test_txt = os.path.join(output_dir, 'pairs-postprocessed-test.txt')
    if not os.path.exists(pairs_postprocessed_test_txt):  # Create test data list if not already existent
        open(pairs_postprocessed_test_txt, 'w+').close()

    # Write out training-validation partitions for DIPS
    output_dirs = [filename
                   for filename in os.listdir(output_dir)
                   if os.path.isdir(os.path.join(output_dir, filename))]

    random.shuffle(output_dirs)
    train_dirs = output_dirs[:-40]
    val_dirs = output_dirs[-40:-20]
    test_dirs = output_dirs[-20:]

    # Ascertain training and validation filename separately
    filenames_frame = pd.read_csv(pairs_postprocessed_txt, header=None)
    train_filenames = [os.path.join(train_dir, filename)
                       for train_dir in train_dirs
                       for filename in os.listdir(os.path.join(output_dir, train_dir))
                       if os.path.join(train_dir, filename) in filenames_frame.values]
    val_filenames = [os.path.join(val_dir, filename)
                     for val_dir in val_dirs
                     for filename in os.listdir(os.path.join(output_dir, val_dir))
                     if os.path.join(val_dir, filename) in filenames_frame.values]
    test_filenames = [os.path.join(test_dir, filename)
                     for test_dir in test_dirs
                     for filename in os.listdir(os.path.join(output_dir, test_dir))
                     if os.path.join(test_dir, filename) in filenames_frame.values]

    # Create separate .txt files to describe the training list and validation list, respectively
    train_filenames_frame, val_filenames_frame, test_filenames_frame = pd.DataFrame(train_filenames), pd.DataFrame(val_filenames), pd.DataFrame(test_filenames)
    train_filenames_frame.to_csv(pairs_postprocessed_train_txt, header=None, index=None, sep=' ', mode='a')
    val_filenames_frame.to_csv(pairs_postprocessed_val_txt, header=None, index=None, sep=' ', mode='a')
    test_filenames_frame.to_csv(pairs_postprocessed_test_txt, header=None, index=None, sep=' ', mode='a')

if __name__ == '__main__':
    log_fmt = '%(asctime)s %(levelname)s %(process)d: %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
