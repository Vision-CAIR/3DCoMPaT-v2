"""
Extract RGB pointclouds from zipped meshes and store as HDF5 files.
"""
import argparse
import h5py
import numpy as np
import tqdm

from compat3D import StylizedShapeLoader, SemanticLevel


N_POINTS = 2048


def get_hdf5_name(hdf5_dir, hdf5_name):
    hdf5_f = "%s/%s.hdf5" % (hdf5_dir, hdf5_name)
    return hdf5_f

def open_hdf5(hdf5_file, mode):
    hdf5_f = h5py.File(hdf5_file, mode)
    return hdf5_f

def write_hdf5(split, hdf5_dir, semantic_level, zips_path, meta_dir, n_compositions):
    data_loader = StylizedShapeLoader(zip_path=zips_path,
                                      meta_dir=meta_dir,
                                      semantic_level=semantic_level,
                                      split=split,
                                      n_compositions=n_compositions,
                                      n_points=N_POINTS,
                                      get_mats=True)
    n_samples = len(data_loader)

    # If the file exists, open it instead
    hdf5_name = get_hdf5_name(hdf5_dir, split + '_' + str(semantic_level))
    open_mode = 'w'

    # Creating the train set HDF5
    train_hdf5 = open_hdf5(hdf5_name, mode=open_mode)
    if open_mode == 'w':
        train_hdf5.create_dataset('points',
                                shape=(n_samples, N_POINTS, 3 + 3),
                                dtype='float32')
        train_hdf5.create_dataset('points_part_labels',
                                shape=(n_samples, N_POINTS),
                                dtype='uint16')
        train_hdf5.create_dataset('points_mat_labels',
                                shape=(n_samples, N_POINTS),
                                dtype='uint8')
        train_hdf5.create_dataset('shape_id',
                                shape=(n_samples),
                                dtype='S6')
        train_hdf5.create_dataset('style_id',
                                shape=(n_samples),
                                dtype='S6')
        train_hdf5.create_dataset('shape_label',
                                shape=(n_samples),
                                dtype='uint8')

    # Iterating over the train set
    for k in tqdm.tqdm(range(n_samples)):
        shape_id, style_id, shape_label, pointcloud,\
              point_part_labels, point_mat_labels, point_colors = data_loader[k]
        
        # Concatenate xyz with rgb
        pointcloud = np.concatenate([pointcloud, point_colors], axis=1)

        train_hdf5['points'][k] = pointcloud
        train_hdf5['points_part_labels'][k] = point_part_labels
        train_hdf5['points_mat_labels'][k] = point_mat_labels
        train_hdf5['shape_id'][k] = shape_id
        train_hdf5['style_id'][k] = style_id
        train_hdf5['shape_label'][k] = shape_label

    # Close the HDF5 file
    train_hdf5.close()


def main(args):
    write_hdf5(args.split,
               args.hdf5_dir,
               args.semantic_level,
               args.zips_path,
               args.meta_dir,
               args.n_compositions)
    print('[%s_%s_%d] Done!' % (args.split, str(args.semantic_level), args.n_compositions))


if __name__ == '__main__':
    # Get the "split" argument from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta-dir', type=str)
    parser.add_argument('--n-compositions', type=int, default=1)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--semantic-level', type=str, default='coarse')
    parser.add_argument('--hdf5-dir', type=str)
    parser.add_argument('--zips-path', type=str)
    args = parser.parse_args()

    # Convert the semantic level to an enum
    args.semantic_level = SemanticLevel(args.semantic_level)

    # Write the HDF5 file
    main(args)
