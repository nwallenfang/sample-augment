import json
import os
import re
import tempfile

import torch
# import sys

# sys.path.insert(0, 'H:\\thesis\\repos\\thesis_nils\\sample_augment\\')
# sys.path.insert(0, 'H:\\thesis\\repos\\thesis_nils\\sample_augment\\models\\stylegan2\\')

import sample_augment.models.stylegan2_old.dnnlib as dnnlib
from sample_augment.models.stylegan2_old.train import UserError, setup_training_loop_kwargs, subprocess_fn


def train_stylegan():
    out_dir = r'E:\Master_Thesis_Nils\stylegan-training'

    config_kwargs = {
        'data': r"H:\thesis\sampling_aug\data\interim\gc10_train.pt",
        # 'custom_name' 'gc10_pre_FFHQ'
        'gpus': 2,
        'snap': None,
        'metrics': None,
        'seed': 16,  # remember to change this when running the experiment a second time ;)
        'cond': True,
        'subset': None,
        'mirror': True,  # checked each class and x-flip can be done semantically for GC10
        'cfg': None,
        'gamma': None,  # tune this parameter with values such as 0.1, 0.5, 1, 5, 10
        'kimg': 5000,
        'batch': None,
        'aug': None,
        'p': None,
        'target': None,  # ADA target value, might need tweaking
        'augpipe': 'bgc-gc10',  # custom augmentation pipeline without 90 degree rotations
        'resume': 'ffhq256',
        # "E:\\Master_Thesis_Nils\\stylegan-training\\00009-gc10_pre_FFHQ-cond-mirror-auto2-kimg5000"
        #      "-resumecelebahq256\\network-snapshot-000200.pkl",
        # 'celebahq256', # 'ffhq256',  # checkpoint for transfer learning / resuming interrupted run
        'freezed': 3,  # int, 'Freeze-D', 'freeze the highest-resolution layers of the discriminator
                       # during transfer'
        'fp32': None,
        'nhwc': None,
        'nobench': None,
        'allow_tf32': None,
        'workers': 1  # could try setting number of workers to 1, since the data is fully in RAM
    }
    dry_run = False

    dnnlib.util.Logger(should_flush=True)

    print('Starting..')
    # Setup training options.
    try:
        run_desc, args = setup_training_loop_kwargs(**config_kwargs)
    except UserError as err:
        print(f"Error: {err}")
        return

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(out_dir):
        prev_run_dirs = [x for x in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    args.run_dir = os.path.join(out_dir, f'{cur_run_id:05d}-{run_desc}')
    assert not os.path.exists(args.run_dir)

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(args, indent=2))
    print()
    print(f'Output directory:   {args.run_dir}')
    # print(f'Training data:      {args.training_set_kwargs.path}')
    print(f'Training duration:  {args.total_kimg} kimg')
    print(f'Number of GPUs:     {args.num_gpus}')
    print(f'Number of images:   {args.training_set_kwargs.max_size}')
    print(f'Image resolution:   {args.training_set_kwargs.resolution}')
    print(f'Conditional model:  {args.training_set_kwargs.use_labels}')
    # print(f'Dataset x-flips:    {args.training_set_kwargs.xflip}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(args.run_dir)
    with open(os.path.join(args.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(args, f, indent=2)

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    # main()  # pylint: disable=no-value-for-parameter
    train_stylegan()
