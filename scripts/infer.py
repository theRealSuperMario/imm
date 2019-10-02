# ==========================================================
# Author: Tomas Jakab
# ==========================================================
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import os.path as osp

from imm.eval import eval_imm
from imm.models.imm_model import IMMModel
import sklearn.linear_model

from imm.utils.dataset_import import import_dataset



def infer(net, net_file, model_config, training_config, dset,
             batch_size=100):
  # %% ---------------------------------------------------------------------------
  # ------------------------------- Run TensorFlow -------------------------------
  # ------------------------------------------------------------------------------
  def _infer(dset):
    results = eval_imm.evaluate(
        dset, net, model_config, net_file, training_config, batch_size=batch_size,
        random_seed=0, eval_tensors=['gauss_yx', "image"])
    results = {k: np.concatenate(v) for k, v in results.items()}
    return results

  test_tensors = _infer(dset)
  return test_tensors


def main(args):
  experiment_name = args.experiment_name
  iteration = args.iteration
  im_size = args.im_size
  batch_size = args.batch_size
  n_train_samples = None

  postfix = ''
  postfix += '-' + args.dataset
  if n_train_samples is not None:
    postfix += '%.0fk' % (n_train_samples / 1000.0)

  config = eval_imm.load_configs(
      [args.paths_config,
       osp.join('configs', 'experiments', experiment_name + '.yaml')])

  if args.dataset == 'mafl':
    test_dataset_class = import_dataset('celeba')
    test_dset = test_dataset_class(
        config.training.datadir, dataset='mafl', subset=args.test,
        order_stream=True, tps=False,
        image_size=[im_size, im_size])
  elif args.dataset == 'aflw':
    test_dataset_class = import_dataset('aflw')
    test_dset = test_dataset_class(
      config.training.datadir, subset=args.test,
      order_stream=True, tps=False,
      image_size=[im_size, im_size])
  elif args.dataset == 'csv':
    test_dataset_class = import_dataset('csv')
    test_dset = test_dataset_class(
      config.training.datadir,
      image_size=[im_size, im_size],
      **config.training.test_dset_params)
  else:
    raise ValueError('Dataset %s not supported.' % args.test_dataset)

  net = IMMModel

  model_config = config.model
  training_config = config.training

  if iteration is not None:
    net_file = 'model.ckpt-' + str(iteration)
  else:
    net_file = 'model.ckpt'
  checkpoint_file = osp.join(config.training.logdir, net_file + '.meta')
  if not osp.isfile(checkpoint_file):
    raise ValueError('Checkpoint file %s not found.' % checkpoint_file)

  inferred_results = infer(
      net, net_file, model_config, training_config, test_dset,
      batch_size=batch_size)

  target_dir = config.training.logdir + "_test"
  fname = osp.join(target_dir, str(iteration) + ".npz")
  print("Writing inferred results to : {}".format(fname))
  for k in inferred_results.keys():
    print("--> {}".format(k))
  np.savez_compressed(fname, **inferred_results)


  if hasattr(config.training.train_dset_params, 'dataset'):
    model_dataset = config.training.train_dset_params.dataset
  else:
    model_dataset = config.training.dset


if  __name__=='__main__':
  import argparse
  parser = argparse.ArgumentParser(description='Test model on face datasets.')
  parser.add_argument('--experiment-name', type=str, required=True, help='Name of the experiment to evaluate.')
  parser.add_argument('--dataset', type=str, required=True, help='dataset landmarks retrieval (mafl|aflw|csv).')

  parser.add_argument('--paths-config', type=str, default='configs/paths/default.yaml', required=False, help='Path to the paths config.')
  parser.add_argument('--iteration', type=int, default=None, required=False, help='Checkpoint iteration to evaluate.')
  parser.add_argument('--im-size', type=int, default=128, required=False, help='Image size.')
  parser.add_argument('--batch-size', type=int, default=100, required=False, help='batch_size')

  args = parser.parse_args()
  main(args)
