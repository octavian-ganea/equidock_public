# -*- coding: utf-8 -*-
#
# Early stopping"""
# pylint: disable= no-member, arguments-differ, invalid-name

import datetime
import torch
import copy

__all__ = ['EarlyStopping']


# pylint: disable=C0103
class EarlyStopping(object):
  """Early stop tracker

  Save model checkpoint when observing a performance improvement on
  the validation set and early stop if improvement has not been
  observed for a particular number of epochs.

  Parameters
  ----------
  mode : str
      * 'higher': Higher metric suggests a better model
      * 'lower': Lower metric suggests a better model
      If ``metric`` is not None, then mode will be determined
      automatically from that.
  patience : int
      The early stopping will happen if we do not observe performance
      improvement for ``patience`` consecutive epochs.
  filename : str or None
      Filename for storing the model checkpoint. If not specified,
      we will automatically generate a file starting with ``early_stop``
      based on the current time.
  metric : str or None
      A metric name that can be used to identify if a higher value is
      better, or vice versa. Default to None. Valid options include:
      ``'r2'``, ``'mae'``, ``'rmse'``, ``'roc_auc_score'``.
  """

  def __init__(self, mode='higher', patience=100, filename=None, metric=None, log=None):
    self.log=log
    if filename is None:
      dt = datetime.datetime.now()
      filename = 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
        dt.date(), dt.hour, dt.minute, dt.second)

    if metric is not None:
      assert metric in ['r2', 'mae', 'rmse', 'roc_auc_score'], \
        "Expect metric to be 'r2' or 'mae' or " \
        "'rmse' or 'roc_auc_score', got {}".format(metric)
      if metric in ['r2', 'roc_auc_score']:
        self.log('For metric {}, the higher the better'.format(metric))
        mode = 'higher'
      if metric in ['mae', 'rmse']:
        self.log('For metric {}, the lower the better'.format(metric))
        mode = 'lower'

    assert mode in ['higher', 'lower']
    self.mode = mode
    if self.mode == 'higher':
      self._check = self._check_higher
    else:
      self._check = self._check_lower

    self.patience = patience
    self.counter = 0
    self.filename = filename
    self.best_score = None
    self.early_stop = False

  def _check_higher(self, score, prev_best_score):
    """Check if the new score is higher than the previous best score.

    Parameters
    ----------
    score : float
        New score.
    prev_best_score : float
        Previous best score.

    Returns
    -------
    bool
        Whether the new score is higher than the previous best score.
    """
    return score > prev_best_score

  def _check_lower(self, score, prev_best_score):
    """Check if the new score is lower than the previous best score.

    Parameters
    ----------
    score : float
        New score.
    prev_best_score : float
        Previous best score.

    Returns
    -------
    bool
        Whether the new score is lower than the previous best score.
    """
    return score < prev_best_score

  def step(self, score, model, optimizer, args, epoch, IsMaster=True):
    """Update based on a new score.

    The new score is typically model performance on the validation set
    for a new epoch.

    Parameters
    ----------
    score : float
        New score.
    model : nn.Module
        Model instance.

    Returns
    -------
    bool
        Whether an early stop should be performed.
    """
    if self.best_score is None:
      self.best_score = score
      if IsMaster:
        self.save_checkpoint(model, optimizer, args, epoch)
    elif self._check(score, self.best_score):
      self.best_score = score
      if IsMaster:
        self.save_checkpoint(model, optimizer, args, epoch)
      self.counter = 0
    else:
      self.counter += 1
      if IsMaster:
        self.log(
          f'EarlyStopping counter: {self.counter} out of {self.patience}')
      if self.counter >= self.patience:
        self.early_stop = True
        self.log('EarlyStopping: patience reached. Stopping training ...')
    return self.early_stop


  # def save_checkpoint(self, model):
  #   '''Saves model when the metric on the validation set gets improved.
  #   Parameters
  #   ----------
  #   model : nn.Module
  #       Model instance.
  #   '''
  #   import copy
  #   # args2 = copy.deepcopy(model.args)
  #   # non_load_keys = ['device']
  #   # for k in non_load_keys:
  #   #   del args2[k]
  #   # torch.save({'model_state_dict': model.state_dict(), 'args': args2}, self.filename)
  #   torch.save({'model_state_dict': model.state_dict()}, self.filename)
  #
  # def load_checkpoint(self, model):
  #   '''Load the latest checkpoint
  #   Parameters
  #   ----------
  #   model : nn.Module
  #       Model instance.
  #   '''
  #   model.load_state_dict(torch.load(self.filename)['model_state_dict'])
  # #
  #
  #
  #
  def save_checkpoint(self, model, optimizer, args, epoch):
    '''Saves model when the metric on the validation set gets improved.

    Parameters
    ----------
    model : nn.Module
        Model instance.
    '''
    args2 = copy.deepcopy(args)
    non_load_keys = ['device', 'debug', 'worker', 'n_jobs', 'toy']
    for k in non_load_keys:
      del args2[k]

    checkpoint = {
      'epoch': epoch,
      'state_dict': model.state_dict(),
      'optimizer': optimizer.state_dict(),
      'args': args2
    }
    torch.save(checkpoint, self.filename)


  def load_checkpoint(self, model, optimizer):
    '''Load the latest checkpoint

    Parameters
    ----------
    model : nn.Module
        Model instance.
    '''
    checkpoint = torch.load(self.filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['args'], checkpoint['epoch']

  #

