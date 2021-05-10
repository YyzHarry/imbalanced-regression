import os
import time
import copy
import random
import logging
import itertools
import ipdb as pdb
import numpy as np

import torch
import torch.optim.lr_scheduler
from torch.nn.utils.clip_grad import clip_grad_norm

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.training.optimizers import Optimizer
from util import device_mapping

def build_trainer(args, model, iterator):
    '''Build a trainer'''
    opt_params = Params({'type': args.optimizer, 'lr': args.lr, 'weight_decay': 1e-5})
    train_params = Params({'max_vals': args.max_vals, 'cuda_device': args.cuda,
                           'patience': args.patience, 'grad_norm': args.max_grad_norm,
                           'lr_decay': .99})
    trainer = SamplingMultiTaskTrainer.from_params(model, args.store_dir, iterator, copy.deepcopy(train_params))
    return trainer, train_params, opt_params

class SamplingMultiTaskTrainer():
    def __init__(self, model, iterator, patience=2, max_vals=50,
                 serialization_dir=None, cuda_device=-1,
                 grad_norm=None, grad_clipping=None, lr_decay=None):
        self._model = model
        self._iterator = iterator

        self._patience = patience
        self._max_vals = max_vals
        self._serialization_dir = serialization_dir
        self._cuda_device = cuda_device
        self._grad_norm = grad_norm
        self._grad_clipping = grad_clipping
        self._lr_decay = lr_decay

        self._task_infos = None
        self._metric_infos = None

        self._log_interval = 10  # seconds
        if self._cuda_device >= 0:
            self._model = self._model.cuda(self._cuda_device)

    def _check_history(self, metric_history, cur_score, should_decrease=False):
        '''
        Given a task, the history of the performance on that task,
        and the current score, check if current score is
        best so far and if out of patience.
        '''
        patience = self._patience + 1
        best_fn = min if should_decrease else max
        best_score = best_fn(metric_history)
        if best_score == cur_score:
            best_so_far = metric_history.index(best_score) == len(metric_history) - 1
        else:
            best_so_far = False

        out_of_patience = False
        if len(metric_history) > patience:
            if should_decrease:
                out_of_patience = max(metric_history[-patience:]) <= cur_score
            else:
                out_of_patience = min(metric_history[-patience:]) >= cur_score

        if best_so_far and out_of_patience:
            pdb.set_trace()

        return best_so_far, out_of_patience


    def _setup_training(self, tasks, train_params, optimizer_params, iterator):
        # Task bookkeeping
        task_infos = {task.name: {} for task in tasks}
        for task in tasks:
            task_info = task_infos[task.name]
            tr_generator = iterator(task.train_data, num_epochs=None, cuda_device=self._cuda_device) # by default, shuffle=True
            task_info['n_tr_batches'] = iterator.get_num_batches(task.train_data)
            task_info['tr_generator'] = tr_generator
            task_info['loss'] = 0.0
            task_info['total_batches_trained'] = 0
            task_info['n_batches_since_val'] = 0
            task_info['optimizer'] = Optimizer.from_params(train_params, copy.deepcopy(optimizer_params))
            task_info['stopped'] = False
            task_info['last_log'] = time.time()
        # Metric bookkeeping
        all_metrics = [task.val_metric for task in tasks]
        metric_infos = {metric: {'hist': [], 'stopped': False, 'best': (-1, -1, {})} for metric in all_metrics}
        self._task_infos = task_infos
        self._metric_infos = metric_infos
        return task_infos, metric_infos


    def train(self, tasks, validation_interval, train_params, optimizer_params, resume=False):

        iterator = self._iterator
        task_infos, metric_infos = self._setup_training(tasks, train_params, optimizer_params, iterator)

        n_pass, should_stop = 0, False
        real_epoch = 0
        if self._serialization_dir is not None: # Resume from serialization path
            if resume and any(["model_state" in x for x in os.listdir(self._serialization_dir)]):
                real_epoch, n_pass, should_stop = self._restore_checkpoint()
                logging.info("Loaded model from checkpoint. Starting at iter %d, epoch %d", n_pass, real_epoch)

        if self._grad_clipping is not None:
            clip_function = lambda grad: grad.clamp(-self._grad_clipping, self._grad_clipping)
            for parameter in self._model.parameters():
                if parameter.requires_grad:
                    parameter.register_hook(clip_function)

        sample_weights = [task_infos[task.name]['n_tr_batches'] for task in tasks]
        samples = random.choices(tasks, weights=sample_weights, k=validation_interval)

        logging.info("Beginning training.")
        all_tr_metrics = {}

        while not should_stop:
            self._model.train()

            task = samples[n_pass % (validation_interval)]
            task_info = task_infos[task.name]
            if task_info['stopped']:
                continue
            tr_generator = task_info['tr_generator']
            optimizer = task_info['optimizer']
            total_batches_trained = task_info['total_batches_trained']
            n_batches_since_val = task_info['n_batches_since_val']
            tr_loss = task_info['loss']
            for batch in itertools.islice(tr_generator, 1):
                n_batches_since_val += 1
                total_batches_trained += 1
                optimizer.zero_grad()
                output_dict = self._forward(batch, task=task, epoch=real_epoch)
                assert "loss" in output_dict, "Model must return a dict containing a 'loss' key"
                loss = output_dict["loss"]
                assert torch.isfinite(loss).all(), logging.info(f'Bad Loss: {loss}')
                loss.backward()
                tr_loss += loss.data.cpu().numpy()

                # Gradient regularization and application
                if self._grad_norm:
                    clip_grad_norm(self._model.parameters(), self._grad_norm)
                optimizer.step()

                n_pass += 1 # update per batch

            # Update training progress on that task
            task_info['n_batches_since_val'] = n_batches_since_val
            task_info['total_batches_trained'] = total_batches_trained
            task_info['loss'] = tr_loss

            if n_pass // task_info['n_tr_batches'] > real_epoch:
                if self._model.args.fds and real_epoch >= self._model.args.start_update:
                    encodings, labels = [], []
                    with torch.no_grad():
                        for batch in self._iterator(task.train_data, num_epochs=1, cuda_device=self._cuda_device):
                            out_dict = self._forward(batch, task=task, epoch=real_epoch)
                            encodings.extend(out_dict['embs'].data.cpu().numpy())
                            labels.extend(out_dict['labels'].data.squeeze(1).cpu().numpy())

                    encodings, labels = torch.from_numpy(np.vstack(encodings)).cuda(self._cuda_device), \
                                        torch.from_numpy(np.hstack(labels)).cuda(self._cuda_device)
                    self._model.FDS.update_last_epoch_stats(real_epoch)
                    self._model.FDS.update_running_stats(encodings, labels, real_epoch)
                    logging.info(f"Create Epoch [{real_epoch}] features of all training data...")
                real_epoch += 1

            # Intermediate logging
            if time.time() - task_info['last_log'] > self._log_interval:
                task_metrics = task.get_metrics(type='overall')
                task_metrics["%s_loss" % task.name] = tr_loss / n_batches_since_val
                description = self._description_from_metrics(task_metrics)
                logging.info("Iter %d (Epoch: %d): task %s, iter_since_val %d: %s", n_pass, real_epoch,
                            task.name, n_batches_since_val, description)
                task_info['last_log'] = time.time()

            # Validation
            if n_pass % (validation_interval) == 0:
                val_check = int(n_pass / validation_interval)
                logging.info("\n***** Iteration %d / Val Check %d *****", n_pass, val_check)
                # Get metrics for all training progress so far
                for task in tasks:
                    task_info = task_infos[task.name]
                    n_batches_since_val = task_info['n_batches_since_val']
                    if n_batches_since_val > 0:
                        task_metrics = task.get_metrics(reset=True)
                        all_tr_metrics["%s_loss" % task.name] = float(task_info['loss'] / n_batches_since_val)
                    else:
                        all_tr_metrics["%s_loss" % task.name] = 0.0
                    logging.info("%s: trained on %d batches, %.3f epochs", task.name,
                                n_batches_since_val, n_batches_since_val / task_info['n_tr_batches'])
                    if n_batches_since_val > 0:
                        logging.info("Training statistics:")
                        logging.info(f"train loss: {all_tr_metrics['%s_loss' % task.name]:.6f}")
                        for shot in ['Overall', 'Many', 'Medium', 'Few']:
                            logging.info(f"{shot}: MSE {task_metrics[shot.lower()]['mse']:.3f}\t"
                                        f"L1 {task_metrics[shot.lower()]['l1']:.3f}\t"
                                        f"G-Mean {task_metrics[shot.lower()]['gmean']:.3f}\t"
                                        f"Pearson {task_metrics[shot.lower()]['pearsonr']:.3f}\t"
                                        f"Spearman {task_metrics[shot.lower()]['spearmanr']:.3f}\t"
                                        f"Number {task_metrics[shot.lower()]['num_samples']}")
                # Validate
                logging.info("\nValidating...")
                all_val_metrics, should_save, task_infos, metric_infos = \
                        self._validate(real_epoch, val_check, tasks, task_infos, metric_infos, iterator)

                # Check stopping conditions
                should_stop, task_infos, metric_infos = \
                        self._check_stop(val_check, tasks, task_infos, metric_infos)

                # Log results
                logging.info("Validation statistics:")
                logging.info(f"validation loss: {all_val_metrics['%s_loss' % task.name]:.6f}")
                for shot in ['Overall', 'Many', 'Medium', 'Few']:
                    logging.info(f" * {shot}: MSE {all_val_metrics[shot.lower()]['mse']:.3f}\t"
                                f"L1 {all_val_metrics[shot.lower()]['l1']:.3f}\t"
                                f"G-Mean {all_val_metrics[shot.lower()]['gmean']:.3f}\t"
                                f"Pearson {all_val_metrics[shot.lower()]['pearsonr']:.3f}\t"
                                f"Spearman {all_val_metrics[shot.lower()]['spearmanr']:.3f}\t"
                                f"Number {all_val_metrics[shot.lower()]['num_samples']}")

                self._metric_infos = metric_infos
                self._task_infos = task_infos
                all_tr_metrics = {}
                samples = random.choices(tasks, weights=sample_weights, k=validation_interval)

                if should_save:
                    self._save_checkpoint({"epoch": real_epoch, "iter": n_pass, "should_stop": should_stop}, best=True)
                self._save_checkpoint({"epoch": real_epoch, "iter": n_pass, "should_stop": should_stop}, best=False)

        logging.info('Stopped training after %d validation checks', n_pass / validation_interval)
        return self._aggregate_results(tasks, task_infos, metric_infos)

    def _aggregate_results(self, tasks, task_infos, metric_infos):
        ''' Ad hoc helper function to print results after finishing training '''
        results = {}
        for task in tasks:
            task_info = task_infos[task.name]
            logging.info('Trained %s for %d batches or %.3f epochs',
                         task.name, task_info['total_batches_trained'],
                         task_info['total_batches_trained'] / task_info['n_tr_batches'])
            results[task.name] = metric_infos[task.val_metric]['best'][0]

        logging.info('\n***** VALIDATION RESULTS *****')
        for metric in metric_infos.keys():
            best_epoch, best_val_check, val_check_metrics = metric_infos[metric]['best']
            logging.info(f'Best Val Check: {best_val_check}; Best Epoch: {best_epoch}; metric: {metric}')
            for shot in ['Overall', 'Many', 'Medium', 'Few']:
                logging.info(f" * {shot}: MSE {val_check_metrics[shot.lower()]['mse']:.3f}\t"
                            f"L1 {val_check_metrics[shot.lower()]['l1']:.3f}\t"
                            f"G-Mean {val_check_metrics[shot.lower()]['gmean']:.3f}\t"
                            f"Pearson {val_check_metrics[shot.lower()]['pearsonr']:.3f}\t"
                            f"Spearman {val_check_metrics[shot.lower()]['spearmanr']:.3f}\t"
                            f"Number {val_check_metrics[shot.lower()]['num_samples']}")
        return results

    def _validate(self, epoch, val_check, tasks, task_infos, metric_infos, iterator):
        self._model.eval()
        all_val_metrics = {("%s_loss" % task.name): 0.0 for task in tasks}
        n_examples_overall = 0.0

        for task in tasks:
            n_examples = 0.0
            task_info = task_infos[task.name]
            val_generator = iterator(task.val_data, num_epochs=1, shuffle=False, cuda_device=self._cuda_device)
            n_val_batches = iterator.get_num_batches(task.val_data)
            all_val_metrics["%s_loss" % task.name] = 0.0
            batch_num = 0
            for batch in val_generator:
                batch_num += 1
                val_output_dict = self._forward(batch, task=task)
                loss = val_output_dict["loss"]
                all_val_metrics["%s_loss" % task.name] += loss.data.cpu().numpy()
                n_examples += batch['label'].size()[0]
            assert batch_num == n_val_batches, pdb.set_trace()

            # Get task validation metrics and store in all_val_metrics
            task_metrics = task.get_metrics(reset=True)
            for shot, dic in task_metrics.items():
                all_val_metrics[shot] = dic
            all_val_metrics["%s_loss" % task.name] /= n_val_batches
            n_examples_overall += n_examples

            # Reset training progress
            task_info['n_batches_since_val'] = 0
            task_info['loss'] = 0

        # Track per task patience
        should_save = False # whether to save or not
        for task in tasks:
            metric = task.val_metric
            if metric_infos[metric]['stopped']:
                continue
            this_val_check_metric = all_val_metrics['overall'][metric]
            metric_history = metric_infos[metric]['hist']
            metric_history.append(this_val_check_metric)
            is_best_so_far, out_of_patience = \
                    self._check_history(metric_history, this_val_check_metric, should_decrease=True)
            if is_best_so_far:
                logging.info("Best model found for %s.", task.name)
                metric_infos[metric]['best'] = (epoch, val_check, all_val_metrics)
                should_save = True
            if out_of_patience:
                metric_infos[metric]['stopped'] = True
                logging.info("Out of patience. Stopped tracking %s", task.name)

        return all_val_metrics, should_save, task_infos, metric_infos

    def _check_stop(self, val_check, tasks, task_infos, metric_infos):
        ''' Check to see if should stop '''
        stop_val = metric_infos[tasks[0].val_metric]['stopped']

        should_stop = False
        if stop_val:
            should_stop = True
            logging.info("All metrics ran out of patience. Stopping training.")
        if val_check >= self._max_vals:
            logging.info("Maximum number of validations hit. Stopping training.")
            should_stop = True

        return should_stop, task_infos, metric_infos

    def _forward(self, batch, epoch=None, task=None):
        tensor_batch = batch
        return self._model.forward(task, epoch, **tensor_batch)

    def _description_from_metrics(self, metrics):
        return ', '.join([("%s: %.4f" if type(value) is not int else "%s: %d") % (name, value) for name, value in metrics.items()]) + " ||"

    def _save_checkpoint(self, training_state, best=False):
        if best:
            suffix = '_best'
        else:
            suffix = ''

        model_path = os.path.join(self._serialization_dir, f"model_state{suffix}.th")
        model_state = self._model.state_dict()
        torch.save(model_state, model_path)

        torch.save(training_state, os.path.join(self._serialization_dir, f"training_state{suffix}.th"))

        task_states = {}
        for task_name, task_info in self._task_infos.items():
            task_states[task_name] = {}
            task_states[task_name]['total_batches_trained'] = task_info['total_batches_trained']
            task_states[task_name]['stopped'] = task_info['stopped']
            task_states[task_name]['optimizer'] = task_info['optimizer'].state_dict()
        torch.save(task_states, os.path.join(self._serialization_dir, f"task_state{suffix}.th"))

        metric_states = {}
        for metric_name, metric_info in self._metric_infos.items():
            metric_states[metric_name] = {}
            metric_states[metric_name]['hist'] = metric_info['hist']
            metric_states[metric_name]['stopped'] = metric_info['stopped']
            metric_states[metric_name]['best'] = metric_info['best']
        torch.save(metric_states, os.path.join(self._serialization_dir, f"metric_state{suffix}.th"))

        logging.info(f"Saved model_state{suffix}.th / training_state{suffix}.th / task_state{suffix}.th / metric_state{suffix}.th "
                     f"to {self._serialization_dir}")

    def _restore_checkpoint(self):
        """
        Restores a model from a serialization_dir to the last saved checkpoint.
        This includes an epoch count and optimizer state, which is serialized separately
        from  model parameters. This function should only be used to continue training.
        """
        if not self._serialization_dir:
            raise ConfigurationError("serialization_dir not specified - cannot "
                                     "restore a model without a directory path.")

        logging.info(f'Recovering last model / training / task / metric states '
                     f'from {self._serialization_dir}...')

        model_path = os.path.join(self._serialization_dir, "model_state.th")
        training_state_path = os.path.join(self._serialization_dir, "training_state.th")
        task_state_path = os.path.join(self._serialization_dir, "task_state.th")
        metric_state_path = os.path.join(self._serialization_dir, "metric_state.th")

        model_state = torch.load(model_path, map_location=device_mapping(self._cuda_device))
        self._model.load_state_dict(model_state)

        task_states = torch.load(task_state_path, map_location=device_mapping(self._cuda_device))
        for task_name, task_state in task_states.items():
            self._task_infos[task_name]['total_batches_trained'] = task_state['total_batches_trained']
            self._task_infos[task_name]['optimizer'].load_state_dict(task_state['optimizer'])
            self._task_infos[task_name]['stopped'] = task_state['stopped']
            generator = self._task_infos[task_name]['tr_generator']
            for _ in itertools.islice(generator, task_state['total_batches_trained'] % \
                                      self._task_infos[task_name]['n_tr_batches']):
                pass

        metric_states = torch.load(metric_state_path, map_location=device_mapping(self._cuda_device))
        for metric_name, metric_state in metric_states.items():
            self._metric_infos[metric_name]['hist'] = metric_state['hist']
            self._metric_infos[metric_name]['stopped'] = metric_state['stopped']
            self._metric_infos[metric_name]['best'] = metric_state['best']

        training_state = torch.load(training_state_path, map_location=device_mapping(self._cuda_device))
        return training_state["epoch"], training_state["iter"], training_state["should_stop"]

    @classmethod
    def from_params(cls, model, serialization_dir, iterator, params):
        ''' Generator trainer from parameters.  '''

        patience = params.pop("patience", 2)
        max_vals = params.pop("max_vals", 100)
        cuda_device = params.pop("cuda_device", -1)
        grad_norm = params.pop("grad_norm", None)
        grad_clipping = params.pop("grad_clipping", None)
        lr_decay = params.pop("lr_decay", None)

        params.assert_empty(cls.__name__)
        return SamplingMultiTaskTrainer(model,
                                        iterator,
                                        patience=patience,
                                        max_vals=max_vals,
                                        serialization_dir=serialization_dir,
                                        cuda_device=cuda_device,
                                        grad_norm=grad_norm,
                                        grad_clipping=grad_clipping,
                                        lr_decay=lr_decay)
