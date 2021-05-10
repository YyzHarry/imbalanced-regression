import logging
import tqdm
import numpy as np

def evaluate(model, tasks, iterator, cuda_device, split="val"):
    '''Evaluate on a dataset'''
    model.eval()

    all_preds = {}
    n_overall_examples = 0
    for task in tasks:
        n_examples = 0
        task_preds, task_idxs, task_labels = [], [], []
        if split == "val":
            dataset = task.val_data
        elif split == 'train':
            dataset = task.train_data
        elif split == "test":
            dataset = task.test_data
        generator = iterator(dataset, num_epochs=1, shuffle=False, cuda_device=cuda_device)
        generator_tqdm = tqdm.tqdm(generator, total=iterator.get_num_batches(dataset), disable=True)
        for batch in generator_tqdm:
            tensor_batch = batch
            out = model.forward(task, **tensor_batch)
            n_examples += batch['label'].size()[0]
            preds, _ = out['logits'].max(dim=1)
            task_preds += list(preds.data.cpu().numpy())
            task_labels += list(batch['label'].squeeze().data.cpu().numpy())

        task_metrics = task.get_metrics(reset=True)
        logging.info('\n***** TEST RESULTS *****')
        for shot in ['Overall', 'Many', 'Medium', 'Few']:
            logging.info(f" * {shot}: MSE {task_metrics[shot.lower()]['mse']:.3f}\t"
                         f"L1 {task_metrics[shot.lower()]['l1']:.3f}\t"
                         f"G-Mean {task_metrics[shot.lower()]['gmean']:.3f}\t"
                         f"Pearson {task_metrics[shot.lower()]['pearsonr']:.3f}\t"
                         f"Spearman {task_metrics[shot.lower()]['spearmanr']:.3f}\t"
                         f"Number {task_metrics[shot.lower()]['num_samples']}")

        n_overall_examples += n_examples
        task_preds = [min(max(np.float32(0.), pred * np.float32(5.)), np.float32(5.)) for pred in task_preds]
        all_preds[task.name] = (task_preds, task_idxs)

    return task_preds, task_labels, task_metrics['overall']['mse']