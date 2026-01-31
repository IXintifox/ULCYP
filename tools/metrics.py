from collections import defaultdict

import joblib
import numpy as np
from sklearn.metrics import (roc_auc_score, matthews_corrcoef, accuracy_score,
                             precision_recall_curve, balanced_accuracy_score, recall_score, auc,
                             precision_score,
                             f1_score)
from sklearn.metrics import confusion_matrix
from ulcyp.downstream import TaskClass

class Metrics:
    @staticmethod
    def auc(y_true, y_pred):
        return roc_auc_score(y_true, y_pred)

    @staticmethod
    def mcc(y_true, y_prob, threshold=0.5):
        y_pred = Metrics.threshold_predict(y_prob, threshold)
        return matthews_corrcoef(y_true, y_pred)

    @staticmethod
    def f1(y_true, y_prob, threshold=0.5):
        y_pred = Metrics.threshold_predict(y_prob, threshold)
        return f1_score(y_true, y_pred)

    @staticmethod
    def precision(y_true, y_prob, threshold=0.5):
        y_pred = Metrics.threshold_predict(y_prob, threshold)
        return precision_score(y_true, y_pred)

    @staticmethod
    def acc(y_true, y_prob, threshold=0.5):
        y_pred = Metrics.threshold_predict(y_prob, threshold)
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def threshold_predict(y_prob, threshold=0.5):
        return np.array([1 if prob > threshold else 0 for prob in y_prob])

    @staticmethod
    def pr_auc(y_true, y_prob):
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        return auc(recall, precision)

    @staticmethod
    def sensitivity(y_true, y_prob, threshold=0.5):
        y_pred = Metrics.threshold_predict(y_prob, threshold)
        return recall_score(y_true, y_pred)

    @staticmethod
    def specificity(y_true, y_prob, threshold=0.5):
        y_pred = Metrics.threshold_predict(y_prob, threshold)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) != 0 else 0

    @staticmethod
    def balanced_accuracy(y_true, y_prob, threshold=0.5):
        y_pred = Metrics.threshold_predict(y_prob, threshold)
        return balanced_accuracy_score(y_true, y_pred)
    @staticmethod
    def pu_auc(positive_labels, unlabeled_scores, pi):

        y_true = np.concatenate([np.ones(len(positive_labels)), np.zeros(len(unlabeled_scores))])
        y_scores = np.concatenate([positive_labels, unlabeled_scores])
        auu = roc_auc_score(y_true, y_scores)
        return (auu - (1 - pi)) / pi


    def metric_aur(self, true, y_scores, percents=tuple(range(1, 11))):
        y_true = np.asarray(true).astype(int)

        y_scores = np.asarray(y_scores)
        if y_true.ndim > 1:
            y_true = y_true.reshape(-1)
        if y_scores.ndim > 1:
            y_scores = y_scores.reshape(-1)

        assert y_true.ndim == 1 and y_scores.ndim == 1 and y_true.size == y_scores.size
        N = y_true.size
        P = int(y_true.sum())
        pi = np.mean(y_true) if np.any(y_true) else 1e-12

        order = np.argsort(-y_scores, kind="mergesort")
        y_sorted = y_true[order]

        tp_cum = np.cumsum(y_sorted)
        ks = np.arange(1, N + 1)

        precision_at_k = tp_cum / ks
        recall_at_k = (tp_cum / max(P, 1)) if P > 0 else np.zeros_like(tp_cum)
        lift_at_k = precision_at_k / pi

        out = {}
        for r in percents:
            k = max(1, int(np.floor(r / 100.0 * N)))
            out[f"Precision@{r}%"] = float(precision_at_k[k - 1])
            out[f"Recall@{r}%"] = float(recall_at_k[k - 1])
            out[f"Lift@{r}%"] = float(lift_at_k[k - 1])
        aur, aul, aup = self.calculate_auc_from_topk(out)
        return  {"AUR_norm": float(aur)}

    @staticmethod
    def calculate_auc_from_topk(topk_dict):
        percents = list(range(1, 11))
        x = np.array(percents) / 100.0

        recall_values = [topk_dict[f"Recall@{r}%"] for r in percents]
        lift_values = [topk_dict[f"Lift@{r}%"] for r in percents]
        precision_values = [topk_dict[f"Precision@{r}%"] for r in percents]

        aur = auc(x, recall_values) / (x[-1] - x[0])
        aul = auc(x, lift_values) / (x[-1] - x[0])
        aup = auc(x, precision_values) / (x[-1] - x[0])

        return aur, aul, aup

    def norm_metric(self, y_true, y_prob, threshold=0.5):
        auc = Metrics.auc(y_true, y_prob)
        # acc = Metrics.acc(y_true, y_prob, threshold)
        # mcc = Metrics.mcc(y_true, y_prob, threshold)
        # pr_auc = Metrics.pr_auc(y_true, y_prob)
        # se = Metrics.sensitivity(y_true, y_prob, threshold)
        # sp = Metrics.specificity(y_true, y_prob, threshold)
        # bal_acc = Metrics.balanced_accuracy(y_true, y_prob, threshold)
        # f1 = Metrics.f1(y_true, y_prob, threshold)
        # precision = Metrics.precision(y_true, y_prob, threshold)

        top_k_precision = self.metric_aur(y_true, y_prob)

        task_metrics = {
            "AUC": np.around(auc, decimals=3),
            # "ACC": np.around(acc, decimals=3),
            # "MCC": np.around(mcc, decimals=3),
            # "PRAUC": np.around(pr_auc, decimals=3),
            # "PRE": np.around(precision, decimals=3),
            # "SE": np.around(se, decimals=3),
            # "SP": np.around(sp, decimals=3),
            # "Balanced ACC": np.around(bal_acc, decimals=3),
            # "F1": np.around(f1, decimals=3),
        }
        for key in top_k_precision:
            task_metrics[key] = top_k_precision[key]

        return task_metrics




class Record:
    def __init__(self, sub_model_num):
        """
        Initialize recording variables, support multi-task learning
        :param n_tasks: Number of tasks
        """
        self.metrics_tools = Metrics()
        self.cls_task = TaskClass(sub_model_num)
        self.sub_model_num = sub_model_num
        self.normal_task_idx = self.cls_task.norm_task_idx.numpy().tolist()
        self.pu_task_idx = self.cls_task.pu_task_idx.numpy().tolist()
        self.sub_full_idx = self.cls_task.loop_special_task_idx

        self.pu_task2prior = {p:pdx for pdx, p in enumerate(self.pu_task_idx)}
        self.prior = None
        self.n_tasks = None
        self.iter_train_loss = defaultdict(list)  # Training loss
        self.iter_valid_loss = defaultdict(list)  # Validation loss
        self.iter_test_loss = defaultdict(list)   # Test loss
        self.iter_train_pu_loss = defaultdict(list)
        self.iter_valid_pu_loss = defaultdict(list)

        # Labels and predictions for distinguishing train, valid, test
        # Each task stores labels and predictions separately
        self.train_labels = defaultdict(lambda: defaultdict(list))  # Training set true labels
        self.train_preds = defaultdict(lambda: defaultdict(list))   # Training set prediction probabilities
        self.valid_labels = defaultdict(lambda: defaultdict(list))  # Validation set true labels
        self.valid_preds = defaultdict(lambda: defaultdict(list))   # Validation set prediction probabilities
        self.test_labels = defaultdict(lambda: defaultdict(list))   # Test set true labels
        self.test_preds = defaultdict(lambda: defaultdict(list))    # Test set prediction probabilities

        # for_pu_task
        # self.train_pu_labels = defaultdict(lambda: defaultdict(list))  # Training set true labels
        # self.train_pu_preds = defaultdict(lambda: defaultdict(list))   # Training set prediction probabilities
        # self.valid_pu_labels = defaultdict(lambda: defaultdict(list))  # Validation set true labels
        # self.valid_pu_preds = defaultdict(lambda: defaultdict(list))   # Validation set prediction probabilities


        # Save evaluation metrics for each epoch
        self.metrics = defaultdict(lambda: defaultdict(dict))  # Save metrics for each task and epoch

    def update_train_pu_loss(self, epoch, loss):
        self.iter_train_pu_loss[epoch].append(loss)

    def update_valid_pu_loss(self, epoch, loss):
        self.iter_valid_pu_loss[epoch].append(loss)


    def update_train_loss(self, epoch, loss):
        """Update training loss"""
        self.iter_train_loss[epoch].append(loss)

    def update_valid_loss(self, epoch, loss):
        """Update validation loss"""
        self.iter_valid_loss[epoch].append(loss)

    def update_test_loss(self, epoch, loss):
        """Update test loss"""
        self.iter_test_loss[epoch].append(loss)

    def update_iter_results(self, epoch, labels, preds, mask=None, phase="train"):
        """
        Record loss, labels, predictions for each iteration
        :param epoch: Current epoch
        :param labels: True labels, shape [batch, n_tasks]
        :param preds: Prediction probabilities, shape [batch, n_tasks]
        :param phase: Data phase, train, valid, or test
        """
        samples, n_tasks = preds.shape  # Determine actual task count from output, e.g., 16 [normal + low]


        multi_task_idx = ( list(range(*self.cls_task.other_idx)) +
                list(range(*self.cls_task.inhibitor_idx)) +
                         list(range(*self.cls_task.substrate_idx)) +
                          list(range(*self.cls_task.inducer_idx)) * self.sub_model_num)

        multi_task_idx_pred = list(range(4,n_tasks))

        labels = labels[:, multi_task_idx]
        mask = mask[:, multi_task_idx]

        self.n_tasks = n_tasks
        # Store labels and predictions separately by phase and task

        for task_idx in multi_task_idx_pred:
            task_labels = labels[:, task_idx]  # Get current task labels
            task_preds = preds[:, task_idx]    # Get current task predictions
            if mask is not None:  # ?
                task_labels = task_labels[mask[:, task_idx]==1]
                task_preds = task_preds[mask[:, task_idx]==1]

            if len(task_labels.shape) == 1:
                task_labels = task_labels[:, None]

            if phase == "train":
                self.train_labels[epoch][task_idx].append(task_labels)
                self.train_preds[epoch][task_idx].append(task_preds)

            elif phase == "train_pu":
                self.train_labels[epoch][str(task_idx)+"pu"].append(task_labels)
                self.train_preds[epoch][str(task_idx)+"pu"].append(task_preds)

            elif phase == "valid":
                self.valid_labels[epoch][task_idx].append(task_labels)
                self.valid_preds[epoch][task_idx].append(task_preds)

            elif phase == "valid_pu":
                self.valid_labels[epoch][str(task_idx)+"pu"].append(task_labels)
                self.valid_preds[epoch][str(task_idx)+"pu"].append(task_preds)

            elif phase == "test":
                self.test_labels[epoch][task_idx].append(task_labels)
                self.test_preds[epoch][task_idx].append(task_preds)
            elif phase == "test_pu":
                self.test_labels[epoch][str(task_idx)+"pu"].append(task_labels)
                self.test_preds[epoch][str(task_idx)+"pu"].append(task_preds)

    def cal_metrics(self, epoch, threshold=0.5, phase="valid", is_pu=False, used_model=None):  # rewrite
        """
        Calculate and record evaluation metrics for current epoch
        :param epoch: Current epoch
        :param threshold: Threshold value, default 0.5
        :param phase: Data phase, train, valid, or test
        """
        # Get true labels and predictions based on phase
        if not is_pu:
            for task_idx in self.normal_task_idx:

                if phase == "train":
                    y_true = np.concatenate(self.train_labels[epoch][task_idx], axis=0)
                    y_prob = np.concatenate(self.train_preds[epoch][task_idx], axis=0)
                elif phase == "valid":
                    y_true = np.concatenate(self.valid_labels[epoch][task_idx], axis=0)
                    y_prob = np.concatenate(self.valid_preds[epoch][task_idx], axis=0)
                elif phase == "test":
                    y_true = np.concatenate(self.test_labels[epoch][task_idx], axis=0)
                    y_prob = np.concatenate(self.test_preds[epoch][task_idx], axis=0)
                else:
                    raise ValueError("phase must be 'train', 'valid', 'test'")
                # print(task_idx, y_true.shape, y_prob.shape)
                task_metrics = self.metrics_tools.norm_metric(y_true, y_prob, threshold=threshold)
                self.metrics[epoch][phase][f"task_{task_idx}"] = task_metrics

        else:
            for model_idx in range(used_model):
                pu_task_idx = self.sub_full_idx[model_idx].numpy().tolist()

                for pidx,task_idx in enumerate(pu_task_idx):

                    if phase == "train_pu":
                        y_true = np.concatenate(self.train_labels[epoch][str(task_idx)+"pu"], axis=0)
                        y_prob = np.concatenate(self.train_preds[epoch][str(task_idx)+"pu"], axis=0)
                    elif phase == "valid_pu":
                        y_true = np.concatenate(self.valid_labels[epoch][str(task_idx)+"pu"], axis=0)
                        y_prob = np.concatenate(self.valid_preds[epoch][str(task_idx)+"pu"], axis=0)
                    elif phase == "test_pu":
                        y_true = np.concatenate(self.test_labels[epoch][str(task_idx)+"pu"], axis=0)
                        y_prob = np.concatenate(self.test_preds[epoch][str(task_idx)+"pu"], axis=0)
                    else:
                        raise ValueError("phase must be 'train', 'valid', 'test'")

                    if len(y_true.shape) == 1:
                        y_true = y_true[:, None]
                    if len(y_prob.shape) == 1:
                        y_prob = y_prob[:, None]

                    y_true[y_true == -1] = 0
                    prior = self.prior[model_idx][pidx]
                    # print(task_idx, y_true.shape, y_prob.shape)
                    if "valid" in phase:
                        # joblib.dump((y_true, y_prob, prior), f"check_code_single/{epoch}_{task_idx}_valid_results.save")
                        # auu = Metrics.pu_auc(y_prob[y_true == 1], y_prob[y_true == 0], prior)

                        task_metrics = self.metrics_tools.norm_metric(y_true, y_prob, threshold=threshold)
                        task_metrics.update({"AUU": 0})

                        if task_idx > 17:
                            task_idx -= model_idx * 4
                        self.metrics[epoch][phase][f"task_{model_idx}_{task_idx}"] = task_metrics

                    elif "test" in phase:
                        # joblib.dump((y_true, y_prob, prior), f"check_code_single/{epoch}_{task_idx}_test_results.save")
                        # auu = Metrics.pu_auc(y_prob[y_true == 1], y_prob[y_true == 0], prior)
                        task_metrics = self.metrics_tools.norm_metric(y_true, y_prob, threshold=threshold)
                        task_metrics.update({"AUU": 0})

                        if task_idx > 17:
                            task_idx -= model_idx * 4
                        self.metrics[epoch][phase][f"task_{model_idx}_{task_idx}"] = task_metrics

    def get_metrics(self, epoch, phase="valid_pu"):
        """Get metrics for specified epoch and phase"""
        return self.metrics.get(epoch, {}).get(phase, None)

    def cal_pu_vote_metrics(self, epoch, phase="valid_pu", used_model=None, threshold=0.5):
        epoch_results = self.metrics.get(epoch, {}).get(phase, None)
        assert used_model is not None
        record_sep_results = defaultdict(dict)
        record_real_results = defaultdict(dict)
        record_prior_results = defaultdict(dict)

        for model_idx in range(used_model):
            pu_task_idx = self.sub_full_idx[model_idx].numpy().tolist()

            for pidx, task_idx in enumerate(pu_task_idx):

                if phase == "valid_pu":
                    y_true = np.concatenate(self.valid_labels[epoch][str(task_idx) + "pu"], axis=0)
                    y_prob = np.concatenate(self.valid_preds[epoch][str(task_idx) + "pu"], axis=0)
                elif phase == "test_pu":
                    y_true = np.concatenate(self.test_labels[epoch][str(task_idx) + "pu"], axis=0)
                    y_prob = np.concatenate(self.test_preds[epoch][str(task_idx) + "pu"], axis=0)
                else:
                    raise ValueError("phase must be 'train', 'valid', 'test'")

                if len(y_true.shape) == 1:
                    y_true = y_true[:, None]
                if len(y_prob.shape) == 1:
                    y_prob = y_prob[:, None]
                # print(task_idx, model_idx)
                task_idx -= model_idx * 4
                record_sep_results[task_idx][model_idx] = y_prob
                record_real_results[task_idx][model_idx] = y_true
                prior = self.prior[model_idx][pidx]
                record_prior_results[task_idx][model_idx] = prior

        # print(list(record_sep_results.keys()))
        # print(list(record_real_results[16].keys()))
        # joblib.dump([record_sep_results, record_real_results, record_prior_results],"k.joblib")
        for pidx, task_idx in enumerate(list(record_sep_results.keys())):
            prob_full = []
            prob_prior = []
            real_label = []
            for model_idx in range(used_model):
                prob_full.append(record_sep_results[task_idx][model_idx])
                prob_prior.append(record_prior_results[task_idx][model_idx])
                real_label.append(record_real_results[task_idx][model_idx])

            y_prob = np.mean(np.concatenate(prob_full, axis=-1), axis=-1)
            prior = np.mean(prob_prior)
            y_true = np.mean(np.concatenate(real_label, axis=-1), axis=-1)

            y_true[y_true == -1] = 0

            if "valid" in phase:
                # joblib.dump((y_true, y_prob, prior), f"check_code_single/{epoch}_{task_idx}_valid_results.save")
                # auu = Metrics.pu_auc(y_prob[y_true == 1], y_prob[y_true == 0], prior)
                task_metrics = self.metrics_tools.norm_metric(y_true, y_prob, threshold=threshold)
                # task_metrics.update({"AUU": np.around(auu, decimals=3)})

                phase = "valid_concat_pu"
                self.metrics[epoch][phase][f"task_{task_idx}"] = task_metrics

            elif "test" in phase:
                # joblib.dump((y_true, y_prob, prior), f"check_code_single/{epoch}_{task_idx}_test_results.save")
                # auu = Metrics.pu_auc(y_prob[y_true == 1], y_prob[y_true == 0], prior)
                task_metrics = self.metrics_tools.norm_metric(y_true, y_prob, threshold=threshold)
                # task_metrics.update({"AUU": np.around(auu, decimals=3)})

                phase = "test_concat_pu"
                self.metrics[epoch][phase][f"task{task_idx}"] = task_metrics

    def to_dict(self):
        """
        Extract all content from Record object as a dictionary for easy saving
        """
        return {
            "iter_train_loss": dict(self.iter_train_loss),
            "iter_valid_loss": dict(self.iter_valid_loss),
            "iter_test_loss": dict(self.iter_test_loss),
            "train_labels": dict(self.train_labels),
            "train_preds": dict(self.train_preds),
            "valid_labels": dict(self.valid_labels),
            "valid_preds": dict(self.valid_preds),
            "test_labels": dict(self.test_labels),
            "test_preds": dict(self.test_preds),

            "metrics": dict(self.metrics)
        }

    def __repr__(self):
        return f"Record(train_loss={len(self.iter_train_loss)}, valid_loss={len(self.iter_valid_loss)}, test_loss={len(self.iter_test_loss)})"
if __name__ == "__main__":
    pass