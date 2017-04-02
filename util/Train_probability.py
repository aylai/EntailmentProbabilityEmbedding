import random

import numpy as np
import tensorflow as tf
import Probability
import math
from util.DataLoader import DataLoader
Sparse = DataLoader()

class Training:
    tf.set_random_seed(20160906)

    def __init__(self, sess, optimizer, mean_loss, x_loss, cpr_loss, x_predicted, y_predicted, xy_predicted, cpr_xy_predicted, cpr_xy_predicted_rev, dataset1, dataset2, x_labels, y_labels, xy_labels, cpr_xy_labels, lengths1, lengths2, batch_size, maxlength):
        self.sess = sess
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.mean_loss = mean_loss
        self.x_loss = x_loss
        self.cpr_loss = cpr_loss
        self.x_predicted = x_predicted
        self.y_predicted = y_predicted
        self.xy_predicted = xy_predicted
        self.cpr_xy_predicted = cpr_xy_predicted
        self.cpr_xy_predicted_rev = cpr_xy_predicted_rev
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.x_labels = x_labels
        self.y_labels = y_labels
        self.xy_labels = xy_labels
        self.cpr_xy_labels = cpr_xy_labels
        self.lengths1 = lengths1
        self.lengths2 = lengths2
        self.maxlength = maxlength

    def eval(self, data1, data2, x_label, y_label, xy_label, cpr_xy_label, lens1, lens2, limit=None):
        predictions = []
        x_predicted = []
        y_predicted = []
        xy_predicted = []
        cpr_xy_predicted = []
        indices = range(len(data1))
        if limit is not None:
            random.shuffle(indices)
        for i in range(len(data1)/self.batch_size):
            if limit is not None and (i + 1) * self.batch_size > limit:
                print('Reached eval limit')
                break
            ind = indices[self.batch_size*i:self.batch_size*(i+1)]
            D1 = []
            D2 = []
            for idx in ind:
                D1.append(data1[idx])
                D2.append(data2[idx])
            data1_pair = Sparse.pad_tensor(D1, self.maxlength)
            data2_pair = Sparse.pad_tensor(D2, self.maxlength)
            XL = x_label[ind]
            YL = y_label[ind]
            XYL = xy_label[ind]
            cpr_XYL = cpr_xy_label[ind]
            l1 = lens1[ind]
            l2 = lens2[ind]
            l, x_l, cpr_l, x_pred, y_pred, xy_pred, cpr_xy_pred, cpr_xy_pred_rev = self.sess.run([self.mean_loss, self.x_loss, self.cpr_loss, self.x_predicted, self.y_predicted, self.xy_predicted, self.cpr_xy_predicted, self.cpr_xy_predicted_rev],
                                    feed_dict={self.dataset1:data1_pair, self.dataset2:data2_pair, self.x_labels:XL, self.y_labels:YL, self.xy_labels:XYL, self.cpr_xy_labels:cpr_XYL, self.lengths1:l1, self.lengths2:l2})
            predictions.append(l)
            x_predicted.extend(x_pred)
            y_predicted.extend(y_pred)
            xy_predicted.extend(xy_pred)
            cpr_xy_predicted.extend(cpr_xy_pred)
        x_label = x_label[:len(x_predicted)]
        y_label = y_label[:len(y_predicted)]
        xy_label = xy_label[:len(xy_predicted)]
        cpr_xy_label = cpr_xy_label[:len(cpr_xy_predicted)]
        return predictions, np.mean(predictions), x_predicted, np.corrcoef(np.exp(x_predicted), x_label)[0, 1], y_predicted, np.corrcoef(np.exp(y_predicted), y_label)[0, 1], xy_predicted, np.corrcoef(np.exp(xy_predicted), xy_label)[0, 1], cpr_xy_predicted, np.corrcoef(np.exp(cpr_xy_predicted), cpr_xy_label)[0, 1], Probability.kl_divergence_batch(cpr_xy_predicted, cpr_xy_label)

    def predict(self, data1, data2, lens1, lens2):
        x_predictions = []
        y_predictions = []
        xy_predictions = []
        cpr_predictions = []
        cpr_predictions_rev = []
        indices = range(len(data1))
        padded = False
        for i in range((len(data1)/self.batch_size) + 1):
            while self.batch_size*(i+1) > len(indices):
                indices.append(indices[-1])
                padded = True
            ind = indices[self.batch_size*i:self.batch_size*(i+1)]
            data1_sub = []
            data2_sub = []
            for idx in ind:
                data1_sub.append(data1[idx])
                data2_sub.append(data2[idx])
            data1_pair = Sparse.pad_tensor(data1_sub, self.maxlength)
            data2_pair = Sparse.pad_tensor(data2_sub, self.maxlength)
            l1 = lens1[ind]
            l2 = lens2[ind]
            x_pred, y_pred, xy_pred, cpr_xy_pred, cpr_xy_pred_reverse = self.sess.run([self.x_predicted, self.y_predicted, self.xy_predicted, self.cpr_xy_predicted, self.cpr_xy_predicted_rev], feed_dict={self.dataset1:data1_pair, self.dataset2:data2_pair, self.lengths1:l1, self.lengths2:l2})
            x_predictions.extend(x_pred)
            y_predictions.extend(y_pred)
            xy_predictions.extend(xy_pred)
            cpr_predictions.extend(cpr_xy_pred)
            cpr_predictions_rev.extend(cpr_xy_pred_reverse)
        if padded:
            while len(cpr_predictions) > len(data1):
                x_predictions = x_predictions[:-1]
                y_predictions = y_predictions[:-1]
                xy_predictions = xy_predictions[:-1]
                cpr_predictions = cpr_predictions[:-1]
                cpr_predictions_rev = cpr_predictions_rev[:-1]
        return x_predictions, y_predictions, xy_predictions, cpr_predictions, cpr_predictions_rev

    def train(self, train1, train2, train_xlabels, train_ylabels, train_xylabels, train_cpr_xylabels, dev1, dev2, dev_xlabels, dev_ylabels, dev_xylabels, dev_cpr_xylabels, train_lens1=None, train_lens2=None, dev_lens1=None, dev_lens2=None, maxstep=None):
        total_loss = 0.0
        count = 0
        data_size = len(train1)
        if maxstep is not None:
            eval_limit = maxstep / 10
        else:
            eval_limit = None
        indices = range(data_size)
        random.shuffle(indices)
        print_step = data_size / 10
        for step in range(data_size/self.batch_size):
            if maxstep is not None and (step + 1) * self.batch_size > maxstep:
                print('Reached step limit')
                break
            while (step + 1) * self.batch_size >= count:
                print(str(count) + " "),
                count += print_step
            r_ind = indices[(step * self.batch_size):((step + 1) * self.batch_size)]
            batch_xlabels = train_xlabels[r_ind]
            batch_ylabels = train_ylabels[r_ind]
            batch_xylabels = train_xylabels[r_ind]
            batch_cpr_xylabels = train_cpr_xylabels[r_ind]
            batch_lens1 = train_lens1[r_ind]
            batch_lens2 = train_lens2[r_ind]
            train1_sub = []
            train2_sub = []
            for i in r_ind:
                train1_sub.append(train1[i])
                train2_sub.append(train2[i])
            training_1_pair = Sparse.pad_tensor(train1_sub, self.maxlength)
            training_2_pair = Sparse.pad_tensor(train2_sub, self.maxlength)
            feed_dict = {self.dataset1: training_1_pair, self.dataset2: training_2_pair, self.x_labels: batch_xlabels, self.y_labels: batch_ylabels, self.xy_labels: batch_xylabels, self.cpr_xy_labels: batch_cpr_xylabels, self.lengths1:batch_lens1, self.lengths2:batch_lens2}
            _, l, x_pred, y_pred, xy_pred, cpr_xy_pred = self.sess.run([self.optimizer, self.mean_loss, self.x_predicted, self.y_predicted, self.xy_predicted, self.cpr_xy_predicted], feed_dict=feed_dict)
            total_loss += l
        _, mean_loss, x_pred, x_corr, y_pred, y_corr, xy_pred, xy_corr, cpr_xy_pred, cpr_xy_corr, dev_cpr_kl = self.eval(dev1, dev2, dev_xlabels, dev_ylabels, dev_xylabels, dev_cpr_xylabels, dev_lens1, dev_lens2, limit=eval_limit)
        print('\ndev %f  %f  %f  %f  %f  %f  %f' % (total_loss, mean_loss, x_corr, y_corr, xy_corr, cpr_xy_corr, dev_cpr_kl))
        return dev_cpr_kl