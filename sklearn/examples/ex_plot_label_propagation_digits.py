# -*- coding: utf-8 -*-
"""
 Created on 2021/7/10 0:13
 Filename   : ex_plot_label_propagation_digits.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012/PythonSamples
 Description:
"""

# =======================================================
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

from sklearn import datasets
from sklearn.semi_supervised import LabelSpreading

from sklearn.metrics import confusion_matrix, classification_report


def plot_label_propagation_digits():
    digits = datasets.load_digits()
    rng = np.random.RandomState(2)
    indices = np.arange(len(digits.data))
    rng.shuffle(indices)

    X = digits.data[indices[:340]]
    y = digits.target[indices[:340]]
    images = digits.images[indices[:340]]

    n_total_samples = len(y)
    n_labeled_points = 40

    indices = np.arange(n_total_samples)

    unlabeled_set = indices[n_labeled_points:]

    # #############################################################################
    # Shuffle everything around
    y_train = np.copy(y)
    y_train[unlabeled_set] = -1

    # #############################################################################
    # Learn with LabelSpreading
    lp_model = LabelSpreading(gamma=.25, max_iter=20)
    lp_model.fit(X, y_train)
    predicted_labels = lp_model.transduction_[unlabeled_set]
    true_labels = y[unlabeled_set]

    cm = confusion_matrix(true_labels, predicted_labels, labels=lp_model.classes_)

    print("Label Spreading model: %d labeled & %d unlabeled points (%d total)" %
          (n_labeled_points, n_total_samples - n_labeled_points, n_total_samples))

    print(classification_report(true_labels, predicted_labels))

    print("Confusion matrix")
    print(cm)

    # #############################################################################
    # Calculate uncertainty values for each transduced distribution
    pred_entropies = stats.distributions.entropy(lp_model.label_distributions_.T)

    # #############################################################################
    # Pick the top 10 most uncertain labels
    uncertainty_index = np.argsort(pred_entropies)[-10:]

    # #############################################################################
    # Plot
    f = plt.figure(figsize=(7, 5))
    for index, image_index in enumerate(uncertainty_index):
        image = images[image_index]

        sub = f.add_subplot(2, 5, index + 1)
        sub.imshow(image, cmap=plt.cm.gray_r)
        plt.xticks([])
        plt.yticks([])
        sub.set_title('predict: %i\ntrue: %i' % (
            lp_model.transduction_[image_index], y[image_index]))

    f.suptitle('Learning with small amount of labeled data')
    plt.show()


if __name__ == '__main__':
    plot_label_propagation_digits()
