import abc
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        loss = None

        x_scores_for_y_class = torch.gather(x_scores, dim=1,
                                            index=y.view(x_scores.shape[0], 1).expand(
                                                *x_scores.shape))
        m = x_scores - x_scores_for_y_class + self.delta

        # Subtract self.delta because it is added in m[i][j] for j = y[i] for every i.
        loss = torch.sum(torch.clamp(m, min=0.0)) / float(x_scores.shape[0]) - self.delta
        # ========================

        self.grad_ctx = {
            'x': x,
            'y': y,
            'm': m,
        }
        # ========================

        return loss

    def grad(self):


        grad = None

        m = self.grad_ctx['m']
        x = self.grad_ctx['x']
        y = self.grad_ctx['y']

        pos_m = torch.gt(m, torch.zeros(*m.shape))

        indices_of_true_class = y.view(m.shape[0], 1).expand(*m.shape)

        ones_at_true_class = torch.zeros(*m.shape).scatter_(dim=1, index=indices_of_true_class,
                                                            src=torch.ones(*m.shape))
        ones_at_wrong_class = torch.ones(*m.shape) - ones_at_true_class.float()

        sums_for_true_classes = torch.sum(
            torch.mul(ones_at_wrong_class, pos_m.float()), dim=1,
            keepdim=True)

        sum_of_indicators = torch.zeros(*m.shape).scatter_(dim=1, index=indices_of_true_class,
                                                           src=sums_for_true_classes.expand(
                                                               *m.shape))

        coefficients_for_wrong_classes = torch.mul(ones_at_wrong_class, pos_m.float())

        g = coefficients_for_wrong_classes - sum_of_indicators

        grad = torch.mm(torch.transpose(x, 0, 1), g) / float(m.shape[0])
        # ========================

        return grad
