import torch


class TensorView(object):
    """
    A transform that returns a new view of a tensor.
    """

    def __init__(self, *view_dims):
        self.view_dims = view_dims

    def __call__(self, tensor: torch.Tensor):

        return tensor.view(*self.view_dims)



class BiasTrick(object):
    """
    A transform that applies the "bias trick": Adds an element equal to 1 to
    a given tensor.
    """

    def __call__(self, tensor: torch.Tensor):
        assert tensor.dim() == 1, "Only 1-d tensors supported"
        return torch.cat((tensor, torch.Tensor([1])), dim=0)



