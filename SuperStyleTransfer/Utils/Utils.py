import torch


def calc_gram_matrix(y: torch.FloatTensor) -> torch.FloatTensor:
    """
    calculate gram matrix for a network batch output
    :param y:
    :return:
    """
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram
