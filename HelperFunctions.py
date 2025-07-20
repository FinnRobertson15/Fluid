def normalize(x, mask=None, eps=1e-6):
    x_ = x if mask is None else x[mask]
    xMax = x_.max()
    xMin = x_.min()
    return (x - xMin).clamp(eps) / (xMax - xMin).clamp(eps)
