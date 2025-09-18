import numpy as np

def num_to_vec(x, ff_num, ff_stride, ff_scale=1.0, ff_wrap=None):
    qs = np.linspace(0.0, 1.0, 2 * ff_num + 1, dtype=np.float32)
    qs = qs[..., ff_stride::ff_stride + 1]

    x = np.asarray(x)
    if x.ndim == 1:
        xN = x.reshape(-1, 1)
    elif x.ndim == 2:
        xN = x
    else:
        raise ValueError("x must be 1D or 2D array")

    # per-dimension quantiles over samples axis (axis=0)
    # np.quantile returns shape (len(qs), D) for 2D input
    p = np.quantile(xN, qs, axis=0)
    # transpose to (D, K)
    p = p.T.astype(np.float32)

    # compute distances for each dimension then concatenate along features
    parts = []
    for d in range(xN.shape[1]):
        p_d = p[d]  # (K,)
        diff = np.abs(xN[:, d][:, None] - p_d[None, :])
        if ff_wrap is not None:
            diff = np.minimum(diff, ff_wrap - diff)
        diff = (diff ** 2) * ff_scale
        parts.append(np.exp(-diff))
    s = np.concatenate(parts, axis=1)
    s = s / np.sum(s, axis=1, keepdims=True)

    return s, p

def vec_to_num(s, p, ff_wrap=None):
    # Reconstruct values from weights and quantile centers
    s = np.asarray(s)
    p = np.asarray(p, dtype=np.float32)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    D, K = p.shape
    if s.shape[1] != D * K:
        raise ValueError("s second dim must equal D*K")
    x_hat = np.empty((s.shape[0], D), dtype=np.float32)
    for d in range(D):
        s_d = s[:, d * K : (d + 1) * K]
        p_d = p[d]
        if ff_wrap is None:
            x_hat[:, d] = s_d @ p_d
        else:
            scale = (2.0 * np.pi) / float(ff_wrap)
            theta = p_d * scale
            c = np.cos(theta)
            si = np.sin(theta)
            C = s_d @ c
            S = s_d @ si
            ang = np.arctan2(S, C)
            x_hat[:, d] = ang * (float(ff_wrap) / (2.0 * np.pi))
    return x_hat

if __name__ == "__main__":
    # Main execution code
    x = np.linspace(-1.0, 1.0, 11, dtype=np.float16)
    x = np.expand_dims(x, -1)
    # x = np.concatenate((x, x + 1), axis=-1)
    s, p = num_to_vec(x, 5, 1, 20.0, 2.0)
    print(" ".join(f"{val:.3f}" for val in p.reshape(-1)))
    for v in s:
        print(" ".join(f"{val:.3f}" for val in v))

    # reconstruction demo
    x_rec = vec_to_num(s, p, ff_wrap=2.0)
    print("rec:")
    for v in x_rec:
        print(" ".join(f"{val:.3f}" for val in v))
    print("err:")
    for v in (x - x_rec):
        print(" ".join(f"{val:.3f}" for val in v))
