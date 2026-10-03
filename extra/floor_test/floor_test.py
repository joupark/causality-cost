"""Controlled test of Theorem 1's non-stationarity floor, in the theorem's own
model where (T1)-(T3) hold exactly.

Model: x(k) ~ N(0, theta(k)), theta in [1, 4]; target w*(theta)=Ss/(Ss+theta),
Ss=4.  Constants: kappa_I = 1/(2 theta_min^2) = 0.5 (Gaussian-variance KL bound),
g = Ss/(Ss+theta_max)^2 = 0.0625 (lower Lipschitz constant of w*).
Episodes follow the proof's hard pair: theta constant at theta0, or a ramp of
slope +-beta over the last m_beta samples ending at the evaluation time;
prior 1/2 each, theta0 ~ U[1+beta*m, 4-beta*m], random ramp sign.
beta is chosen so that m_beta = floor((kappa beta^2)^(-1/3)) = 8,16,32,64,128.

PREDICTIONS, written before running:
 P1 every estimator (incl. a network trained on unlimited fresh data) has
    average risk >= the exact two-point bound E[d^2/8] (d = |w*(theta0) -
    w*(theta0 +- beta m)|) and >= the global bound (g^2/32)(beta/kappa)^(2/3);
 P2 the best achievable risk scales as beta^(2/3) (log-log slope 2/3);
 P3 the best sliding-window length scales as beta^(-2/3);
 P4 enlarging the network's context from m_beta to 8 m_beta does not take
    its risk below the floor, and the longest windows do worse than the best.
"""
import os, sys, json, hashlib, numpy as np, torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__))
TMIN, TMAX, SS, KAPPA = 1.0, 4.0, 4.0, 0.5
G = SS / (SS + TMAX) ** 2
MS = [8, 16, 32, 64, 128]


def beta_of(m):
    return 1.0 / np.sqrt(KAPPA * m ** 3)


def wstar(th):
    return SS / (SS + th)


def rng(*p):
    return np.random.default_rng(int(hashlib.md5('_'.join(map(str, p)).encode()).hexdigest()[:8], 16))


def episodes(n, m, C, r):
    """Returns x^2 lags (n, C) with lag 0 = current sample, target, d."""
    beta = beta_of(m)
    th0 = r.uniform(TMIN + beta * m, TMAX - beta * m, n)
    hyp = r.integers(0, 2, n); sign = r.choice([-1.0, 1.0], n)
    lag = np.arange(C)
    ramp = beta * np.maximum(0, m - lag)                       # lag 0 -> beta*m
    th = th0[:, None] + (hyp * sign)[:, None] * ramp[None, :]
    x2 = th * r.standard_normal((n, C)) ** 2
    target = wstar(th[:, 0])
    d = np.abs(wstar(th0) - wstar(th0 + sign * beta * m))
    return x2.astype(np.float32), target.astype(np.float32), d


def feats(x2, C_in):
    dim = min(64, C_in); b = C_in // dim
    return x2[:, :C_in].reshape(len(x2), dim, b).mean(2)


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                               nn.Linear(128, 1))
    def forward(self, z):
        return self.f(z).squeeze(-1)


def train_mlp(m, C_in, epochs=40, n_ep=20000):
    torch.manual_seed(0)
    xv, tv, _ = episodes(10000, m, C_in, rng('val', m, C_in))
    fv = torch.from_numpy(feats(xv, C_in)); tv = torch.from_numpy(tv)
    net = MLP(fv.shape[1]); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    best, state = 1e9, None
    for ep in range(epochs):
        x, t, _ = episodes(n_ep, m, C_in, rng('train', m, C_in, ep))   # fresh data every epoch
        f = torch.from_numpy(feats(x, C_in)); t = torch.from_numpy(t)
        for i in range(0, n_ep, 256):
            opt.zero_grad(); loss = ((net(f[i:i+256]) - t[i:i+256]) ** 2).mean(); loss.backward(); opt.step()
        with torch.no_grad():
            v = ((net(fv) - tv) ** 2).mean().item()
        if v < best:
            best, state = v, {k: w.clone() for k, w in net.state_dict().items()}
    net.load_state_dict(state)
    return net


def run(m):
    beta = beta_of(m); Cmax = min(8 * m, 1024)
    x2, t, d = episodes(20000, m, Cmax, rng('test', m))
    out = dict(m=m, beta=beta, lb_global=(G ** 2 / 32) * (beta / KAPPA) ** (2 / 3),
               lb_exact=float(np.mean(d ** 2) / 8))
    cs = np.cumsum(x2, 1)
    Cs = sorted(set([c for c in [1, 2, 4] + list(range(8, Cmax + 1, max(1, Cmax // 64)))] + [Cmax]))
    win = {c: float(np.mean((wstar(cs[:, c - 1] / c) - t) ** 2)) for c in Cs}
    cbest = min(win, key=win.get)
    out.update(win_best=win[cbest], win_best_C=cbest, win_Cmax=win[Cmax], win_curve=win)
    lam_best = None; ew_best = 1e9
    for lam in 1 - np.logspace(-3.5, -0.3, 40):
        wts = lam ** np.arange(Cmax); wts /= wts.sum()
        r_ = float(np.mean((wstar(x2 @ wts) - t) ** 2))
        if r_ < ew_best: ew_best, lam_best = r_, float(lam)
    out.update(ewma_best=ew_best, ewma_eff_ctx=1 / (1 - lam_best))
    for C_in, key in [(m, 'mlp_short'), (Cmax, 'mlp_long')]:
        net = train_mlp(m, C_in)
        with torch.no_grad():
            out[key] = float(((net(torch.from_numpy(feats(x2, C_in))) - torch.from_numpy(t)) ** 2).mean())
    return out


if __name__ == '__main__':
    m = int(sys.argv[1])
    res = run(m)
    path = os.path.join(HERE, 'floor_results.json')
    allr = json.load(open(path)) if os.path.exists(path) else {}
    allr[str(m)] = res; json.dump(allr, open(path, 'w'), indent=1)
    print({k: (round(v, 8) if isinstance(v, float) else v) for k, v in res.items() if k != 'win_curve'})
