"""Full-covariance GMM by EM on the GPU, with the four sklearn methods the pipeline uses."""
import math, numpy as np, torch
class GaussianMixtureGPU:
    def __init__(s, K, reg_covar=1e-3, n_init=2, iters=100, tol=1e-4, random_state=0, device='cuda', covariance_type='full'):
        s.K, s.reg, s.n_init, s.iters, s.tol, s.seed, s.dev = K, reg_covar, n_init, iters, tol, random_state, device
    def _logpdf(s, X):
        N, D = X.shape; out = torch.empty(N, s.K, device=s.dev, dtype=X.dtype)
        for k in range(s.K):
            L = s.L[k]; sol = torch.linalg.solve_triangular(L, (X - s.mu[k]).T, upper=False)
            out[:, k] = -0.5 * (sol ** 2).sum(0) - torch.log(torch.diagonal(L)).sum() - 0.5 * D * math.log(2 * math.pi)
        return out
    def _fit_once(s, X, g):
        N, D = X.shape; K = s.K
        idx = [int(torch.randint(N, (1,), generator=g, device=s.dev))]; d2 = ((X - X[idx[0]]) ** 2).sum(1)
        for _ in range(K - 1):
            idx.append(int(torch.multinomial(d2 / d2.sum(), 1, generator=g))); d2 = torch.minimum(d2, ((X - X[idx[-1]]) ** 2).sum(1))
        s.mu = X[idx].clone(); cov = torch.cov(X.T) + s.reg * torch.eye(D, device=s.dev)
        s.L = torch.linalg.cholesky(cov)[None].repeat(K, 1, 1); s.logw = torch.full((K,), -math.log(K), device=s.dev); prev = -1e30
        for it in range(s.iters):
            lp = s._logpdf(X) + s.logw; ll = torch.logsumexp(lp, 1); R = torch.exp(lp - ll[:, None])
            Nk = R.sum(0) + 1e-8; s.logw = torch.log(Nk / N); s.mu = (R.T @ X) / Nk[:, None]
            for k in range(K):
                d = X - s.mu[k]; s.L[k] = torch.linalg.cholesky((d.T * R[:, k]) @ d / Nk[k] + s.reg * torch.eye(D, device=s.dev))
            cur = float(ll.mean())
            if abs(cur - prev) < s.tol: break
            prev = cur
        return cur
    def fit(s, X):
        X = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=s.dev); best = None
        for i in range(s.n_init):
            g = torch.Generator(device=s.dev); g.manual_seed(s.seed + i); ll = s._fit_once(X, g)
            if best is None or ll > best[0]: best = (ll, s.mu.clone(), s.L.clone(), s.logw.clone())
        _, s.mu, s.L, s.logw = best; return s
    def _post(s, X):
        X = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=s.dev)
        return torch.cat([s._logpdf(X[i:i + 500000]) + s.logw for i in range(0, len(X), 500000)])
    def predict(s, X): return s._post(X).argmax(1).cpu().numpy()
    def predict_proba(s, X): return torch.softmax(s._post(X), 1).cpu().numpy()
    def score(s, X): return float(torch.logsumexp(s._post(X), 1).mean())
