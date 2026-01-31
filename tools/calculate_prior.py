import tqdm
from ulcyp.downstream import TaskClass
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def estimate_prior_em(X_p, X_u, n_components=2, reg_covar=1e-3, clip_min=1e-6, clip_max=1-1e-6, standardize=True):
    X_p = X_p.astype(np.float64)
    X_u = X_u.astype(np.float64)

    if standardize:
        scaler = StandardScaler()
        X_all = scaler.fit_transform(np.vstack([X_p, X_u]))
        X_p = X_all[:len(X_p)]
        X_u = X_all[len(X_p):]

    gmm = GaussianMixture(n_components=n_components, reg_covar=reg_covar)
    gmm.fit(np.vstack([X_p, X_u]))

    log_prob_u = gmm.score_samples(X_u)
    log_prob_p = gmm.score_samples(X_p)

    prior = np.exp(np.mean(log_prob_u) - np.mean(log_prob_p))
    prior = np.clip(prior, clip_min, clip_max)
    return prior

def estimate_prior_elkan_noto(X_p, X_u, standardize=True, clf=None, random_state=42):
    X_p = X_p.astype(np.float64)
    X_u = X_u.astype(np.float64)
    Xp_tr, Xp_val = train_test_split(X_p, test_size=0.3, random_state=random_state, shuffle=True)

    X_tr = np.vstack([Xp_tr, X_u])
    y_tr = np.hstack([np.ones(len(Xp_tr)), np.zeros(len(X_u))])

    if standardize:
        scaler = StandardScaler().fit(X_tr)
        X_tr = scaler.transform(X_tr)
        Xp_val = scaler.transform(Xp_val)
        X_u_std = scaler.transform(X_u)
    else:
        X_u_std = X_u

    if clf is None:
        base = LogisticRegression(max_iter=1000, n_jobs=None)
        g = CalibratedClassifierCV(base, method='isotonic', cv=3)
    else:
        g = clf
    g.fit(X_tr, y_tr)

    c_hat = np.clip(g.predict_proba(Xp_val)[:,1].mean(), 1e-6, 1-1e-6)

    p_s = g.predict_proba(X_u_std)[:,1]
    p_y = np.clip(p_s / c_hat, 0.0, 1.0)
    pi_hat = p_y.mean()
    return float(pi_hat), float(c_hat)


def calculate_prior(embedding, mask, label, sub_num, cluster):
    print("Calculating prior for dataset...")
    task_tools = TaskClass(sub_num)

    pu_task_num = task_tools.loop_special_task_idx[cluster]
    mask = mask
    label = label
    embedding = embedding[:, pu_task_num, :]
    mask = mask[:, pu_task_num]
    label = label[:, pu_task_num]


    full_p = []

    for task in tqdm.tqdm(range(len(pu_task_num)), total=len(pu_task_num),
                          desc="Calculating prior for Task dataset..."):

        task_X_p = embedding[:, task, :][(mask[:, task]==1)&(label[:, task]==1)]
        task_X_u = embedding[:, task, :][(mask[:, task]==1)&(label[:, task]==-1)]


        p = estimate_prior_elkan_noto(task_X_p, task_X_u)
        full_p.append(p[0])
    return full_p

