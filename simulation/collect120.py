import experiments as E, numpy as np, random
from concurrent.futures import ProcessPoolExecutor as P
rng=random.Random(120)
jobs=[(s,l,b,rng.choice(['med','high']),120) for s in E.TRAIN_SEEDS for l in E.TRAIN_LOCS for b in (1,2)]
with P(2) as ex: parts=list(ex.map(E._collect,jobs))
# keep seed id per sample for grouped validation
X=np.concatenate([p[0] for p in parts]); y=np.concatenate([p[1] for p in parts])
g=np.concatenate([np.full(len(p[1]),j[0]) for p,j in zip(parts,jobs)])
np.savez_compressed('results/train_h120.npz',X=X,y=y,g=g); print('done',X.shape)
