"""
Color-Invariant Saree Design Recognition  (single-file, Kaggle-notebook friendly)

Expected data layout (one folder per DESIGN, each image = one colorway):
    ROOT/<design_id>/<any_name>.jpg|png
Nested folders are fine (ROOT/a/<design_id>/x.jpg): the design id is the image's folder path relative to ROOT.
If no root in Cfg.roots exists, a synthetic dataset is generated so the script runs as-is.

Pipeline: PK sampling -> heavy colour augmentation -> ResNet + GeM + 256-d BN embedding
          -> ArcFace + batch-hard triplet loss -> cosine retrieval / thresholded verification.
Split is by DESIGN ID (open-set): test designs are never seen in training.
"""
import math, random, time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms as T

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ----------------------------------------------------------------------------- config
@dataclass
class Cfg:
    roots: list = field(default_factory=lambda: ["/kaggle/input/indian-saree-patterns", "/data/deeplure"])
    synthetic_dir: str = "/tmp/synthetic_sarees"
    backbone: str = "resnet50"      # any torchvision ResNet: resnet18/34/50
    emb_dim: int = 256
    img_size: int = 224
    P: int = 16                      # designs per batch
    K: int = 4                       # images per design per batch
    iters_per_epoch: int = 60
    epochs: int = 15
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    arc_s: float = 30.0
    arc_m: float = 0.4
    tri_margin: float = 0.3
    gray_infer: bool = True          # drop colour entirely at test time
    gray_p: float = 0.5              # FIX: train-time grayscale prob; keep it high when gray_infer=True
    seed: int = 42
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def use_cuda(self):
        return str(self.device).startswith("cuda")


def seed_everything(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def autocast_ctx(device):
    """FIX: no more 'device_type=cuda but CUDA unavailable' warning on CPU."""
    use = str(device).startswith("cuda")
    return torch.autocast(device_type="cuda" if use else "cpu", enabled=use)


def load_rgb(path):
    """FIX: closes the file handle (thousands of open images otherwise)."""
    with Image.open(path) as im:
        return im.convert("RGB")


# ----------------------------------------------------------------------------- data
def make_synthetic(root, n_designs=60, colorways=4, size=224, seed=0):
    """Fake 'designs': each design = a fixed sinusoidal motif, rendered in several random 2-colour palettes."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size] / size
    for d in range(n_designs):
        out = Path(root) / f"design_{d:03d}"
        out.mkdir(parents=True, exist_ok=True)
        fx, fy, ph = rng.uniform(2, 10, 3), rng.uniform(2, 10, 3), rng.uniform(0, 6.28, 3)
        mask = sum(np.sin(2 * np.pi * (fx[i] * xx + fy[i] * yy) + ph[i]) for i in range(3))[..., None] > 0
        for c in range(colorways):
            c1, c2 = rng.integers(0, 256, 3), rng.integers(0, 256, 3)
            Image.fromarray(np.where(mask, c1, c2).astype(np.uint8)).save(out / f"cw{c}.png")


def scan_designs(roots):
    """-> {design_id: [image paths]}.
    FIX: design id = folder path RELATIVE to root. Before, only the immediate parent name was used, so
    'train/x' and 'test/x' silently merged, and a flat folder collapsed into a single design."""
    designs = defaultdict(list)
    for r in roots:
        r = Path(r)
        for p in sorted(r.rglob("*")):
            if p.suffix.lower() in IMG_EXT:
                designs[f"{r.name}/{p.parent.relative_to(r).as_posix()}"].append(str(p))
    designs = dict(designs)
    if len(designs) < 10:
        raise ValueError(f"Found only {len(designs)} design folder(s). Expected ROOT/<design_id>/<image>; "
                         f"need at least ~10 designs for a train/val/test split.")
    return designs


def split_by_design(designs, seed, frac=(0.7, 0.1, 0.2)):
    """Open-set split: disjoint design IDs in train / val / test."""
    ids = sorted(designs); random.Random(seed).shuffle(ids)
    a, b = int(frac[0] * len(ids)), int((frac[0] + frac[1]) * len(ids))
    return ({k: designs[k] for k in ids[:a]}, {k: designs[k] for k in ids[a:b]}, {k: designs[k] for k in ids[b:]})


class ChannelShuffle:
    """Randomly permute RGB channels (a cheap way to fake arbitrary colourways)."""
    def __init__(self, p=0.5): self.p = p
    def __call__(self, x):
        return x[torch.randperm(3)] if random.random() < self.p else x


def train_tf(size, gray_p=0.5):
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.4, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.6, 0.6, 0.8, 0.5)], p=0.9),   # extreme jitter, hue up to +-0.5
        T.RandomGrayscale(p=gray_p),
        T.RandomInvert(p=0.1),
        T.ToTensor(),
        ChannelShuffle(0.5),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


def eval_tf(size, gray):
    ops = [T.Resize((size, size))]
    if gray: ops.append(T.Grayscale(3))
    ops += [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return T.Compose(ops)


class SareeDataset(Dataset):
    """Returns (augmented image, design label). Pair/triplet structure comes from the PK sampler."""
    def __init__(self, designs, tf):
        self.ids = sorted(designs); self.id2l = {k: i for i, k in enumerate(self.ids)}
        self.items = [(p, self.id2l[k]) for k, ps in designs.items() for p in ps]
        self.tf = tf
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        for _ in range(5):                       # FIX: a corrupt file no longer kills the whole epoch
            p, y = self.items[i]
            try:
                return self.tf(load_rgb(p)), y
            except Exception as e:
                print(f"[warn] unreadable image {p}: {e}")
                i = random.randrange(len(self.items))
        raise RuntimeError("Too many unreadable images in a row")
    @property
    def labels(self): return [y for _, y in self.items]


class PKBatchSampler(Sampler):
    """P designs x K images. A design with < K images is sampled with replacement: the SAME image gets
    different random colour augmentations, i.e. synthetic colorways act as positives."""
    def __init__(self, labels, P, K, iters):
        self.by = defaultdict(list)
        for i, l in enumerate(labels): self.by[l].append(i)
        self.cls, self.P, self.K, self.iters = list(self.by), P, K, iters
    def __iter__(self):
        for _ in range(self.iters):
            batch = []
            for c in random.sample(self.cls, min(self.P, len(self.cls))):
                pool = self.by[c]
                batch += random.sample(pool, self.K) if len(pool) >= self.K else random.choices(pool, k=self.K)
            yield batch
    def __len__(self): return self.iters


# ----------------------------------------------------------------------------- model + losses
class GeM(nn.Module):
    """Generalised-mean pooling: keeps texture statistics better than plain average pooling."""
    def __init__(self, p=3.0): super().__init__(); self.p = nn.Parameter(torch.tensor(p))
    def forward(self, x):
        p = self.p.clamp(1.0, 6.0)               # FIX: keep p in a sane range
        x = x.float().clamp(min=1e-6)            # FIX: pow() in fp32 -> no fp16 overflow
        return F.avg_pool2d(x.pow(p), x.shape[-2:]).pow(1.0 / p).flatten(1)


class SareeEmbedder(nn.Module):
    def __init__(self, name="resnet50", emb_dim=256):
        super().__init__()
        try:
            net = torchvision.models.get_model(name, weights="DEFAULT")   # ImageNet weights (needs internet)
        except Exception as e:
            print(f"[warn] pretrained weights unavailable ({e}); using random init")
            net = torchvision.models.get_model(name, weights=None)
        self.body = nn.Sequential(*list(net.children())[:-2])
        self.pool, self.fc = GeM(), nn.Linear(net.fc.in_features, emb_dim, bias=False)
        self.bn = nn.BatchNorm1d(emb_dim)
    def forward(self, x):
        return F.normalize(self.bn(self.fc(self.pool(self.body(x)))), dim=-1)


class ArcFace(nn.Module):
    """Additive angular margin softmax over training design IDs."""
    def __init__(self, emb_dim, n_cls, s=30.0, m=0.4):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_cls, emb_dim)); nn.init.xavier_uniform_(self.W)
        self.s, self.m = s, m
        self.th, self.mm = math.cos(math.pi - m), math.sin(math.pi - m) * m
    def forward(self, emb, y):
        cos = F.linear(emb, F.normalize(self.W)).clamp(-1 + 1e-7, 1 - 1e-7)
        phi = cos * math.cos(self.m) - (1 - cos ** 2).sqrt() * math.sin(self.m)   # cos(theta + m)
        phi = torch.where(cos > self.th, phi, cos - self.mm)
        logits = torch.where(F.one_hot(y, cos.size(1)).bool(), phi, cos) * self.s
        return F.cross_entropy(logits, y)


def batch_hard_triplet(emb, y, margin):
    """Hardest positive / hardest negative per anchor, cosine distance."""
    d = 1 - emb @ emb.t()
    same = y[:, None] == y[None, :]
    eye = torch.eye(len(y), dtype=torch.bool, device=y.device)
    hp = d.masked_fill(~(same & ~eye), -1e4).max(1).values
    hn = d.masked_fill(same, 1e4).min(1).values
    return F.relu(hp - hn + margin).mean()


# ----------------------------------------------------------------------------- embedding + inference
@torch.no_grad()
def embed_paths(model, paths, tf, device, bs=64):
    """Post-processing: horizontal-flip TTA average -> L2 norm."""
    model.eval(); out = []
    for i in range(0, len(paths), bs):
        x = torch.stack([tf(load_rgb(p)) for p in paths[i:i + bs]]).to(device)
        with autocast_ctx(device):
            e = model(x) + model(torch.flip(x, dims=[3]))
        out.append(F.normalize(e.float(), dim=-1).cpu())
    return torch.cat(out)


class Recognizer:
    """Deployment wrapper: identification (retrieval) + verification (thresholded cosine)."""
    def __init__(self, model, tf, device, threshold=0.5):
        self.model, self.tf, self.device, self.thr = model, tf, device, threshold
    def build_gallery(self, paths, names=None):
        self.g_paths, self.g_names = paths, names or paths
        self.g_emb = embed_paths(self.model, paths, self.tf, self.device)
    def identify(self, query_path, topk=5):
        q = embed_paths(self.model, [query_path], self.tf, self.device)
        sims = (q @ self.g_emb.t()).squeeze(0)
        s, idx = sims.topk(min(topk, len(sims)))
        return [(self.g_names[i], float(v)) for v, i in zip(s, idx)]
    def verify(self, path_a, path_b):
        e = embed_paths(self.model, [path_a, path_b], self.tf, self.device)
        s = float(e[0] @ e[1]); return s, s >= self.thr


# ----------------------------------------------------------------------------- evaluation
def query_gallery_split(designs, seed):
    """One random image (colorway) per design = query; the rest = gallery. Single-image designs
    stay in the gallery as distractors. Query and its matches therefore always differ in colourway."""
    rng = random.Random(seed); q, g = [], []
    for k, ps in sorted(designs.items()):
        ps = list(ps)
        if len(ps) >= 2:
            qi = rng.randrange(len(ps)); q.append((ps.pop(qi), k))
        g += [(p, k) for p in ps]
    return q, g


def identification_metrics(qe, ql, ge, gl):
    """Rank-1 / Rank-5 / mAP over the whole gallery ranking."""
    order = (qe @ ge.t()).argsort(1, descending=True)
    match = (gl[order] == ql[:, None]).float()
    ranks = torch.arange(1, match.size(1) + 1)
    ap = ((match.cumsum(1) / ranks) * match).sum(1) / match.sum(1).clamp(min=1)
    return dict(rank1=match[:, 0].mean().item(), rank5=match[:, :5].amax(1).mean().item(), mAP=ap.mean().item())


_COLOR_CACHE = {}
def mean_rgb(paths):
    """FIX: cached, so per-epoch validation no longer re-reads every image just to get mean colour."""
    for p in paths:
        if p not in _COLOR_CACHE:
            _COLOR_CACHE[p] = np.asarray(load_rgb(p).resize((16, 16))).reshape(-1, 3).mean(0)
    return np.stack([_COLOR_CACHE[p] for p in paths])


def build_pairs(paths, labels, n_pos=2000, seed=0):
    """Positives: same design (different colourway). Negatives: random pairs AND 'hard' pairs = different
    design but the closest mean colour (tests that the model does not cheat with colour)."""
    rng = random.Random(seed); labels = np.asarray(labels)
    by = defaultdict(list)
    for i, l in enumerate(labels): by[l].append(i)
    pos = [(a, b) for v in by.values() for a in v for b in v if a < b]
    if not pos:
        raise ValueError("No positive pairs: every design has a single image.")
    rng.shuffle(pos); pos = pos[:n_pos]
    rand_neg = []
    while len(rand_neg) < len(pos):
        a, b = rng.randrange(len(paths)), rng.randrange(len(paths))
        if labels[a] != labels[b]: rand_neg.append((a, b))
    col = mean_rgb(paths); hard_neg = []
    for a in rng.sample(range(len(paths)), min(len(pos), len(paths))):
        d = np.linalg.norm(col - col[a], axis=1); d[labels == labels[a]] = 1e9
        hard_neg.append((a, int(d.argmin())))
    return pos, rand_neg, hard_neg


def verification_metrics(emb, pos, rand_neg, hard_neg, thr=None):
    def sim(prs):
        a, b = zip(*prs)
        return (emb[list(a)] * emb[list(b)]).sum(1).numpy()      # FIX: vectorised (was a Python loop)
    sp, sr, sh = sim(pos), sim(rand_neg), sim(hard_neg)
    res = {}
    for name, sn in [("random_neg", sr), ("hard_same_colour_neg", sh)]:
        y, s = np.r_[np.ones(len(sp)), np.zeros(len(sn))], np.r_[sp, sn]
        fpr, tpr, _ = roc_curve(y, s)
        res[f"AUC_{name}"] = roc_auc_score(y, s)
        res[f"TAR@FAR1%_{name}"] = float(np.interp(0.01, fpr, tpr))
    if thr is None:
        # FIX: threshold is tuned on random + hard negatives together (before: random only, then scored on
        # hard negatives only -> optimistic threshold that over-accepts same-colour impostors).
        y, s = np.r_[np.ones(len(sp)), np.zeros(len(sr) + len(sh))], np.r_[sp, sr, sh]
        fpr, tpr, th = roc_curve(y, s)
        res["thr"] = float(th[(tpr - fpr).argmax()])             # Youden J
    else:
        res["acc@thr_random"] = float(((sp >= thr).sum() + (sr < thr).sum()) / (len(sp) + len(sr)))
        res["acc@thr_hard"] = float(((sp >= thr).sum() + (sh < thr).sum()) / (len(sp) + len(sh)))
    return res


def evaluate(model, designs, cfg, thr=None, verification=True):
    """FIX: every image is embedded ONCE; query/gallery/pair metrics all index into the same matrix
    (before: images were embedded twice per call, and this runs every epoch)."""
    tf = eval_tf(cfg.img_size, cfg.gray_infer)
    ids = {k: i for i, k in enumerate(sorted(designs))}
    allp = [p for k in sorted(designs) for p in designs[k]]
    alll = [ids[k] for k in sorted(designs) for _ in designs[k]]
    emb = embed_paths(model, allp, tf, cfg.device)
    labels = torch.tensor(alll)
    pidx = {p: i for i, p in enumerate(allp)}

    q, g = query_gallery_split(designs, cfg.seed)
    if not q:
        raise ValueError("Evaluation split has no design with >= 2 images; cannot build queries.")
    qi, gi = [pidx[p] for p, _ in q], [pidx[p] for p, _ in g]
    out = identification_metrics(emb[qi], labels[qi], emb[gi], labels[gi])
    if verification:
        pos, rn, hn = build_pairs(allp, alll, seed=cfg.seed)
        out.update(verification_metrics(emb, pos, rn, hn, thr))
    return out


# ----------------------------------------------------------------------------- efficiency report
def efficiency_report(model, cfg):
    from torch.utils.flop_counter import FlopCounterMode
    model.eval(); x = torch.randn(1, 3, cfg.img_size, cfg.img_size, device=cfg.device)
    with FlopCounterMode(display=False) as fc, torch.no_grad(): model(x)
    with torch.no_grad():
        for _ in range(5): model(x)
        if cfg.use_cuda: torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(30): model(x)
        if cfg.use_cuda: torch.cuda.synchronize()
    return dict(params_M=sum(p.numel() for p in model.parameters()) / 1e6,
                GFLOPs=fc.get_total_flops() / 1e9,          # FIX: was /2e9 (that is GMACs, not GFLOPs)
                latency_ms=(time.perf_counter() - t) / 30 * 1e3, embedding_dim=cfg.emb_dim)


# ----------------------------------------------------------------------------- training
def train(cfg):
    seed_everything(cfg.seed)
    roots = [r for r in cfg.roots if Path(r).exists()]
    if not roots:
        print("[info] no real data found -> generating synthetic dataset")
        make_synthetic(cfg.synthetic_dir); roots = [cfg.synthetic_dir]
    designs = scan_designs(roots)
    tr, va, te = split_by_design(designs, cfg.seed)
    print(f"designs train/val/test = {len(tr)}/{len(va)}/{len(te)}  images = {sum(map(len, designs.values()))}")
    if len(va) < 2 or len(te) < 2:
        raise ValueError("Validation/test sets need at least 2 designs each; add more designs.")

    ds = SareeDataset(tr, train_tf(cfg.img_size, cfg.gray_p))
    dl = DataLoader(ds, batch_sampler=PKBatchSampler(ds.labels, cfg.P, cfg.K, cfg.iters_per_epoch),
                    num_workers=cfg.num_workers, pin_memory=cfg.use_cuda)
    model = SareeEmbedder(cfg.backbone, cfg.emb_dim).to(cfg.device)
    arc = ArcFace(cfg.emb_dim, len(ds.ids), cfg.arc_s, cfg.arc_m).to(cfg.device)
    opt = torch.optim.AdamW([
        {"params": model.body.parameters(), "lr": cfg.lr_backbone},
        {"params": list(model.pool.parameters()) + list(model.fc.parameters()) + list(model.bn.parameters()) + list(arc.parameters()),
         "lr": cfg.lr_head}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[cfg.lr_backbone, cfg.lr_head],
                                                total_steps=cfg.epochs * cfg.iters_per_epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_cuda)
    best, best_state = -1, None
    for ep in range(cfg.epochs):
        model.train(); tot = 0
        for x, y in dl:
            x, y = x.to(cfg.device), y.to(cfg.device)
            with autocast_ctx(cfg.device):
                e = model(x)
            e = e.float()
            loss = arc(e, y) + batch_hard_triplet(e, y, cfg.tri_margin)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        m = evaluate(model, va, cfg, verification=False)     # FIX: per-epoch val only needs retrieval metrics
        print(f"ep {ep + 1:02d} loss {tot / len(dl):.3f} | val mAP {m['mAP']:.3f} rank1 {m['rank1']:.3f}")
        if m["mAP"] >= best:
            best, best_state = m["mAP"], {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    val_m = evaluate(model, va, cfg)            # threshold is tuned on VAL designs only
    test_m = evaluate(model, te, cfg, thr=val_m["thr"])
    print("VAL :", {k: round(v, 4) for k, v in val_m.items()})
    print("TEST:", {k: round(v, 4) for k, v in test_m.items()})
    print("EFFICIENCY:", {k: round(v, 3) for k, v in efficiency_report(model, cfg).items()})
    torch.save({"model": model.state_dict(), "thr": val_m["thr"], "cfg": cfg.__dict__}, "saree_embedder.pt")
    return model, te, val_m["thr"]


if __name__ == "__main__":
    cfg = Cfg()
    model, test_designs, thr = train(cfg)
    # ---- demo inferenc