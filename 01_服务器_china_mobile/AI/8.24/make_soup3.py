"""make_soup3.py — n-way 权重汤(8.25)。用法:
  python make_soup3.py out.tar ckptA ckptB [ckptC ...] --w 0.33,0.33,0.34
权重缺省均分。只平均 embedding_model(评测只需它)。"""
import sys, os, torch
args = sys.argv[1:]
ws = [1.0/len(args[:-1])] * (len(args)-1)
if '--w' in args:
    i = args.index('--w'); ws = [float(x) for x in args[i+1].split(',')]
    args = args[:i]
out, cks = args[0], args[1:]
assert len(ws) == len(cks) and abs(sum(ws)-1) < 1e-6
sts = [torch.load(c, map_location='cpu')['embedding_model'] for c in cks]
for s in sts[1:]:
    assert set(s) == set(sts[0]) and all(s[k].shape == sts[0][k].shape for k in s)
avg = {k: sum(w*s[k].float() for w, s in zip(ws, sts)).to(sts[0][k].dtype) for k in sts[0]}
os.makedirs(os.path.dirname(out), exist_ok=True)
torch.save({'embedding_model': avg, 'epoch': 'soup3', 'soup_sources': cks, 'soup_w': ws}, out)
print('saved', out, 'tensors:', len(avg), 'w:', ws)
