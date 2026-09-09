import sys, os, torch
A = "/home/ligangrui/china_mobile/AI/8.21/exp/proto_robust/checkpoints/checkpoint_best.tar"
B = "/home/ligangrui/china_mobile/AI/8.22/exp/matched_robust_90k/checkpoints/checkpoint_best.tar"
wa = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
out = sys.argv[2] if len(sys.argv) > 2 else "/home/ligangrui/china_mobile/AI/8.24/exp/soup_821_90k/checkpoints/checkpoint_soup_w%02d.tar" % round(wa*100)
ca = torch.load(A, map_location="cpu"); cb = torch.load(B, map_location="cpu")
print("A epoch:", ca.get("epoch"), " B epoch:", cb.get("epoch"))
sa, sb = ca["embedding_model"], cb["embedding_model"]
assert set(sa) == set(sb), "state dict key mismatch"
for k in sa: assert sa[k].shape == sb[k].shape, "shape mismatch: " + k
avg = {k: (wa * sa[k].float() + (1 - wa) * sb[k].float()).to(sa[k].dtype) for k in sa}
os.makedirs(os.path.dirname(out), exist_ok=True)
torch.save({"embedding_model": avg, "epoch": "soup_w%.2f" % wa, "soup_sources": [A, B]}, out)
print("saved", out, "tensors:", len(avg))
