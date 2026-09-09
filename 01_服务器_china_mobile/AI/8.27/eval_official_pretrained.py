"""
eval_official_pretrained.py — 原始 ERes2Net 官方 ckpt 后端 × best TSE 前端 官方评测
（#18 口径闭合：任务书"原始 ReDimNet2/ERes2Net-large 骨干基线（不用任何改进方法）"
 + 我们的前端 = -30% 公式直接成立， soup5 是否算"改进方法"的口径争议彻底绕开）

与 8.21/evaluate_combo.py 的唯一差异：
  - 后端不装 robust/soup ckpt，直接装 pretrained 官方 eres2net_large_model.ckpt
    （load_model_for_validation(None, eres2net_ckpt) 的 pretrained 分支）
  - 自检判据：raw 模式 EER 必须复现官方基线 8.3911/11.0557（容差 0.05，
    装载等价性证明）；不复现则 FE 结果标 UNVERIFIED 不采信
其余（变长安全提取、缓存守卫、官方打分、汇总）全部复用 8.21 构件，与主链同源。
"""
import os
import sys
import json
import argparse
import torch
import toml

_HERE = os.path.dirname(os.path.abspath(__file__))
_V821 = os.path.join(os.path.dirname(_HERE), "8.21")
if _V821 not in sys.path:
    sys.path.insert(0, _V821)

from evaluate_combo import run_mode, extract_guarded, index_by_stem  # noqa: E402
from common_utils import setup_speakerlab                              # noqa: E402
from validate_robust import load_model_for_validation                  # noqa: E402
from evaluate_robust import get_wav_list                               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--enh-0db", required=True, help="noise_0db 增强后的 wav 目录")
    ap.add_argument("--enh-m5db", required=True, help="noise_m5db 增强后的 wav 目录")
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    cfg = toml.load(args.config)
    base = os.path.dirname(os.path.abspath(args.config))
    p_cfg = cfg["paths"]
    sv_ecapa = p_cfg["sv_ecapa"] if os.path.isabs(p_cfg["sv_ecapa"]) else \
        os.path.join(base, p_cfg["sv_ecapa"])
    speakerlab_root = os.path.dirname(os.path.dirname(os.path.dirname(sv_ecapa)))
    setup_speakerlab(p_cfg["sv_ecapa"], base)

    exp_dir = os.path.join(base, p_cfg["exp_dir"])
    eres2net_ckpt = p_cfg["eres2net_ckpt"]
    if not os.path.isabs(eres2net_ckpt):
        eres2net_ckpt = os.path.join(base, eres2net_ckpt)
    assert os.path.isfile(eres2net_ckpt), f"官方 ckpt 不存在: {eres2net_ckpt}"

    # ★ 唯一差异点：pretrained 官方基线分支（无 robust/soup ckpt）
    model = load_model_for_validation(None, eres2net_ckpt, device)
    ckpt_tag = "PRETRAINED_OFFICIAL::" + eres2net_ckpt   # 缓存守卫指纹

    test_root = os.path.join(sv_ecapa, "data", "data_for_test")
    trials_path = os.path.join(test_root, "clean", "3s", "trials_cross_device")
    assert os.path.isfile(trials_path), f"缺 trials: {trials_path}"
    out_root = os.path.join(exp_dir, "eval", "combo")
    os.makedirs(out_root, exist_ok=True)

    # ---- enrollment 侧：官方 clean/3s，同一个官方 ckpt，不过前端 ----
    clean_wavs = get_wav_list(os.path.join(test_root, "clean", "3s"), sv_ecapa)
    enrol_dir = os.path.join(out_root, "common", "enrol_clean")
    extract_guarded(model, clean_wavs, enrol_dir, device, args, ckpt_tag,
                    desc="enrol-clean(official pretrained)")

    enh_index = {
        "noise_0db": index_by_stem(
            args.enh_0db if os.path.isabs(args.enh_0db) else os.path.join(base, args.enh_0db)),
        "noise_m5db": index_by_stem(
            args.enh_m5db if os.path.isabs(args.enh_m5db) else os.path.join(base, args.enh_m5db)),
    }

    all_results = {}
    for mode in ("raw", "frontend"):
        print(f"\n===== 模式 {mode.upper()}：官方 pretrained ERes2Net 后端 =====")
        all_results[mode] = run_mode(mode, model, cfg, base, device, args, ckpt_tag,
                                     sv_ecapa, speakerlab_root, enrol_dir,
                                     trials_path, enh_index, out_root)

    bl = cfg.get("baseline", {})
    b0, b5 = bl["noise_0db"], bl["noise_m5db"]
    raw0 = all_results["raw"]["noise_0db"]["eer"]
    raw5 = all_results["raw"]["noise_m5db"]["eer"]
    fe0 = all_results["frontend"]["noise_0db"]["eer"]
    fe5 = all_results["frontend"]["noise_m5db"]["eer"]

    print("\n===== 原始 ERes2Net 官方后端 × TSE 前端（口径闭合评测） =====")
    ok0, ok5 = abs(raw0 - b0) < 0.05, abs(raw5 - b5) < 0.05
    rc = all_results["raw"].get("clean", {}).get("eer")
    print(f"raw 自检: 0dB {raw0:.4f} vs 官方 {b0} {'✓' if ok0 else '✗ MISMATCH'}; "
          f"-5dB {raw5:.4f} vs 官方 {b5} {'✓' if ok5 else '✗ MISMATCH'}; "
          f"clean {rc:.4f}（官方 {bl.get('clean')}）")
    if not (ok0 and ok5):
        print("!!! raw 未复现官方基线——装载不等价，FE 结果标 UNVERIFIED 不采信")
    d0, d5 = 100.0 * (b0 - fe0) / b0, 100.0 * (b5 - fe5) / b5
    print(f"frontend: 0dB {fe0:.4f} ({d0:+.1f}%)  -5dB {fe5:.4f} ({d5:+.1f}%)")
    ok30 = fe0 <= b0 * 0.7 and fe5 <= b5 * 0.7 and ok0 and ok5
    print(f"验收判据（原始基线 ≥30% 双档 + raw 复现）：{'达标 ✓' if ok30 else '未达标'}")

    summary = {"backend": "official_pretrained_eres2net",
               "eres2net_ckpt": eres2net_ckpt,
               "results": all_results, "baseline": bl,
               "selfcheck_raw_reproduces_baseline": bool(ok0 and ok5)}
    with open(os.path.join(out_root, "combo_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[done] {os.path.join(out_root, 'combo_summary.json')}")


if __name__ == "__main__":
    main()
