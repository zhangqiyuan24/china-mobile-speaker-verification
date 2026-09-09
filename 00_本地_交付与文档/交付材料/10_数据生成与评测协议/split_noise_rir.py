"""
噪声和混响的训测物理隔离脚本。
所有输出放到 data/divide/ 目录，不影响 data/ 根目录。

噪声：MUSAN 三类按 80/20 随机拆分
混响：模拟 RIR → 训练，真实 RIR → 测试
"""

import os, random

SEED = 42
random.seed(SEED)

OUT_DIR = "data/divide"
os.makedirs(OUT_DIR, exist_ok=True)


def split_noise():
    print("=" * 60)
    print("  噪声训测拆分 (MUSAN → 80% train / 20% test)")
    print("=" * 60)

    train_pool = []

    for category in ["music", "noise", "speech"]:
        cat_dir = os.path.join("data/musan", category)
        wavs = []
        for root, dirs, files in os.walk(cat_dir):
            for f in files:
                if f.endswith(".wav"):
                    wavs.append(os.path.join(root, f))
        random.shuffle(wavs)
        n_test = max(1, int(len(wavs) * 0.2))
        test_pool = wavs[:n_test]
        train_pool.extend(wavs[n_test:])

        # 每类噪声单独输出测试文件（供 gen_test_conditions.py 按类型加噪）
        test_file = os.path.join(OUT_DIR, f"noise_test_{category}.txt")
        with open(test_file, "w") as f:
            for p in test_pool:
                f.write(p + "\n")
        print(f"  {category}: {len(wavs)} 条 → train={len(wavs)-n_test}, test={n_test}")
        print(f"    → {test_file}")

    # 训练池仍合并（训练时不需要区分噪声类型）
    with open(os.path.join(OUT_DIR, "noise_train.scp"), "w") as f:
        for i, p in enumerate(train_pool):
            f.write(f"noise_{i:06d} {p}\n")

    print(f"  → {OUT_DIR}/noise_train.scp ({len(train_pool)} 条)")


def split_rir():
    print("\n" + "=" * 60)
    print("  混响训测拆分 (模拟RIR → train / 真实RIR → test)")
    print("=" * 60)

    # 训练：模拟 RIR。parts[4] 如 "RIRS_NOISES/simulated_rirs/smallroom/..."
    train_pool = []
    sim_dir = "data/rirs_noises/RIRS_NOISES/simulated_rirs"
    for room in ["smallroom", "mediumroom", "largeroom"]:
        rl = os.path.join(sim_dir, room, "rir_list")
        if not os.path.exists(rl):
            continue
        with open(rl) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    p = os.path.join("data/rirs_noises", parts[4])
                    if os.path.exists(p):
                        train_pool.append(p)

    # 测试：真实 RIR
    test_pool = []
    rl = "data/rirs_noises/RIRS_NOISES/real_rirs_isotropic_noises/rir_list"
    if os.path.exists(rl):
        with open(rl) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    p = os.path.join("data/rirs_noises", parts[4])
                    if os.path.exists(p):
                        test_pool.append(p)

    with open(os.path.join(OUT_DIR, "rir_train.scp"), "w") as f:
        for i, p in enumerate(train_pool):
            f.write(f"rir_{i:06d} {p}\n")
    with open(os.path.join(OUT_DIR, "rir_test.txt"), "w") as f:
        for p in test_pool:
            f.write(p + "\n")

    print(f"  → {OUT_DIR}/rir_train.scp ({len(train_pool)} 条)")
    print(f"  → {OUT_DIR}/rir_test.txt ({len(test_pool)} 条)")


if __name__ == "__main__":
    split_noise()
    split_rir()
    print(f"\n全部输出在 {OUT_DIR}/ 下")
