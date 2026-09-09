# 修复 cryptography GLIBC 不兼容（离线）

## 问题

```
ImportError: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.33' not found
(required by .../cryptography/hazmat/bindings/_rust.abi3.so)
```

- 服务器 GLIBC = **2.31**（Ubuntu 20.04）
- 服务器 cryptography = **47.0.0**（需要 GLIBC 2.33）
- 根因：cryptography 被升级到 47.0.0，其 Rust 扩展针对 GLIBC 2.33 编译，与服务器不兼容

## 解决

降级到 `cryptography 40.0.2`（针对 manylinux_2_28 = GLIBC 2.28 编译，兼容 GLIBC 2.31）。

## 两段式流程（服务器无网络）

### 本地（有网络，Windows）

```bash
cd AI/8.14
bash download_crypto.sh
# 下载 cryptography 40.0.2 + cffi 到 crypto_wheels/
```

### 服务器（无网络）

```bash
# 把 AI/8.14/ 传到服务器
cd AI/8.14
bash install_crypto.sh
# 离线降级 cryptography 到 40.0.2

# 重新跑评测
cd ../code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa
bash evaluate_clean.sh
```

## 版本对照

| cryptography | manylinux 标签 | 需要 GLIBC | 服务器 2.31 |
|-------------|---------------|-----------|------------|
| 47.0.0 | manylinux_2_33 | 2.33 | ✗ 报错 |
| 40.0.2 | manylinux_2_28 | 2.28 | ✓ 可用 |
| 3.4.8 | manylinux2014 | 2.17 | ✓ 可用（更保守） |

如果 40.0.2 仍有问题，可改成 3.4.8（改 download_crypto.sh 里的版本号）。

## 为什么之前能跑、现在不行

之前某次 `pip install`（如装 silero-vad / soundfile）顺带把 cryptography 升级到了 47.0.0。
以后在服务器离线环境，避免 `pip install` 时用 `-U/--upgrade`，防止依赖被连带升级。
