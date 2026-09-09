"""
生成 infer_sv_new.py 的辅助脚本。
如果后续需要再次生成或修改 infer_sv_new.py，可以直接复用此脚本的逻辑。

使用方法：
    python gen_infer_sv_new.py > speakerlab/bin/infer_sv_new.py
"""

# 此文件记录了 infer_sv_new.py 的设计要点和关键改动位置，
# 方便后续维护时快速定位需要修改的地方。

DESIGN_NOTES = """
infer_sv_new.py 改动清单（相对于原 infer_sv.py）：

1. 模型来源检测（第 225-229 行）
   - redimnet2/ 前缀 → _run_redimnet2()
   - iic/ 或 damo/ 前缀 → _run_modelscope()（原逻辑）

2. _run_modelscope() 函数
   - 完全复制原 infer_sv.py 的 main() 逻辑
   - 无任何改动

3. _run_redimnet2() 函数（新增）
   - 解析 model_id: redimnet2/<model_name>/<dataset>/<train_type>
   - 调用 _find_redimnet2_root() 自动定位 redimnet2 代码
   - 调用 _load_redimnet2_from_github() 下载并加载模型
   - compute_embedding 跳过 FBank，直接喂原始波形给模型

4. _dispatch_wavs() 函数（提取公共逻辑）
   - 原 infer_sv.py 第 297-336 行的 wav 分发逻辑
   - 两个路径共用，避免重复代码

5. _find_redimnet2_root() 函数（新增）
   - 自动搜索 redimnet2 代码位置
   - 支持 --redimnet2_path 手动指定

6. _load_redimnet2_from_github() 函数（新增）
   - 从 GitHub Releases 下载权重
   - 用 load_custom() 加载（架构参数自动从 checkpoint 读取）

如果需要新增其他 GitHub 来源的模型（非 ModelScope）：
   - 在 _run_redimnet2() 里加 model_id 解析分支
   - 或新增 _run_xxx() 函数，在 main() 里加前缀检测

如果需要新增 ReDimNet2 的模型变体：
   - 不需要改代码
   - 直接用 redimnet2/<model>/<dataset>/<train_type> 格式传参即可
   - 只要 PalabraAI/redimnet2 Releases 上有对应的 .pt 文件就能加载
"""

if __name__ == "__main__":
    print(DESIGN_NOTES)
