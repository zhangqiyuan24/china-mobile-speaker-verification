# -*- coding: utf-8 -*-
import io
p = "/data/home/ligangrui/china_mobile/AI/9.3/README_实验结果汇总.md"
s = io.open(p, encoding="utf-8").read()
b1_old = u"产物:(launch_fe_clean.sh、fe_clean_eres_3s/result.metrics)。"
b1_new = u"产物:`AI/9.3/fe_clean/`(launch_fe_clean.sh、fe_clean_eres_3s/result.metrics)。"
b2_old = u"产物:(wavlm_summary.json、scores/);训练脚本 、checkpoint 。"
b2_new = u"产物:`AI/9.3/wavlm_ft_official/`(wavlm_summary.json、scores/);训练脚本 `AI/8.24/wavlm_finetune.py`、checkpoint `AI/8.24/exp/wavlm_ft/checkpoints/`。"
assert b1_old in s, "b1 not found"
assert b2_old in s, "b2 not found"
s = s.replace(b1_old, b1_new).replace(b2_old, b2_new)
io.open(p, "w", encoding="utf-8").write(s)
print("fixed")
