# README

MOE (Mixture of Experts) 模块通常被用于模型的 前馈子层（Feed-Forward Network，FFN）。
- FFN 模块通常占据 Transformer 参数的大头，替换成 MoE 能显著提升模型容量。
- MoE 模型只对每个 token 激活少数专家，计算开销接近普通 FFN，但模型可达到稠密模型数倍甚至万倍参数规模。

## 参考
- [BiliBili](https://www.bilibili.com/video/BV1ZbFpeHEYr/?spm_id_from=333.337.search-card.all.click&vd_source=ecc24dd03d67b9fec1e5e1e7f0f85646)