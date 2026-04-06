# 团队使用说明

这个仓库只放代码，不放大体积数据、训练产物和日志。这些内容需要从团队共享网盘单独获取。

## 仓库里不包含的内容

以下内容不会上传到 GitHub：

- `dataset/humanml/`
- `dataset/humanml（官方）/`
- `dataset/t2m_train.npy`
- `dataset/t2m_test.npy`
- `checkpoints/`
- `logging/`

如果本地缺这些文件，请先从团队共享网盘下载。

## 本地目录结构

把网盘里的资源放到项目根目录后，目录结构应当大致如下：

```text
motion-diffusion-model-main/
├─ dataset/
│  ├─ humanml/
│  ├─ humanml_opt.txt
│  ├─ t2m_mean.npy
│  ├─ t2m_std.npy
│  ├─ t2m_train.npy
│  └─ t2m_test.npy
├─ body_models/
├─ checkpoints/
├─ model/
├─ train/
├─ eval/
└─ ...
```

## 环境配置

如果团队已经有现成环境，直接激活：

```powershell
conda activate mdm_clean
```

如果没有环境，就根据仓库里的 `environment.yml` 创建：

```powershell
conda env create -f environment.yml
conda activate mdm_clean
```

## 常用命令

Baseline HumanML 评测：

```powershell
python -m eval.eval_humanml --model_path checkpoints\baseline_trans_enc\model000021448.pt --eval_mode wo_mm --device 0
```

训练 HumanML `trans_enc` 模型：

```powershell
python -m train.train_mdm --save_dir save\my_humanml_trans_enc_512 --dataset humanml
```

生成动作样本：

```powershell
python -m sample.generate --model_path save\humanml_trans_enc_512\model000200000.pt --text_prompt "a person walks forward"
```

## 给队友的说明

- 官方原始项目说明请看 `README.md`。
- 本文件只负责团队内部使用说明。
- 如果评测支持断点续跑，日志会继续写在对应 checkpoint 目录下。
- 不要把数据集、checkpoint、日志、生成结果重新提交到 GitHub。
