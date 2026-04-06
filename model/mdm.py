import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from model.rotation2xyz import Rotation2xyz
from model.BERT.BERT_encoder import load_bert
from utils.misc import WeightedSum


class MDM(nn.Module):  # 定义主模型 MDM
    def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,  # 初始化基础输入参数
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,  # 主干网络超参数
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,  # 数据和模型配置
                 arch='trans_enc', emb_trans_dec=False, clip_version=None, **kargs):  # 架构、decoder 时间 token 选项和扩展参数
        super().__init__()  # 调用父类初始化

        self.legacy = legacy  # 保存旧版兼容标记
        self.modeltype = modeltype  # 保存模型类型名
        self.njoints = njoints  # 保存关节数量
        self.nfeats = nfeats  # 保存每个关节的特征维度
        self.num_actions = num_actions  # 保存动作类别总数
        self.data_rep = data_rep  # 保存动作表示方式
        self.dataset = dataset  # 保存数据集名称

        self.pose_rep = pose_rep  # 保存姿态表示方式
        self.glob = glob  # 保存是否处理全局信息
        self.glob_rot = glob_rot  # 保存是否处理全局旋转
        self.translation = translation  # 保存是否包含平移

        self.latent_dim = latent_dim  # 保存隐藏层维度

        self.ff_size = ff_size  # 保存 Transformer FFN 维度
        self.num_layers = num_layers  # 保存层数
        self.num_heads = num_heads  # 保存多头注意力头数
        self.dropout = dropout  # 保存 dropout 比例

        self.ablation = ablation  # 保存消融实验配置
        self.activation = activation  # 保存激活函数类型
        self.clip_dim = clip_dim  # 保存文本编码输出维度
        self.action_emb = kargs.get('action_emb', None)  # 读取动作嵌入配置
        self.input_feats = self.njoints * self.nfeats  # 计算每帧展平后的特征维度

        self.normalize_output = kargs.get('normalize_encoder_output', False)  # 是否归一化 encoder 输出

        self.cond_mode = kargs.get('cond_mode', 'no_cond')  # 条件模式，默认无条件
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)  # 条件随机 mask 概率
        self.mask_frames = kargs.get('mask_frames', False)  # 是否对帧使用 mask
        self.arch = arch  # 保存主干网络类型
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0  # 如果是 GRU，则给输入额外拼上条件嵌入通道
        self.input_process = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)  # 构建输入预处理模块

        self.emb_policy = kargs.get('emb_policy', 'add')  # 文本和时间嵌入的融合策略
        self.use_temporal_tcn = kargs.get('use_temporal_tcn', False)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout, max_len=kargs.get('pos_embed_max_len', 5000))  # 构建位置编码器
        self.emb_trans_dec = emb_trans_dec  # 保存 decoder 是否拼时间 token 的开关

        self.pred_len = kargs.get('pred_len', 0)  # 读取预测长度
        self.context_len = kargs.get('context_len', 0)  # 读取上下文前缀长度
        self.total_len = self.pred_len + self.context_len  # 计算总长度
        self.is_prefix_comp = self.total_len > 0  # 判断是否在做 prefix completion
        self.all_goal_joint_names = kargs.get('all_goal_joint_names', [])  # 读取所有目标关节名字
        
        self.multi_target_cond = kargs.get('multi_target_cond', False)  # 是否启用多目标关节条件
        self.multi_encoder_type = kargs.get('multi_encoder_type', 'multi')  # 多目标条件编码器类型
        self.target_enc_layers = kargs.get('target_enc_layers', 1)  # 多目标条件编码器层数
        if self.multi_target_cond:  # 如果启用了多目标条件
            if self.multi_encoder_type == 'multi':  # 如果用 multi 编码方式
                self.embed_target_cond = EmbedTargetLocMulti(self.all_goal_joint_names, self.latent_dim)  # 构建 multi 编码器
            elif self.multi_encoder_type == 'single':  # 如果用 single 编码方式
               self.embed_target_cond = EmbedTargetLocSingle(self.all_goal_joint_names, self.latent_dim, self.target_enc_layers)  # 构建 single 编码器
            elif self.multi_encoder_type == 'split':  # 如果用 split 编码方式
               self.embed_target_cond = EmbedTargetLocSplit(self.all_goal_joint_names, self.latent_dim, self.target_enc_layers)  # 构建 split 编码器
        
        if self.arch == 'trans_enc':  # 如果主干是 Transformer Encoder
            print("TRANS_ENC init")  # 打印初始化信息
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,  # 定义单层 encoder
                                                              nhead=self.num_heads,  # 设置注意力头数
                                                              dim_feedforward=self.ff_size,  # 设置前馈层维度
                                                              dropout=self.dropout,  # 设置 dropout
                                                              activation=self.activation)  # 设置激活函数

            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,  # 堆叠 encoder 层
                                                         num_layers=self.num_layers)  # 指定堆叠层数
            # ===修改部分===
            if self.use_temporal_tcn:
                self.temporal_tcn = LightweightTemporalTCN(self.latent_dim, dropout=self.dropout, dilations=(1, 2, 4))
        elif self.arch == 'trans_dec':  # 如果主干是 Transformer Decoder
            print("TRANS_DEC init")  # 打印初始化信息
            seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,  # 定义单层 decoder
                                                              nhead=self.num_heads,  # 设置注意力头数
                                                              dim_feedforward=self.ff_size,  # 设置前馈层维度
                                                              dropout=self.dropout,  # 设置 dropout
                                                              activation=activation)  # 设置激活函数
            self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,  # 堆叠 decoder 层
                                                         num_layers=self.num_layers)  # 指定堆叠层数
        elif self.arch == 'gru':  # 如果主干是 GRU
            print("GRU init")  # 打印初始化信息
            self.gru = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)  # 构建 GRU
        else:  # 如果架构类型不支持
            raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')  # 直接报错

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)  # 构建 diffusion timestep 编码器

        if self.cond_mode != 'no_cond':  # 如果不是无条件模型
            if 'text' in self.cond_mode:  # 如果条件里包含文本
                # We support CLIP encoder and DistilBERT  # 说明支持的文本编码器
                print('EMBED TEXT')  # 打印文本条件初始化信息
                
                self.text_encoder_type = kargs.get('text_encoder_type', 'clip')  # 读取文本编码器类型
                
                if self.text_encoder_type == "clip":  # 如果用 CLIP
                    print('Loading CLIP...')  # 打印加载信息
                    self.clip_version = clip_version  # 保存 CLIP 版本
                    self.clip_model = self.load_and_freeze_clip(clip_version)  # 加载并冻结 CLIP
                    self.encode_text = self.clip_encode_text  # 绑定 CLIP 编码函数
                elif self.text_encoder_type == 'bert':  # 如果用 BERT
                    assert self.arch == 'trans_dec'  # BERT 路线要求 decoder 架构
                    # assert self.emb_trans_dec == False # passing just the time embed so it's fine  # 原作者留下的旧注释
                    print("Loading BERT...")  # 打印加载信息
                    # bert_model_path = 'model/BERT/distilbert-base-uncased'  # 旧路径写法
                    bert_model_path = 'distilbert/distilbert-base-uncased'  # 当前使用的 BERT 路径
                    self.clip_model = load_bert(bert_model_path)  # 复用 clip_model 名字存 BERT 模型以兼容旧代码
                    self.encode_text = self.bert_encode_text  # 绑定 BERT 编码函数
                    self.clip_dim = 768  # BERT 输出维度是 768
                else:  # 如果传入了未知文本编码器
                    raise ValueError('We only support [CLIP, BERT] text encoders')  # 直接报错
                
                self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)  # 把文本特征投影到 latent_dim
                
            if 'action' in self.cond_mode:  # 如果条件里包含动作标签
                self.embed_action = EmbedAction(self.num_actions, self.latent_dim)  # 构建动作嵌入模块
                print('EMBED ACTION')  # 打印动作条件初始化信息

        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,  # 构建输出后处理模块
                                            self.nfeats)  # 补充每关节特征维度

        self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)  # 构建 rot2xyz 工具

    def parameters_wo_clip(self):  # 返回不含 CLIP 的参数
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]  # 过滤掉 clip_model 参数

    def load_and_freeze_clip(self, clip_version):  # 加载并冻结 CLIP
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',  # 从 CLIP 库加载模型
                                                jit=False)  # 训练时要求 jit=False
        clip.model.convert_weights(  # 转换 CLIP 权重格式
            clip_model)  # 实际上通常不是必须步骤

        # Freeze CLIP weights  # 下面冻结 CLIP 权重
        clip_model.eval()  # 切到 eval 模式
        for p in clip_model.parameters():  # 遍历全部参数
            p.requires_grad = False  # 禁止梯度更新

        return clip_model  # 返回冻结后的模型

    def mask_cond(self, cond, force_mask=False):  # 条件 mask 函数
        bs = cond.shape[-2]  # 取 batch size
        if force_mask:  # 如果强制无条件
            return torch.zeros_like(cond)  # 直接返回全零条件
        elif self.training and self.cond_mask_prob > 0.:  # 如果训练中且设置了随机 mask 概率
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(1, bs, 1)  # 为每个样本采样是否丢弃条件
            return cond * (1. - mask)  # 把被 mask 的条件变为 0
        else:  # 否则不处理
            return cond  # 原样返回条件

    def clip_encode_text(self, raw_text):  # 使用 CLIP 编码文本
        # raw_text - list (batch_size length) of strings with input text prompts  # 输入是一组文本 prompt
        device = next(self.parameters()).device  # 获取当前模型设备
        max_text_len = 20 if self.dataset in ['humanml', 'kit'] else None  # 对 humanml 和 kit 强行截断到 20 token
        if max_text_len is not None:  # 如果当前数据集需要特殊处理文本长度
            default_context_length = 77  # CLIP 默认上下文长度
            context_length = max_text_len + 2 # start_token + 20 + end_token  # 额外加起止 token
            assert context_length < default_context_length  # 保证不超过默认最大长度
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate  # 先按短长度 tokenize
            # print('texts', texts.shape)  # 调试打印 token 形状
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)  # 手动补齐剩余长度
            texts = torch.cat([texts, zero_pad], dim=1)  # 拼接补零后的 token
            # print('texts after pad', texts.shape, texts)  # 调试打印补齐结果
        else:  # 如果不需要特殊长度处理
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate  # 直接按默认长度 tokenize
        return self.clip_model.encode_text(texts).float().unsqueeze(0)  # 编码文本并转成 [1, bs, dim]
    
    def bert_encode_text(self, raw_text):  # 使用 BERT 编码文本
        # enc_text = self.clip_model(raw_text)  # 旧实现写法
        # enc_text = enc_text.permute(1, 0, 2)  # 旧实现中的维度变换
        # return enc_text  # 旧实现直接返回编码
        enc_text, mask = self.clip_model(raw_text)  # self.clip_model.get_last_hidden_state(raw_text, return_mask=True)  # 获取 token 特征和有效位 mask
        enc_text = enc_text.permute(1, 0, 2)  # 变成 [seq, bs, dim] 以适配 transformer
        mask = ~mask  # mask: True means no token there, we invert since the meaning of mask for transformer is inverted  https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html  # 反转 mask 语义适配 transformer
        return enc_text, mask  # 返回文本编码和 mask

    def forward(self, x, timesteps, y=None):  # 前向传播
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper  # 输入动作张量
        timesteps: [batch_size] (int)  # diffusion 时间步
        """
        bs, njoints, nfeats, nframes = x.shape  # 解析输入形状
        time_emb = self.embed_timestep(timesteps)  # [1, bs, d]  # 计算 diffusion 时间步嵌入

        if 'target_cond' in y.keys():  # 如果给了目标条件
            # NOTE: We don't use CFG for joints - but we do wat to support uncond sampling for generation and eval!  # 说明 joint 条件的 CFG 处理方式
            time_emb += self.mask_cond(self.embed_target_cond(y['target_cond'], y['target_joint_names'], y['is_heading'])[None], force_mask=y.get('target_uncond', False))  # 将目标条件编码后加到时间嵌入上
            # time_emb += self.embed_target_cond(y['target_cond'], y['target_joint_names'], y['is_heading'])[None]  # 不加 mask 的旧写法

        # Build input for prefix completion  # 下面处理前缀补全任务
        if self.is_prefix_comp:  # 如果当前任务是 prefix completion
            x = torch.cat([y['prefix'], x], dim=-1)  # 把 prefix 和待预测部分沿时间维拼接
            y['mask'] = torch.cat([torch.ones([bs, 1, 1, self.context_len], dtype=y['mask'].dtype, device=y['mask'].device),  # 给 prefix 创建全 1 mask
                                   y['mask']], dim=-1)  # 再和原 mask 拼接

        force_mask = y.get('uncond', False)  # 获取是否强制无条件生成
        if 'text' in self.cond_mode:  # 如果启用文本条件
            if 'text_embed' in y.keys():  # caching option  # 如果上游已经缓存好文本特征
                enc_text = y['text_embed']  # 直接复用缓存
            else:  # 否则需要现场编码
                enc_text = self.encode_text(y['text'])  # 调用对应文本编码器
            if type(enc_text) == tuple:  # 如果返回的是 (特征, mask)，说明是 BERT
                enc_text, text_mask = enc_text  # 拆分文本特征和文本 mask
                if text_mask.shape[0] == 1 and bs > 1:  # casting mask for the single-prompt-for-all case  # 单条 prompt 共享给整个 batch 的情况
                    text_mask = torch.repeat_interleave(text_mask, bs, dim=0)  # 复制 mask 到整个 batch
            text_emb = self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))  # 对文本条件做 mask 后映射到 latent 维度
            if self.emb_policy == 'add':  # 如果采用相加融合
                emb = text_emb + time_emb  # 直接把文本嵌入和时间嵌入相加
            else:  # 如果采用拼接融合
                emb = torch.cat([time_emb, text_emb], dim=0)  # 把时间 token 拼在文本 token 前面
                text_mask = torch.cat([torch.zeros_like(text_mask[:, 0:1]), text_mask], dim=1)  # 给新增时间 token 拼一个 False mask
        if 'action' in self.cond_mode:  # 如果启用动作条件
            action_emb = self.embed_action(y['action'])  # 获取动作嵌入
            emb = time_emb + self.mask_cond(action_emb, force_mask=force_mask)  # 把动作嵌入和时间嵌入相加
        if self.cond_mode == 'no_cond':  # 如果完全无条件
            # unconstrained  # 注释说明这是无约束生成
            emb = time_emb  # 条件仅为 timestep

        if self.arch == 'gru':  # 如果主干是 GRU
            x_reshaped = x.reshape(bs, njoints*nfeats, 1, nframes)  # 把关节和特征展平并保留时间维
            emb_gru = emb.repeat(nframes, 1, 1)     #[#frames, bs, d]  # 把条件嵌入复制到每一帧
            emb_gru = emb_gru.permute(1, 2, 0)      #[bs, d, #frames]  # 调整维度顺序
            emb_gru = emb_gru.reshape(bs, self.latent_dim, 1, nframes)  #[bs, d, 1, #frames]  # reshape 成可拼到输入上的形式
            x = torch.cat((x_reshaped, emb_gru), axis=1)  #[bs, d+joints*feat, 1, #frames]  # 把条件嵌入拼到输入通道上

        x = self.input_process(x)  # 将输入序列编码到 latent space

        # TODO - move to collate  # TODO：mask 逻辑后续可移动到数据整理阶段
        frames_mask = None  # 默认没有帧 mask
        is_valid_mask = y['mask'].shape[-1] > 1  # Don't use mask with the generate script  # 只有长度大于 1 的 mask 才认为有效
        if self.mask_frames and is_valid_mask:  # 如果启用了帧 mask 并且 mask 有效
            frames_mask = torch.logical_not(y['mask'][..., :x.shape[0]].squeeze(1).squeeze(1)).to(device=x.device)  # 把有效帧 mask 转换成 transformer 的 padding mask
            if self.emb_trans_dec or self.arch == 'trans_enc':  # 如果序列最前面还有额外 token
                step_mask = torch.zeros((bs, 1), dtype=torch.bool, device=x.device)  # 为额外 token 创建一个不屏蔽的 mask 位
                frames_mask = torch.cat([step_mask, frames_mask], dim=1)  # 把它拼到帧 mask 前面

        if self.arch == 'trans_enc':  # 如果主干是 transformer encoder
            # ===原代码（已注释）===
            # # adding the timestep embed  # 把 timestep/条件 token 拼到输入序列开头
            # xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]  # 形成完整输入序列
            # xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]  # 给序列加入位置编码
            # output = self.seqTransEncoder(xseq, src_key_padding_mask=frames_mask)[1:]  # , src_key_padding_mask=~maskseq)  # 编码后丢掉最前面的条件 token 输出

            # ===修改部分===
            if self.use_temporal_tcn:
                x = self.temporal_tcn(x)  # [seqlen, bs, d]  # 在 Transformer temporal encoder 前加入轻量级 TCN
            xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            output = self.seqTransEncoder(xseq, src_key_padding_mask=frames_mask)[1:]

        elif self.arch == 'trans_dec':  # 如果主干是 transformer decoder
            if self.emb_trans_dec:  # 如果 decoder 目标序列前面也拼时间 token
                xseq = torch.cat((time_emb, x), axis=0)  # 把时间嵌入拼到目标序列前面
            else:  # 否则不拼
                xseq = x  # 直接用动作序列
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]  # 给 decoder 输入加入位置编码

            if self.text_encoder_type == 'clip':  # 如果文本编码器是 CLIP
                output = self.seqTransDecoder(tgt=xseq, memory=emb, tgt_key_padding_mask=frames_mask)  # 用 emb 作为 memory 做 decoder 计算
            elif self.text_encoder_type == 'bert':  # 如果文本编码器是 BERT
                output = self.seqTransDecoder(tgt=xseq, memory=emb, memory_key_padding_mask=text_mask, tgt_key_padding_mask=frames_mask)  # 额外传入文本 memory 的 mask
            else:  # 如果文本编码器类型未知
                raise ValueError()  # 直接报错

            if self.emb_trans_dec:  # 如果前面拼了时间 token
                output = output[1:] # [seqlen, bs, d]  # 把时间 token 对应输出裁掉

        elif self.arch == 'gru':  # 如果主干是 GRU
            xseq = x  # 直接使用编码后的输入序列
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen, bs, d]  # 给序列加位置编码
            output, _ = self.gru(xseq)  # 通过 GRU 得到输出序列

        # Extract completed suffix  # 提取真正要预测的后缀部分
        if self.is_prefix_comp:  # 如果是 prefix completion
            output = output[self.context_len:]  # 去掉前缀对应的输出
            y['mask'] = y['mask'][..., self.context_len:]  # 同步裁剪 mask
        
        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]  # 把 latent 输出还原为动作格式
        return output  # 返回最终结果


    def _apply(self, fn):  # 重写 _apply 以同步处理内部 SMPL 模型
        super()._apply(fn)  # 对当前模块应用 fn
        self.rot2xyz.smpl_model._apply(fn)  # 对 rot2xyz 内部 smpl 也应用 fn


    def train(self, *args, **kwargs):  # 重写 train 以同步切换内部 SMPL 模型状态
        super().train(*args, **kwargs)  # 切换当前模块状态
        self.rot2xyz.smpl_model.train(*args, **kwargs)  # 切换内部 smpl 状态


class PositionalEncoding(nn.Module):  # 标准正弦位置编码模块
    def __init__(self, d_model, dropout=0.1, max_len=5000):  # 初始化位置编码器
        super(PositionalEncoding, self).__init__()  # 调用父类初始化
        self.dropout = nn.Dropout(p=dropout)  # 创建 dropout 层

        pe = torch.zeros(max_len, d_model)  # 创建位置编码表
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # 创建位置索引列向量
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))  # 计算不同频率的缩放项
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维用 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维用 cos
        pe = pe.unsqueeze(0).transpose(0, 1)  # 调整成 [max_len, 1, d_model]

        self.register_buffer('pe', pe)  # 注册为 buffer，随模型移动但不参与训练

    def forward(self, x):  # 前向计算位置编码
        # not used in the final model  # 原注释说明这个接口不是最终模型重点
        x = x + self.pe[:x.shape[0], :]  # 把位置编码加到输入上
        return self.dropout(x)  # 经过 dropout 后返回


class TimestepEmbedder(nn.Module):  # diffusion timestep 编码模块
    def __init__(self, latent_dim, sequence_pos_encoder):  # 初始化 timestep 编码器
        super().__init__()  # 调用父类初始化
        self.latent_dim = latent_dim  # 保存隐藏维度
        self.sequence_pos_encoder = sequence_pos_encoder  # 保存位置编码器引用

        time_embed_dim = self.latent_dim  # 时间嵌入内部维度
        self.time_embed = nn.Sequential(  # 构建两层 MLP
            nn.Linear(self.latent_dim, time_embed_dim),  # 第一层线性映射
            nn.SiLU(),  # 激活函数
            nn.Linear(time_embed_dim, time_embed_dim),  # 第二层线性映射
        )

    def forward(self, timesteps):  # 输入 batch 的 timestep
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)  # 取对应 pe 并映射成 [1, bs, d]


# ===修改部分===
class LightweightTemporalTCN(nn.Module):
    def __init__(self, latent_dim, dropout=0.1, dilations=(1, 2, 4)):
        super().__init__()
        self.latent_dim = latent_dim
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(latent_dim, latent_dim, kernel_size=3, padding=dilation, dilation=dilation),
                nn.SiLU(),
                nn.Dropout(dropout)
            )
            for dilation in dilations
        ])
        self.out_norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        # x: [seqlen, bs, d]
        x_conv = x.permute(1, 2, 0)  # [bs, d, seqlen]
        residual = x_conv
        for block in self.blocks:
            x_conv = block(x_conv) + residual
            residual = x_conv
        x = x_conv.permute(2, 0, 1)  # [seqlen, bs, d]
        x = self.out_norm(x)
        return x


class InputProcess(nn.Module):  # 输入动作预处理模块
    def __init__(self, data_rep, input_feats, latent_dim):  # 初始化输入处理器
        super().__init__()  # 调用父类初始化
        self.data_rep = data_rep  # 保存输入表示方式
        self.input_feats = input_feats  # 保存输入维度
        self.latent_dim = latent_dim  # 保存目标隐藏维度
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)  # 姿态输入线性映射层
        if self.data_rep == 'rot_vel':  # 如果是 rot_vel 表示
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)  # 给速度部分单独建映射层

    def forward(self, x):  # 输入处理前向传播
        bs, njoints, nfeats, nframes = x.shape  # 解析输入张量形状
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)  # 变成 [时间, batch, 展平特征]

        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:  # 如果是常规动作表示
            x = self.poseEmbedding(x)  # [seqlen, bs, d]  # 直接映射到 latent 空间
            return x  # 返回结果
        elif self.data_rep == 'rot_vel':  # 如果是 rot_vel 表示
            first_pose = x[[0]]  # [1, bs, 150]  # 取第一帧姿态
            first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]  # 第一帧走 pose 映射
            vel = x[1:]  # [seqlen-1, bs, 150]  # 后续帧作为速度项
            vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]  # 速度项走 vel 映射
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]  # 拼回完整序列
        else:  # 如果输入表示非法
            raise ValueError  # 直接报错


class OutputProcess(nn.Module):  # 输出后处理模块
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):  # 初始化输出处理器
        super().__init__()  # 调用父类初始化
        self.data_rep = data_rep  # 保存输出表示方式
        self.input_feats = input_feats  # 保存输出展平维度
        self.latent_dim = latent_dim  # 保存隐藏维度
        self.njoints = njoints  # 保存关节数量
        self.nfeats = nfeats  # 保存每关节特征维度
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)  # 把 latent 还原为姿态特征
        if self.data_rep == 'rot_vel':  # 如果是 rot_vel 表示
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)  # 给速度项单独建输出层

    def forward(self, output):  # 输出还原前向传播
        nframes, bs, d = output.shape  # 解析输出序列形状
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:  # 如果是常规表示
            output = self.poseFinal(output)  # [seqlen, bs, 150]  # 直接映射回原特征维
        elif self.data_rep == 'rot_vel':  # 如果是 rot_vel 表示
            first_pose = output[[0]]  # [1, bs, d]  # 取出第一帧
            first_pose = self.poseFinal(first_pose)  # [1, bs, 150]  # 第一帧按姿态输出
            vel = output[1:]  # [seqlen-1, bs, d]  # 剩余帧作为速度项
            vel = self.velFinal(vel)  # [seqlen-1, bs, 150]  # 速度项映射回特征维
            output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]  # 拼回完整序列
        else:  # 如果表示方式非法
            raise ValueError  # 直接报错
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)  # 恢复关节和特征维
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]  # 调整成模型标准输出格式
        return output  # 返回最终动作张量


class EmbedAction(nn.Module):  # 动作类别嵌入模块
    def __init__(self, num_actions, latent_dim):  # 初始化动作嵌入
        super().__init__()  # 调用父类初始化
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))  # 创建可训练的动作嵌入表

    def forward(self, input):  # 动作嵌入前向传播
        idx = input[:, 0].to(torch.long)  # an index array must be long  # 取出动作索引并转成长整型
        output = self.action_embedding[idx]  # 查表得到动作嵌入
        return output  # 返回动作嵌入
    
class EmbedTargetLocSingle(nn.Module):  # single 版本目标位置编码器
    def __init__(self, all_goal_joint_names, latent_dim, num_layers=1):  # 初始化 single 编码器
        super().__init__()  # 调用父类初始化
        self.extended_goal_joint_names = all_goal_joint_names + ['traj', 'heading']  # 扩展目标列表，加上 traj 和 heading
        self.target_cond_dim = len(self.extended_goal_joint_names) * 4  # 4 => (x,y,z,is_valid)  # 每个目标占 4 维
        self.latent_dim = latent_dim  # 保存隐藏维度
        _layers = [nn.Linear(self.target_cond_dim, self.latent_dim)]  # 第一层把全部目标条件展平后映射到 latent
        for _ in range(num_layers):  # 继续追加若干层 MLP
            _layers += [nn.SiLU(), nn.Linear(self.latent_dim, self.latent_dim)]  # 每层是 SiLU 加线性层
        self.mlp = nn.Sequential(*_layers)  # 构建顺序 MLP

    def forward(self, input, target_joint_names, target_heading):  # single 编码器前向传播
        # TODO - generate validity from outside the model  # TODO：validity 最好由外部预先生成
        validity = torch.zeros_like(input)[..., :1]  # 初始化 validity 标记
        for sample_idx, sample_joint_names in enumerate(target_joint_names):  # 遍历 batch 中每个样本
            sample_joint_names_w_heading = np.append(sample_joint_names, 'heading') if target_heading[sample_idx] else sample_joint_names  # 如有需要则补上 heading
            for j in sample_joint_names_w_heading:  # 遍历当前样本的有效目标
                validity[sample_idx, self.extended_goal_joint_names.index(j)] = 1.  # 把对应目标的有效位设为 1

        mlp_input = torch.cat([input, validity], dim=-1).view(input.shape[0], -1)  # 拼接 xyz 和 validity 后展平
        return self.mlp(mlp_input)  # 输出条件嵌入


class EmbedTargetLocSplit(nn.Module):  # split 版本目标位置编码器
    def __init__(self, all_goal_joint_names, latent_dim, num_layers=1):  # 初始化 split 编码器
        super().__init__()  # 调用父类初始化
        self.extended_goal_joint_names = all_goal_joint_names + ['traj', 'heading']  # 扩展目标名称列表
        self.target_cond_dim = 4  # 单个目标的输入维度为 xyz + validity
        self.latent_dim = latent_dim  # 保存总隐藏维度
        self.splited_dim = self.latent_dim // len(self.extended_goal_joint_names)  # 为每个目标均分一个子维度
        assert self.latent_dim % len(self.extended_goal_joint_names) == 0  # 必须能整除
        self.mini_mlps = nn.ModuleList()  # 创建目标级别的小 MLP 列表
        for _ in self.extended_goal_joint_names:  # 为每个目标各建一个小网络
            _layers = [nn.Linear(self.target_cond_dim, self.splited_dim)]  # 先映射到分配给该目标的子维度
            for _ in range(num_layers):  # 根据配置继续堆叠层
                _layers += [nn.SiLU(), nn.Linear(self.splited_dim, self.splited_dim)]  # 每层是 SiLU 和线性层
            self.mini_mlps.append(nn.Sequential(*_layers))  # 把这个小网络加入列表

    def forward(self, input, target_joint_names, target_heading):  # split 编码器前向传播
        # TODO - generate validity from outside the model  # TODO：validity 更适合在外部准备
        validity = torch.zeros_like(input)[..., :1]  # 初始化 validity 标记
        for sample_idx, sample_joint_names in enumerate(target_joint_names):  # 遍历 batch 中每个样本
            sample_joint_names_w_heading = np.append(sample_joint_names, 'heading') if target_heading[sample_idx] else sample_joint_names  # 如有需要补上 heading
            for j in sample_joint_names_w_heading:  # 遍历有效目标
                validity[sample_idx, self.extended_goal_joint_names.index(j)] = 1.  # 标记该目标有效

        mlp_input = torch.cat([input, validity], dim=-1)  # 对每个目标拼上 validity
        mlp_splits = [self.mini_mlps[i](mlp_input[:, i]) for i in range(mlp_input.shape[1])]  # 每个目标单独通过自己的小 MLP
        return torch.cat(mlp_splits, dim=-1)  # 把各目标特征拼接成总向量
  
class EmbedTargetLocMulti(nn.Module):  # multi 版本目标位置编码器
    def __init__(self, all_goal_joint_names, latent_dim):  # 初始化 multi 编码器
        super().__init__()  # 调用父类初始化
        
        # todo: use a tensor of weight per joint, and another one for biases, then apply a selection in one go like we to for actions  # TODO：未来可进一步向量化
        self.extended_goal_joint_names = all_goal_joint_names + ['traj', 'heading']  # 扩展目标名称列表
        self.extended_goal_joint_idx = {joint_name: idx for idx, joint_name in enumerate(self.extended_goal_joint_names)}  # 构建名称到索引的映射
        self.n_extended_goal_joints = len(self.extended_goal_joint_names)  # 记录扩展目标数量
        self.target_loc_emb = nn.ParameterDict({joint_name:  # 为每个目标各建一套编码层
            nn.Sequential(  # 使用一个两层 MLP 编码位置
                nn.Linear(3, latent_dim),  # 把 xyz 映射到 latent
                nn.SiLU(),  # 激活函数
                nn.Linear(latent_dim, latent_dim))  # 再映射一次得到最终特征
            for joint_name in self.extended_goal_joint_names})  # todo: check if 3 works for heading and traj  # 对所有扩展目标统一创建
            # nn.Linear(3, latent_dim) for joint_name in self.extended_goal_joint_names})  # todo: check if 3 works for heading and traj  # 更简单的旧写法
        self.target_all_loc_emb = WeightedSum(self.n_extended_goal_joints) # nn.Linear(self.n_extended_goal_joints, latent_dim)  # 聚合所有目标特征
        self.latent_dim = latent_dim  # 保存隐藏维度

    def forward(self, input, target_joint_names, target_heading):  # multi 编码器前向传播
        output = torch.zeros((input.shape[0], self.latent_dim), dtype=input.dtype, device=input.device)  # 初始化 batch 输出
        
        # Iterate over the batch and apply the appropriate filter for each joint  # 按样本处理有效目标
        for sample_idx, sample_joint_names in enumerate(target_joint_names):  # 遍历 batch 中每个样本
            sample_joint_names_w_heading = np.append(sample_joint_names, 'heading') if target_heading[sample_idx] else sample_joint_names  # 如果需要则补上 heading
            output_one_sample = torch.zeros((self.n_extended_goal_joints, self.latent_dim), dtype=input.dtype, device=input.device)  # 初始化当前样本的逐目标特征表
            for joint_name in sample_joint_names_w_heading:  # 遍历当前样本有效目标
                layer = self.target_loc_emb[joint_name]  # 取出对应目标的编码层
                output_one_sample[self.extended_goal_joint_idx[joint_name]] = layer(input[sample_idx, self.extended_goal_joint_idx[joint_name]])  # 编码该目标位置并写回对应槽位
            output[sample_idx] = self.target_all_loc_emb(output_one_sample)  # 聚合当前样本的所有目标特征
            # print(torch.where(output_one_sample.sum(axis=1)!=0)[0].cpu().numpy())  # 调试查看哪些目标被激活
               
        return output  # 返回 batch 级目标条件嵌入
