* 手动关闭 GQA，保证 q 和 kv 的头数一致
    - 如果想让一组 Q 头共享同一份 topk（类似 GQA 里多 Q 头共享一组 KV），可以把 index_n_heads 配成 num_key_value_heads，然后在注意力里按 n_rep = num_attention_heads // num_key_value_heads 将 topk 结果重复到对应的 Q 头组。
* 手动关闭 flash，保证走 dsa 分支

对于prefix阶段来说，给到indexer的隐藏变量x维度为[bsz,seqlen,embed_size]，indexer应该得到[bsz,seqlen,topk]维度，其中每一个topk都对应一个query token需要点积的kv序列中topk token

对于 Q，不需要用 topk 筛选，需要筛选的是与 Q 点积的 K，从 pastlen+1 中选出 topk 的 K 再与 Q 做点积

1. indexer 选 topk

deepseek 开源代码中的 indexer 是 MQA 的形式，多头的 q 对应单头的 k

* prefix 阶段 curlen = n，decode 阶段 curlen = 1
* `x [bsz, curlen, embed_size]` -> wq_i -> `qi [bsz, curlen, indexer_heads*head_dim]` -> view and transpose -> `qi [bsz, indexer_heads, curlen, head_dim]`
* `x [bsz, curlen, embed_size]` -> wk_i -> `ki [bsz, curlen, head_dim]` -> unsqueeze -> `ki [bsz, curlen, 1, head_dim]`

* 点积前的一系列操作，rope，量化等
* 点积 qi @ ki^T -> `scores [bsz, indexer_heads, curlen, curlen]`

2. 输入隐藏状态 x 线性变换
* prefix 阶段 curlen = n，decode 阶段 curlen = 1
* x [bsz, curlen, embed_size]-> wq -> q [bsz, curlen, n_q_heads*head_dim]
* x [bsz, curlen, embed_size]-> wk -> k [bsz, curlen, n_kv_heads*head_dim]
* x [bsz, curlen, embed_size]-> wv -> v [bsz, curlen, n_kv_heads*head_dim]

2. 过滤非法/未来/pad
    - 线性变换后

        prefix 阶段 curlen 为 prompt 长度，生成阶段 curlen 为 1

        xq：[bsz, curlen, n_q_heads, head_dim]

        xk/xv：[bsz, curlen, n_kv_heads, head_dim]

    - kv cache 后

        xq：[bsz, curlen, n_q_heads, head_dim]

        xk/xv：[bsz, pastlen+curlen, n_kv_heads, head_dim]

    - GQA 后
        xq：[bsz, n_q_heads, curlen, head_dim]

        xk/xv：[bsz, n_q_heads, pastlen+curlen, head_dim]

        每 n_q_heads / n_kv_heads 个 q 共享 kv，做法是广播 kv 至 q 的数量

3. gather K/V（可加当前 token）
4. 手写 attention”，只在 topk 子集上计算