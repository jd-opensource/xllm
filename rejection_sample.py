import random
import pytest

import torch
import torch_mlu_ops as tmo
from common_utils import *

def run_gen_case(dic):
    dump_data = dic.pop('dump_data')
    if dump_data:
        launch(*dic.values())
    else:
        draft_token_ids = dic['draft_token_ids']['data']
        num_draft_tokens = dic['num_draft_tokens']['data']
        cu_num_draft_tokens = dic['cu_num_draft_tokens']['data']
        draft_probs = create_tensor_from_dic(dic['draft_probs'])
        target_probs = create_tensor_from_dic(dic['target_probs'])
        bonus_token_ids = dic['bonus_token_ids']['data']
        uniform_rand = create_tensor_from_dic(dic['uniform_rand'])
        uniform_probs = create_tensor_from_dic(dic['uniform_probs'])
        max_spec_len = dic["max_spec_len"]["data"]
        launch(draft_token_ids, num_draft_tokens, cu_num_draft_tokens, draft_probs, target_probs, bonus_token_ids, uniform_rand, uniform_probs, max_spec_len)


@pytest.mark.skipif(check_env_flag('TMO_MEM_CHECK'), reason="Skipping test_prevent due to ASan issues.")
def test_prevent():
    args = gen_args(128, 129280, 4, torch.float, 1)
    draft_token_ids, num_draft_tokens, cu_num_draft_tokens, draft_probs, target_probs, bonus_token_ids, uniform_rand, uniform_probs, max_spec_len = args
    output_token_ids = torch.empty(num_draft_tokens.numel() + draft_token_ids.numel(), dtype=torch.int32, device=draft_token_ids.device)
    func = torch.ops.torch_mlu_ops.rejection_sample

    assertException("output_token_ids, draft_token_ids, num_draft_tokens, cu_num_draft_tokens, bonus_token_ids, draft_probs, target_probs, uniform_rand, uniform_probs must be contiguous.", func, output_token_ids.reshape(2, output_token_ids.shape[0] // 2).transpose(-1,-2), draft_token_ids,
                            num_draft_tokens, cu_num_draft_tokens, draft_probs, target_probs, bonus_token_ids, uniform_rand, uniform_probs, max_spec_len)
    assertException("output_token_ids, draft_token_ids, num_draft_tokens, cu_num_draft_tokens, bonus_token_ids, draft_probs, target_probs, uniform_rand, uniform_probs must be contiguous.", func, output_token_ids, draft_token_ids.reshape(2, draft_token_ids.shape[0] // 2).transpose(-1,-2),
                            num_draft_tokens, cu_num_draft_tokens, draft_probs, target_probs, bonus_token_ids, uniform_rand, uniform_probs, max_spec_len)
    assertException("output_token_ids, draft_token_ids, num_draft_tokens, cu_num_draft_tokens, bonus_token_ids, draft_probs, target_probs, uniform_rand, uniform_probs must be contiguous.", func, output_token_ids, draft_token_ids,
                            num_draft_tokens.reshape(2, num_draft_tokens.shape[0] // 2).transpose(-1,-2), cu_num_draft_tokens, draft_probs, target_probs, bonus_token_ids, uniform_rand, uniform_probs, max_spec_len)
    assertException("output_token_ids, draft_token_ids, num_draft_tokens, cu_num_draft_tokens, bonus_token_ids, draft_probs, target_probs, uniform_rand, uniform_probs must be contiguous.", func, output_token_ids, draft_token_ids,
                            num_draft_tokens, cu_num_draft_tokens.reshape(2, cu_num_draft_tokens.shape[0] // 2).transpose(-1,-2), draft_probs, target_probs, bonus_token_ids, uniform_rand, uniform_probs, max_spec_len)
    assertException("output_token_ids, draft_token_ids, num_draft_tokens, cu_num_draft_tokens, bonus_token_ids, draft_probs, target_probs, uniform_rand, uniform_probs must be contiguous.", func, output_token_ids, draft_token_ids,
                            num_draft_tokens, cu_num_draft_tokens, draft_probs.transpose(-1, -2), target_probs, bonus_token_ids, uniform_rand, uniform_probs, max_spec_len)
    assertException("output_token_ids, draft_token_ids, num_draft_tokens, cu_num_draft_tokens, bonus_token_ids, draft_probs, target_probs, uniform_rand, uniform_probs must be contiguous.", func, output_token_ids, draft_token_ids,
                            num_draft_tokens, cu_num_draft_tokens, draft_probs, target_probs.transpose(-1, -2), bonus_token_ids, uniform_rand, uniform_probs, max_spec_len)
    assertException("output_token_ids, draft_token_ids, num_draft_tokens, cu_num_draft_tokens, bonus_token_ids, draft_probs, target_probs, uniform_rand, uniform_probs must be contiguous.", func, output_token_ids, draft_token_ids,
                            num_draft_tokens, cu_num_draft_tokens, draft_probs, target_probs, bonus_token_ids.reshape(2, bonus_token_ids.shape[0] // 2).transpose(-1,-2), uniform_rand, uniform_probs, max_spec_len)
    assertException("output_token_ids, draft_token_ids, num_draft_tokens, cu_num_draft_tokens, bonus_token_ids, draft_probs, target_probs, uniform_rand, uniform_probs must be contiguous.", func, output_token_ids, draft_token_ids,
                            num_draft_tokens, cu_num_draft_tokens, draft_probs, target_probs, bonus_token_ids, uniform_rand.reshape(2, uniform_rand.shape[0] // 2).transpose(-1,-2), uniform_probs, max_spec_len)
    assertException("output_token_ids, draft_token_ids, num_draft_tokens, cu_num_draft_tokens, bonus_token_ids, draft_probs, target_probs, uniform_rand, uniform_probs must be contiguous.", func, output_token_ids, draft_token_ids,
                            num_draft_tokens, cu_num_draft_tokens, draft_probs, target_probs, bonus_token_ids, uniform_rand, uniform_probs.transpose(-1, -2), max_spec_len)

    assertException("dtype of output_token_ids must be int32.", func, output_token_ids.to(torch.int8), draft_token_ids,
                            num_draft_tokens, cu_num_draft_tokens, draft_probs, target_probs, bonus_token_ids, uniform_rand, uniform_probs, max_spec_len)
    assertException("Tensor dtype is not same. original dtype: Int, now dtype is: Char", func, output_token_ids, draft_token_ids.to(torch.int8),
                            num_draft_tokens, cu_num_draft_tokens, draft_probs, target_probs, bonus_token_ids, uniform_rand, uniform_probs, max_spec_len)
    assertException("Tensor dtype is not same. original dtype: Int, now dtype is: Char", func, output_token_ids, draft_token_ids,
                            num_draft_tokens.to(torch.int8), cu_num_draft_tokens, draft_probs, target_probs, bonus_token_ids, uniform_rand, uniform_probs, max_spec_len)
    assertException("Tensor dtype is not same. original dtype: Int, now dtype is: Char", func, output_token_ids, draft_token_ids,
                            num_draft_tokens, cu_num_draft_tokens.to(torch.int8), draft_probs, target_probs, bonus_token_ids, uniform_rand, uniform_probs, max_spec_len)
    assertException("Tensor dtype is not same. original dtype: Float, now dtype is: Char", func, output_token_ids, draft_token_ids,
                            num_draft_tokens, cu_num_draft_tokens, draft_probs.to(torch.int8), target_probs, bonus_token_ids, uniform_rand, uniform_probs, max_spec_len)
    assertException("dtype of target_probs must be float, half or bfloat16.", func, output_token_ids, draft_token_ids,
                            num_draft_tokens, cu_num_draft_tokens, draft_probs, target_probs.to(torch.int8), bonus_token_ids, uniform_rand, uniform_probs, max_spec_len)
    assertException("Tensor dtype is not same. original dtype: Int, now dtype is: Char", func, output_token_ids, draft_token_ids,
                            num_draft_tokens, cu_num_draft_tokens, draft_probs, target_probs, bonus_token_ids.to(torch.int8), uniform_rand, uniform_probs, max_spec_len)
    assertException("Tensor dtype is not same. original dtype: Float, now dtype is: Char", func, output_token_ids, draft_token_ids,
                            num_draft_tokens, cu_num_draft_tokens, draft_probs, target_probs, bonus_token_ids, uniform_rand.to(torch.int8), uniform_probs, max_spec_len)
    assertException("dtype of uniform_probs must be float.", func, output_token_ids, draft_token_ids,
                            num_draft_tokens, cu_num_draft_tokens, draft_probs, target_probs, bonus_token_ids, uniform_rand, uniform_probs.to(torch.int8), max_spec_len)

    assertException("dim of draft_probs, target_probs and uniform_probs must be the same and equal 2.", func, output_token_ids, draft_token_ids,
                            num_draft_tokens, cu_num_draft_tokens, draft_probs.unsqueeze(0), target_probs, bonus_token_ids, uniform_rand, uniform_probs, max_spec_len)
    assertException("dim of draft_probs, target_probs and uniform_probs must be the same and equal 2.", func, output_token_ids, draft_token_ids,
                            num_draft_tokens, cu_num_draft_tokens, draft_probs, target_probs.unsqueeze(0), bonus_token_ids, uniform_rand, uniform_probs, max_spec_len)
    assertException("dim of draft_probs, target_probs and uniform_probs must be the same and equal 2.", func, output_token_ids, draft_token_ids,
                            num_draft_tokens, cu_num_draft_tokens, draft_probs, target_probs, bonus_token_ids, uniform_rand, uniform_probs.unsqueeze(0), max_spec_len)
    assertException("shape of draft_probs, target_probs and uniform_probs must be the same.", func, output_token_ids, draft_token_ids,
                            num_draft_tokens, cu_num_draft_tokens, draft_probs, target_probs.reshape(target_probs.shape[0] // 2, target_probs.shape[1] * 2), bonus_token_ids, uniform_rand, uniform_probs, max_spec_len)
    assertException("shape of draft_probs, target_probs and uniform_probs must be the same.", func, output_token_ids, draft_token_ids,
                            num_draft_tokens, cu_num_draft_tokens, draft_probs, target_probs, bonus_token_ids, uniform_rand, uniform_probs.reshape(uniform_probs.shape[0] // 2, uniform_probs.shape[1] * 2), max_spec_len)
    assertException("elements number of draft_token_ids and uniform_rand must be the same and equal target_probs.size(0).", func, output_token_ids, num_draft_tokens,
                            num_draft_tokens, cu_num_draft_tokens, draft_probs, target_probs, bonus_token_ids, uniform_rand, uniform_probs, max_spec_len)
    assertException("elements number of draft_token_ids and uniform_rand must be the same and equal target_probs.size(0).", func, output_token_ids, draft_token_ids,
                            num_draft_tokens, cu_num_draft_tokens, draft_probs, target_probs, bonus_token_ids, uniform_probs, uniform_probs, max_spec_len)
    assertException("elements number of num_draft_tokens, cu_num_draft_tokens and bonus_token_ids must be the same.", func, output_token_ids, draft_token_ids,
                            draft_token_ids, cu_num_draft_tokens, draft_probs, target_probs, bonus_token_ids, uniform_rand, uniform_probs, max_spec_len)
    assertException("elements number of num_draft_tokens, cu_num_draft_tokens and bonus_token_ids must be the same.", func, output_token_ids, draft_token_ids,
                            num_draft_tokens, draft_token_ids, draft_probs, target_probs, bonus_token_ids, uniform_rand, uniform_probs, max_spec_len)
    assertException("elements number of num_draft_tokens, cu_num_draft_tokens and bonus_token_ids must be the same.", func, output_token_ids, draft_token_ids,
                            num_draft_tokens, cu_num_draft_tokens, draft_probs, target_probs, draft_token_ids, uniform_rand, uniform_probs, max_spec_len)
    assertException("elements number of output_token_ids should >= batch_size + num_tokens.", func, output_token_ids[:-1], draft_token_ids,
                            num_draft_tokens, cu_num_draft_tokens, draft_probs, target_probs, bonus_token_ids, uniform_rand, uniform_probs, max_spec_len)

def test_inductor():
    args = gen_args(128, 129280, 4, torch.float, 1)
    output_token_ids = torch.empty(args[0].numel() + args[1].numel(), dtype=torch.int32, device=args[0].device)
    args_all = (output_token_ids, *args)
    base_opcheck(torch.ops.torch_mlu_ops.rejection_sample, args_all)

@pytest.mark.parametrize("batch", [16, 128, 512])
@pytest.mark.parametrize("vocab", [129280])
@pytest.mark.parametrize("k", [1, 3, 4])
@pytest.mark.parametrize("dtype", [torch.half, torch.float, torch.bfloat16])
@pytest.mark.parametrize("draft_prob", [1, 0])
@pytest.mark.cnsanitizer
@pytest.mark.skipif(check_env_flag('TMO_MEM_CHECK'), reason="Skipping test_cnsanitizer_check due to ASan issues")
def test_cnsanitizer_check(batch, vocab, k, dtype, draft_prob):
    print("rejection_sample, cnsanitizer_check")
    args = gen_args(batch, vocab, k, dtype, draft_prob)
    output_token_ids = tmo.rejection_sample(*args)
    if check_env_flag('TMO_CHECK_CNSANITIZER_CASE_PRECISION'):
        output_token_ids_baseline = op_impl_base(*args).reshape(-1)
        assertTensorsEqual(output_token_ids.cpu(), output_token_ids_baseline.cpu(), 0, use_MSE=True, use_RAE=True, use_RMA=True)


# ======================================================================================================================================== #


def set_device():
    return "mlu:0"

def gen_args(batch_size,
             vocab_size,
             k,
             input_dtype,
             has_draft_probs):
    device = set_device()

    num_draft_tokens = torch.randint(low=1, high=k + 1, size=(batch_size,), dtype=torch.int32, device=device)
    cu_num_draft_tokens = torch.cumsum(num_draft_tokens.to(device=device, dtype=torch.int32), dim=0).to(device=device, dtype=torch.int32)
    num_tokens = cu_num_draft_tokens[batch_size - 1]
    max_spec_len = torch.max(num_draft_tokens)

    if has_draft_probs:
        draft_probs = torch.rand(num_tokens, vocab_size, dtype=input_dtype, device=device)
        draft_probs = draft_probs / draft_probs.sum(dim=-1, keepdim=True)
    else:
        draft_probs = None

    target_probs = torch.rand(num_tokens, vocab_size, dtype=input_dtype, device=device)
    # make target_probs similar to uniform_rand when draft_probs is None
    if has_draft_probs:
        target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True)
    bonus_token_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, 1),
                                   dtype=torch.int32, device=device)
    draft_token_ids = torch.randint(low=0, high=vocab_size, size=(num_tokens,),
                                  dtype=torch.int32, device=device)

    uniform_rand = torch.rand(draft_token_ids.shape, dtype=torch.float, device=device)
    uniform_probs = torch.empty_like(target_probs, device=device).to(dtype=torch.float)
    uniform_probs.exponential_(1.0)

    return [draft_token_ids, num_draft_tokens, cu_num_draft_tokens, draft_probs, target_probs, bonus_token_ids, uniform_rand, uniform_probs, max_spec_len]

def launch(*args):
    draft_token_ids, num_draft_tokens, cu_num_draft_tokens, draft_probs, target_probs, bonus_token_ids, uniform_rand, uniform_probs, max_spec_len = args
    output_token_ids = tmo.rejection_sample(draft_token_ids, num_draft_tokens, cu_num_draft_tokens, draft_probs, target_probs, bonus_token_ids,
                                                uniform_rand, uniform_probs, max_spec_len)
    output_token_ids_baseline = op_impl_base(*args).reshape(-1)

    assertTensorsEqual(output_token_ids.cpu(), output_token_ids_baseline.cpu(), 0, use_MSE=True, use_RAE=True, use_RMA=True)

def test_random():
    # filter the valid case
    set_seed(0)
    def filter(batch_size, vocab_size, k, dtype, has_draft_probs):
        buffer_size = 992 * 1024
        core_num = 48
        probs_dtype_size = 4
        ids_dtype_size = 4
        max_tokens_ipu = ((batch_size / core_num + (int)(batch_size % core_num > 0)) * core_num) / core_num * k
        # block_v should be 1 at least in step 2
        block_v = (buffer_size - (2 * batch_size + 1) * ids_dtype_size - max_tokens_ipu * (2 * probs_dtype_size + 2 * ids_dtype_size)) / 2 / ((2 + int(has_draft_probs)) * max_tokens_ipu * probs_dtype_size)

        # allocate 24G memory at most
        free_memory = 24 * 1024 * 1024 * 1024
        allocated_memory = batch_size * k * vocab_size * probs_dtype_size * 3

        return block_v >= 1 and allocated_memory < free_memory

    case_num = tmo_daily_unittest_case_size()
    batch_list = torch.randint(low=1, high=513, size=(case_num, ))
    vocab_list = torch.randint(low=1, high=151937, size=(case_num, ))
    k_list = torch.randint(low=1, high=129, size=(case_num, ))
    dtype_list = [torch.half, torch.bfloat16, torch.float]
    has_draft_probs_list = [1, 0]
    for i in range(case_num):
        dtype = random.choice(dtype_list)
        has_draft_probs = random.choice(has_draft_probs_list)

        print(f"case:{i}, batch:{batch_list[i]}, max_spec_len:{k_list[i]}, vocab:{vocab_list[i]}, probs_dtype:{dtype}, has_draft_probs:{has_draft_probs}")
        if filter(batch_list[i], vocab_list[i], k_list[i], dtype, has_draft_probs) == False:
            print("the case is invalid!")
            continue

        args = gen_args(batch_list[i], vocab_list[i], k_list[i], dtype, has_draft_probs)
        launch(*args)

def op_impl_base(*args):
    """
    拒绝采样算子的 PyTorch 基准实现（用于推测解码 Speculative Decoding）。

    根据目标模型与草稿模型的概率比，对草稿 token 做接受/拒绝决策；
    在首次拒绝位置从 (target - draft) 的残差分布中采样一个恢复 token；
    若全部接受则追加一个 bonus token。输出为扁平化的各样本结果拼接。
    """
    draft_token_ids_all, num_draft_tokens, cu_num_draft_tokens, draft_probs_all, target_probs_all, bonus_token_ids_all, uniform_rand_all, uniform_probs_all, max_spec_len = args

    num_tokens, vocab_size = target_probs_all.shape
    # 变长序列：按 num_draft_tokens 切分才能逐样本做拒绝采样
    split_list = num_draft_tokens.tolist()
    if draft_probs_all is not None:
        split_draft_probs = torch.split(draft_probs_all, split_size_or_sections=split_list, dim=0)
    split_target_probs = torch.split(target_probs_all, split_size_or_sections=split_list, dim=0)
    split_uniform_probs = torch.split(uniform_probs_all, split_size_or_sections=split_list, dim=0)
    split_draft_token_ids = torch.split(draft_token_ids_all, split_size_or_sections=split_list)
    split_uniform_rand = torch.split(uniform_rand_all, split_size_or_sections=split_list)
    split_bonus_token_ids = torch.unbind(bonus_token_ids_all, dim=0)
    output_all = torch.tensor([], dtype=torch.int32, device=target_probs_all.device)

    for i in range(len(num_draft_tokens)):
        if draft_probs_all is not None:
            draft_probs = split_draft_probs[i].unsqueeze(0).to(torch.float)
        target_probs = split_target_probs[i].unsqueeze(0).to(torch.float)
        uniform_probs = split_uniform_probs[i].unsqueeze(0)
        draft_token_ids = split_draft_token_ids[i].unsqueeze(0)
        uniform_rand = split_uniform_rand[i].unsqueeze(0)
        bonus_token_ids = split_bonus_token_ids[i].unsqueeze(0)

        batch_size, k, vocab_size = target_probs.shape

        # 拒绝采样需用「草稿 token 所在位置」的 target/draft 概率做接受判定
        batch_indices = torch.arange(batch_size, dtype=torch.int32, device=target_probs.device)[:, None]
        probs_indices = torch.arange(k, dtype=torch.int32, device=target_probs.device)
        if draft_probs_all is not None:
            selected_draft_probs = draft_probs[batch_indices, probs_indices, draft_token_ids]
        selected_target_probs = target_probs[batch_indices, probs_indices, draft_token_ids]
        # 接受概率 = min(target/draft, 1)；无 draft_probs 时视为 draft=1，即 min(target, 1)
        if draft_probs_all is None:
            capped_ratio = torch.minimum(selected_target_probs, torch.full((1, ), 1, device=target_probs.device))
        else:
            capped_ratio = torch.minimum(selected_target_probs / selected_draft_probs, torch.full((1, ), 1, device=target_probs.device))
        accepted = uniform_rand < capped_ratio

        # 残差分布：max(0, target - draft)，用于在拒绝位置采样恢复 token；无 draft 时把草稿位置置 0
        if draft_probs_all is None:
            difference = target_probs.clone()
            rows = torch.arange(difference.size(1))
            difference[:, rows, draft_token_ids.reshape(-1).tolist()] = 0
        else:
            difference = (target_probs - draft_probs).to(target_probs.device)
        f = torch.clamp(difference, min=0)
        # 按 (residual / uniform_probs) 做归一化后 argmax，得到恢复 token（Gumbel 式采样等价）
        recovered_token_ids = f.div_(uniform_probs).argmax(dim=2).to(torch.int32)

        # 首次拒绝位置 limit；全接受时设为 k，避免后续 mask 越界且用于 bonus 判定
        bonus_token_ids = bonus_token_ids.squeeze(-1)
        limits = (accepted == 0).max(1).indices
        limits[~(accepted == 0).any(1)] = k
        indices = torch.arange(k, device=accepted.device).unsqueeze(0)
        accepted_mask = indices < limits.unsqueeze(1)
        after_false_mask = indices == limits.unsqueeze(1)
        output_with_bonus_tokens = -torch.ones((batch_size, k + 1), dtype=torch.int32, device=accepted.device)
        output = output_with_bonus_tokens[:, :k]
        output[:, :k] = torch.where(accepted_mask, draft_token_ids, -torch.ones_like(draft_token_ids))
        all_accepted = (limits == k)
        output_with_bonus_tokens[all_accepted, -1] = bonus_token_ids[all_accepted]
        output.mul_(~after_false_mask).add_(recovered_token_ids.mul(after_false_mask))

        output_all = torch.cat((output_all.reshape(-1), output_with_bonus_tokens.reshape(-1)))

    return output_all

def test_rejection_sample():
    set_seed(1)
    batch_list = [16,32,64,128,256,512]
    vocab_list = [129280, 151936]
    k_list = [1,2,3,4]
    dtype_list = [torch.half, torch.bfloat16, torch.float]
    has_draft_probs = [1, 0]
    count = 0
    for i in range (len(batch_list)):
        for j in range (len(vocab_list)):
            for k in range (len(k_list)):
                for d in range (len(dtype_list)):
                    for h in range(len(has_draft_probs)):
                        print(f"case:{count}, batch:{batch_list[i]}, max_spec_len:{k_list[k]}, vocab:{vocab_list[j]}, probs_dtype:{dtype_list[d]}, has_draft_probs:{has_draft_probs[h]}")
                        args = gen_args(batch_list[i], vocab_list[j], k_list[k], dtype_list[d], has_draft_probs[h])
                        launch(*args)
                        count = count + 1
