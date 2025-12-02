import torch
import triton
import torch_npu
import triton.language as tl
import torch.nn.functional as F

# 当前TP4 num_v_head(NUM_HEADS)数量为8
@triton.jit
def fused_gdn_gating_head8_kernel(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    seq_len,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off
    mask = head_off < NUM_HEADS
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a = tl.load(a + off, mask=mask)
    blk_b = tl.load(b + off, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)
    # If the model is loaded in fp16, without the .float() here, A might be -inf
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)
    # compute beta_output = sigmoid(b)
    blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
    tl.store(
        beta_output + off, blk_beta_output.to(beta_output.dtype.element_ty), mask=mask
    )

def torch_fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    softplus_nn = torch.nn.Softplus(beta = beta, threshold = threshold)
    g = -A_log.float().exp() * softplus_nn(a.float() + dt_bias)
    beta_output = b.sigmoid()
    return (g, beta_output)

def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused computation of g and beta for Gated Delta Net.
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    beta_output = b.sigmoid()
    TODO maybe use torch.compile to replace this triton kernel
    """
    batch, num_heads = a.shape
    seq_len = 1
    grid = (batch, seq_len, triton.cdiv(num_heads, 8))
    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=a.device)
    beta_output = torch.empty(1, batch, num_heads, dtype=b.dtype, device=b.device)
    fused_gdn_gating_head8_kernel[grid](
        g,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        seq_len,
        num_heads,
        beta,
        threshold,
        8,
        num_warps=1,
    )
    return g, beta_output

def test_op(num_tokens, num_v_heads = 8):
    A_log = torch.randn((num_v_heads), dtype = torch.float32)
    a = torch.randn((num_tokens, num_v_heads), dtype = torch.float16)
    b = torch.randn((num_tokens, num_v_heads), dtype = torch.float16)
    dt_bias = torch.ones((num_v_heads), dtype = torch.float32)
    golden_g, golden_beta = torch_fused_gdn_gating(A_log, a, b, dt_bias)
    npu_g, npu_beta = fused_gdn_gating(A_log.npu(), a.npu(), b.npu(), dt_bias.npu())
    print(f"npu_g:{npu_g}")
    print(f"npu_beta:{npu_beta}")
    assert torch.allclose(golden_g, npu_g.cpu(), atol = 0.001, rtol = 0.001)
    assert torch.allclose(golden_beta, npu_beta.cpu(), atol = 0.001, rtol = 0.001)
    print("test pass")

if __name__ == '__main__':
    # pass
    test_op(3)

