import torch
import triton
import triton.language as tl


@triton.jit
def triton_rope(
    q_buffer,
    q_buffer_stride,
    k_buffer,
    k_buffer_stride,
    cos_values,
    cos_values_stride,
    sin_values,
    sin_values_stride,
    seq_length,
    batch_size: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_k_heads: tl.constexpr,
    head_dim: tl.constexpr,
    padded_num_q_heads: tl.constexpr,
    padded_num_k_heads: tl.constexpr,
    padded_head_dim: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    IS_BACKWARD: tl.constexpr = False,
):
    prog_id = tl.program_id(0)

    q_buffer = q_buffer + prog_id * q_buffer_stride
    k_buffer = k_buffer + prog_id * k_buffer_stride

    cos_index = prog_id % seq_length
    cos_values = cos_values + cos_index * cos_values_stride
    sin_values = sin_values + cos_index * sin_values_stride
    cos_indices = tl.arange(0, padded_head_dim // 2)
    cos_active_mask = cos_indices < head_dim // 2
    cos_vec = tl.load(cos_values + cos_indices, mask=cos_active_mask, other=0)
    sin_vec = tl.load(sin_values + cos_indices, mask=cos_active_mask, other=0)

    q_half_offsets = (
        tl.arange(0, padded_num_q_heads)[:, None] * head_dim + tl.arange(0, padded_head_dim // 2)[None, :]
    )
    k_half_offsets = (
        tl.arange(0, padded_num_k_heads)[:, None] * head_dim + tl.arange(0, padded_head_dim // 2)[None, :]
    )
    q_mask = (tl.arange(0, padded_num_q_heads)[:, None] < num_q_heads) & (
        tl.arange(0, padded_head_dim // 2)[None, :] < head_dim // 2
    )
    k_mask = (tl.arange(0, padded_num_k_heads)[:, None] < num_k_heads) & (
        tl.arange(0, padded_head_dim // 2)[None, :] < head_dim // 2
    )
    q_tile_part1 = tl.load(q_buffer + q_half_offsets, mask=q_mask, other=0).to(
        sin_vec.dtype
    )
    k_tile_part1 = tl.load(k_buffer + k_half_offsets, mask=k_mask, other=0).to(
        sin_vec.dtype
    )

    q_half2_offsets = q_half_offsets + (head_dim // 2)
    k_half2_offsets = k_half_offsets + (head_dim // 2)
    q_half2_mask = q_mask
    k_half2_mask = k_mask
    q_tile_part2 = tl.load(q_buffer + q_half2_offsets, mask=q_half2_mask, other=0).to(
        sin_vec.dtype
    )
    k_tile_part2 = tl.load(k_buffer + k_half2_offsets, mask=k_half2_mask, other=0).to(
        sin_vec.dtype
    )

    if not IS_BACKWARD:
        updated_q_part1 = q_tile_part1 * cos_vec - q_tile_part2 * sin_vec
        tl.store(q_buffer + q_half_offsets, updated_q_part1, mask=q_mask)
        updated_q_part2 = q_tile_part2 * cos_vec + q_tile_part1 * sin_vec
        tl.store(q_buffer + q_half2_offsets, updated_q_part2, mask=q_half2_mask)

        updated_k_part1 = k_tile_part1 * cos_vec - k_tile_part2 * sin_vec
        tl.store(k_buffer + k_half_offsets, updated_k_part1, mask=k_mask)
        updated_k_part2 = k_tile_part2 * cos_vec + k_tile_part1 * sin_vec
        tl.store(k_buffer + k_half2_offsets, updated_k_part2, mask=k_half2_mask)
    else:
        reversed_q_part1 = q_tile_part1 * cos_vec + q_tile_part2 * sin_vec
        tl.store(q_buffer + q_half_offsets, reversed_q_part1, mask=q_mask)
        reversed_q_part2 = q_tile_part2 * cos_vec - q_tile_part1 * sin_vec
        tl.store(q_buffer + q_half2_offsets, reversed_q_part2, mask=q_half2_mask)

        reversed_k_part1 = k_tile_part1 * cos_vec + k_tile_part2 * sin_vec
        tl.store(k_buffer + k_half_offsets, reversed_k_part1, mask=k_mask)
        reversed_k_part2 = k_tile_part2 * cos_vec - k_tile_part1 * sin_vec
        tl.store(k_buffer + k_half2_offsets, reversed_k_part2, mask=k_half2_mask)


def rope_forward(q, k, cos, sin):
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)

    batch_size, seq_len, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[2]
    padded_head_dim = triton.next_power_of_2(head_dim)
    padded_num_q_heads = triton.next_power_of_2(num_q_heads)
    padded_num_kv_heads = triton.next_power_of_2(num_kv_heads)
    TILE_SIZE = max(padded_num_q_heads, padded_num_kv_heads)

    row_count = batch_size * seq_len

    q = q.contiguous()
    k = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    triton_rope[(row_count,)](
        q,
        q.stride(1),
        k,
        k.stride(1),
        cos,
        cos.stride(-2),
        sin,
        sin.stride(-2),
        seq_len,
        batch_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
        padded_num_q_heads,
        padded_num_kv_heads,
        padded_head_dim,
        TILE_SIZE=TILE_SIZE,
        IS_BACKWARD=False,
    )
    return q.transpose(1, 2), k.transpose(1, 2), cos, sin


def rope_backward(dq, dk, cos, sin):
    dq = dq.transpose(1, 2)
    dk = dk.transpose(1, 2)

    batch_size, seq_len, num_q_heads, head_dim = dq.shape
    num_kv_heads = dk.shape[2]
    padded_head_dim = triton.next_power_of_2(head_dim)
    padded_num_q_heads = triton.next_power_of_2(num_q_heads)
    padded_num_kv_heads = triton.next_power_of_2(num_kv_heads)
    TILE_SIZE = max(padded_num_q_heads, padded_num_kv_heads)

    row_count = batch_size * seq_len

    dq = dq.contiguous()
    dk = dk.contiguous()

    triton_rope[(row_count,)](
        dq,
        dq.stride(1),
        dk,
        dk.stride(1),
        cos,
        cos.stride(-2),
        sin,
        sin.stride(-2),
        seq_len,
        batch_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
        padded_num_q_heads,
        padded_num_kv_heads,
        padded_head_dim,
        TILE_SIZE=TILE_SIZE,
        IS_BACKWARD=True,
    )
    return dq.transpose(1, 2), dk.transpose(1, 2)


class RoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin, pos_ids=None, unsqueeze_dim=1):
        q, k, cos, sin = rope_forward(q, k, cos, sin)
        ctx.save_for_backward(cos, sin)
        return q, k

    @staticmethod
    def backward(ctx, dq, dk):
        cos, sin = ctx.saved_tensors
        dq, dk = rope_backward(dq, dk, cos, sin)
        return dq, dk, None, None, None, None
