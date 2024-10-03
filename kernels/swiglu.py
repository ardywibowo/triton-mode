import torch
import triton
import triton.language as tl

from kernels.utils import ensure_contiguous


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)


@triton.jit
def triton_swiglu_forward(
    input_a_ptr, input_b_ptr, output_ptr, row_stride, num_columns: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    prog_id = tl.program_id(0).to(tl.int64)

    # Compute starting pointer for this program
    input_a_ptr += prog_id * row_stride
    input_b_ptr += prog_id * row_stride
    output_ptr += prog_id * row_stride

    column_offsets = tl.arange(0, BLOCK_SIZE)
    active_mask = column_offsets < num_columns

    # Apply SiLU to input_a and then multiply by input_b
    input_a_row = tl.load(input_a_ptr + column_offsets, mask=active_mask, other=0).to(tl.float32)
    input_b_row = tl.load(input_b_ptr + column_offsets, mask=active_mask, other=0)
    result_row = silu(input_a_row) * input_b_row
    tl.store(output_ptr + column_offsets, result_row, mask=active_mask)


@triton.jit
def triton_swiglu_backward(
    grad_output_ptr, input_a_ptr, input_b_ptr, row_stride, num_columns: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    prog_id = tl.program_id(0).to(tl.int64)

    # Compute starting pointer for this program
    grad_output_ptr += prog_id * row_stride
    input_a_ptr += prog_id * row_stride
    input_b_ptr += prog_id * row_stride

    column_offsets = tl.arange(0, BLOCK_SIZE)
    active_mask = column_offsets < num_columns

    grad_output_row = tl.load(grad_output_ptr + column_offsets, mask=active_mask, other=0)
    # Apply sigmoid to input_a, then recompute SiLU and gradient updates
    input_a_row = tl.load(input_a_ptr + column_offsets, mask=active_mask, other=0).to(tl.float32)
    input_b_row = tl.load(input_b_ptr + column_offsets, mask=active_mask, other=0)

    sigmoid_a = tl.sigmoid(input_a_row)
    silu_a = input_a_row * sigmoid_a
    grad_b_row = grad_output_row * silu_a
    grad_a_row = grad_output_row * (silu_a * (1 - sigmoid_a) + sigmoid_a) * input_b_row

    tl.store(input_a_ptr + column_offsets, grad_a_row, mask=active_mask)
    tl.store(input_b_ptr + column_offsets, grad_b_row, mask=active_mask)


def swiglu_forward(a, b):
    input_shape = a.shape

    num_columns = input_shape[-1]
    a = a.view(-1, num_columns)
    b = b.view(-1, num_columns)
    output = torch.empty_like(a)
    num_rows = a.shape[0]

    TILE_SIZE = tl.next_power_of_2(num_columns)
    NUM_WARPS = 32

    triton_swiglu_forward[(num_rows,)](
        a,
        b,
        output,
        output.stride(-2),
        num_columns=num_columns,
        TILE_SIZE=TILE_SIZE,
        num_warps=NUM_WARPS,
    )
    return a, b, output.view(*input_shape)


def swiglu_backward(a, b, grad_output):
    input_shape = grad_output.shape
    num_columns = input_shape[-1]
    grad_output = grad_output.view(-1, num_columns)
    num_rows = grad_output.shape[0]

    BLOCK_SIZE = tl.next_power_of_2(num_columns)
    NUM_WARPS = 32

    triton_swiglu_backward[(num_rows,)](
        grad_output,
        a,
        b,
        grad_output.stride(-2),
        num_columns=num_columns,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )
    return a.view(*input_shape), b.view(*input_shape)


class SwiGLU(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, a, b):
        a, b, output = swiglu_forward(a, b)
        ctx.save_for_backward(a, b)
        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        a, b = swiglu_backward(a, b, grad_output)
        return a, b
