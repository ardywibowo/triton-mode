import torch
import triton
import triton.language as tl

MAX_FUSED_SIZE = 65536 // 2

@triton.jit
def triton_cross_entropy_forward(
    input_ptr,
    input_stride,
    target_ptr,
    target_stride,
    loss_output_ptr,
    loss_output_stride,
    num_classes,
    num_valid_targets,
    ignore_label,
    smoothing_factor: tl.constexpr,
    reduction_mode: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0).to(tl.int64)

    target_ptr += row_id * target_stride
    target_label = tl.load(target_ptr)

    input_ptr += row_id * input_stride
    if target_label == ignore_label:
        for i in range(0, num_classes, BLOCK_SIZE):
            input_offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(input_ptr + input_offsets, 0.0, mask=input_offsets < num_classes)
        return

    loss_output_ptr += row_id * loss_output_stride

    max_val = float("-inf")
    normalization_factor = 0.0
    target_input_val = tl.load(input_ptr + target_label)

    smoothing_sum = 0.0
    epsilon = smoothing_factor / num_classes

    for i in range(0, num_classes, BLOCK_SIZE):
        input_offsets = i + tl.arange(0, BLOCK_SIZE)
        input_block = tl.load(input_ptr + input_offsets, mask=input_offsets < num_classes, other=float("-inf"))
        block_max = tl.max(input_block)
        if smoothing_factor > 0:
            smoothing_sum += tl.sum(tl.where(input_offsets < num_classes, -epsilon * input_block, 0.0))
        new_max = tl.maximum(max_val, block_max)
        normalization_factor = normalization_factor * tl.exp(max_val - new_max) + tl.sum(tl.exp(input_block - new_max))
        max_val = new_max

    for i in range(0, num_classes, BLOCK_SIZE):
        input_offsets = i + tl.arange(0, BLOCK_SIZE)
        input_block = tl.load(input_ptr + input_offsets, mask=input_offsets < num_classes, other=float("-inf"))
        if reduction_mode == "mean":
            input_block = (tl.exp(input_block - max_val) / normalization_factor - epsilon) / num_valid_targets
        else:
            input_block = tl.exp(input_block - max_val) / normalization_factor - epsilon

        tl.store(input_ptr + input_offsets, input_block, mask=input_offsets < num_classes)

    tl.debug_barrier()

    row_loss = -(target_input_val - max_val - tl.log(normalization_factor))

    if smoothing_factor > 0:
        smooth_loss = smoothing_sum + smoothing_factor * (max_val + tl.log(normalization_factor))
        row_loss = row_loss * (1 - smoothing_factor) + smooth_loss

    if reduction_mode == "mean":
        row_loss /= num_valid_targets

    updated_target_val = tl.load(input_ptr + target_label)
    if reduction_mode == "mean":
        updated_target_val += -(1 - smoothing_factor) / num_valid_targets
    else:
        updated_target_val += -(1 - smoothing_factor)

    tl.store(loss_output_ptr, row_loss)
    tl.store(input_ptr + target_label, updated_target_val)


@triton.jit
def triton_cross_entropy_backward(
    input_grad_ptr,
    input_stride,
    grad_output_ptr,
    num_classes,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0).to(tl.int64)

    input_grad_ptr += row_id * input_stride

    grad_output = tl.load(grad_output_ptr)

    for i in range(0, num_classes, BLOCK_SIZE):
        input_offsets = i + tl.arange(0, BLOCK_SIZE)
        input_grad_block = tl.load(input_grad_ptr + input_offsets, mask=input_offsets < num_classes)
        tl.store(input_grad_ptr + input_offsets, input_grad_block * grad_output, mask=input_offsets < num_classes)


def cross_entropy_forward(input_tensor, target_tensor, ignore_label, smoothing_factor, reduction_mode):
    batch_size, num_classes = input_tensor.shape

    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(num_classes))

    loss_output = torch.zeros(batch_size, dtype=input_tensor.dtype, device=input_tensor.device)
    num_valid_targets = (target_tensor != ignore_label).sum().item()
    if input_tensor.stride(-1) != 1:
        input_tensor = input_tensor.contiguous()
    if target_tensor.stride(-1) != 1:
        target_tensor = target_tensor.contiguous()

    triton_cross_entropy_forward[(batch_size,)](
        input_ptr=input_tensor,
        input_stride=input_tensor.stride(-2),
        target_ptr=target_tensor,
        target_stride=target_tensor.stride(-1),
        loss_output_ptr=loss_output,
        loss_output_stride=loss_output.stride(-1),
        num_classes=num_classes,
        num_valid_targets=num_valid_targets,
        ignore_label=ignore_label,
        smoothing_factor=smoothing_factor,
        reduction_mode=reduction_mode,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32
    )

    total_loss = torch.sum(loss_output)
    return total_loss, input_tensor


def cross_entropy_backward(input_tensor, grad_output_tensor):
    if torch.equal(grad_output_tensor, torch.tensor(1.0, device=grad_output_tensor.device)):
        return input_tensor
    else:
        batch_size, num_classes = input_tensor.shape
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(num_classes))

        triton_cross_entropy_backward[(batch_size,)](
            input_grad_ptr=input_tensor,
            input_stride=input_tensor.stride(-2),
            grad_output_ptr=grad_output_tensor,
            num_classes=num_classes,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32,
        )

    return input_tensor


class CrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, target_tensor, ignore_label=-100, smoothing_factor=0.0, reduction_mode="mean"):
        loss, input_tensor = cross_entropy_forward(input_tensor, target_tensor, ignore_label, smoothing_factor, reduction_mode)
        ctx.save_for_backward(input_tensor.detach())
        return loss

    @staticmethod
    def backward(ctx, grad_output_tensor):
        (input_tensor,) = ctx.saved_tensors
        input_tensor = cross_entropy_backward(input_tensor, grad_output_tensor)
        return input_tensor, None, None, None, None
