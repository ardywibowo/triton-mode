from typing import Literal
import torch
import triton
import triton.language as tl

from kernels.utils import ensure_contiguous


def compute_num_warps(BLOCK_SIZE):
    if BLOCK_SIZE >= 32768:
        return 32
    elif BLOCK_SIZE >= 8192:
        return 16
    elif BLOCK_SIZE >= 2048:
        return 8
    else:
        return 4

MAX_FUSED_SIZE = 65536 // 4
REDUCTION_MODE_LITERAL = Literal["none", "sum", "mean", "batchmean"]

# Triton reduction modes mapped to constants
REDUCE_NONE = tl.constexpr(0)
REDUCE_SUM = tl.constexpr(1)
REDUCE_MEAN = tl.constexpr(2)
REDUCE_BATCH_MEAN = tl.constexpr(3)

# Map string values of reduction mode to internal constants
_str_to_reduction_mode = {
    "none": REDUCE_NONE.value,
    "sum": REDUCE_SUM.value,
    "mean": REDUCE_MEAN.value,
    "batchmean": REDUCE_BATCH_MEAN.value,
}


@triton.jit
def triton_kl_forward(
    y_pred_ptr,
    y_pred_stride,
    y_true_ptr,
    y_true_stride,
    output_loss_ptr,
    output_loss_stride,
    num_classes,
    epsilon,
    BLOCK_SIZE: tl.constexpr,
    log_target: tl.constexpr = False,
    reduction_mode: tl.constexpr = REDUCE_BATCH_MEAN,
):
    row_id = tl.program_id(0).to(tl.int64)
    
    y_pred_ptr += row_id * y_pred_stride
    y_true_ptr += row_id * y_true_stride
    output_loss_ptr += row_id * output_loss_stride

    base_offsets = tl.arange(0, BLOCK_SIZE)

    loss_sum = 0.0
    for i in range(0, num_classes, BLOCK_SIZE):
        offsets = i + base_offsets
        mask = offsets < num_classes

        y_pred = tl.load(y_pred_ptr + offsets, mask=mask, other=0.0)
        y_true = tl.load(y_true_ptr + offsets, mask=mask, other=0.0)

        # Compute KL Divergence (y_true || y_pred)
        if not log_target:
            loss = y_true * (tl.log(tl.maximum(y_true, epsilon)) - y_pred)
        else:
            loss = tl.exp(y_true) * (y_true - y_pred)

        if reduction_mode == REDUCE_NONE:
            tl.store(output_loss_ptr + offsets, loss, mask=mask)
        else:
            loss_sum += tl.sum(loss, axis=0)

    if reduction_mode != REDUCE_NONE:
        tl.store(output_loss_ptr, loss_sum)


@triton.jit
def triton_kl_backward(
    target_ptr,
    target_stride,
    grad_output_ptr,
    grad_output_stride,
    num_classes,
    BLOCK_SIZE: tl.constexpr,
    log_target: tl.constexpr = False,
):
    row_id = tl.program_id(0).to(tl.int64)

    target_ptr += row_id * target_stride
    grad_output_ptr += row_id * grad_output_stride

    base_offsets = tl.arange(0, BLOCK_SIZE)
    mask = base_offsets < num_classes

    for i in range(0, num_classes, BLOCK_SIZE):
        offsets = i + base_offsets
        mask = offsets < num_classes

        target_val = tl.load(target_ptr + offsets, mask=mask, other=0.0)

        if not log_target:
            grad = target_val * -1
        else:
            grad = -tl.exp(target_val)

        tl.store(grad_output_ptr + offsets, grad, mask=mask)


def kl_forward(y_pred, y_true, log_target, reduction_mode, epsilon):
    batch_size, num_classes = y_pred.shape

    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(num_classes))
    num_warps = compute_num_warps(BLOCK_SIZE)

    grid_size = (batch_size,)
    reduction_mode = _str_to_reduction_mode[reduction_mode]

    # Output tensor size depends on reduction mode
    output_size = (batch_size, num_classes) if reduction_mode == REDUCE_NONE.value else (batch_size,)
    output_loss = torch.zeros(output_size, device=y_pred.device, dtype=torch.float32)

    # Call the forward kernel
    triton_kl_forward[grid_size](
        y_pred_ptr=y_pred,
        y_pred_stride=y_pred.stride(0),
        y_true_ptr=y_true,
        y_true_stride=y_true.stride(0),
        output_loss_ptr=output_loss,
        output_loss_stride=output_loss.stride(0),
        num_classes=num_classes,
        epsilon=epsilon,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        log_target=log_target,
        reduction_mode=reduction_mode,
    )
    
    # Compute the final loss based on the reduction mode
    if reduction_mode == REDUCE_BATCH_MEAN.value:
        return output_loss.sum() / batch_size
    elif reduction_mode == REDUCE_SUM.value:
        return output_loss.sum(dim=0)
    elif reduction_mode == REDUCE_MEAN.value:
        return output_loss.sum() / (batch_size * num_classes)
    else:
        return output_loss


def kl_backward(target, grad_output, grad_result, log_target):
    batch_size, num_classes = target.shape

    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(num_classes))
    num_warps = compute_num_warps(BLOCK_SIZE)

    grid_size = (batch_size,)

    # Call the backward kernel
    triton_kl_backward[grid_size](
        target_ptr=target,
        target_stride=target.stride(0),
        grad_output_ptr=grad_result,
        grad_output_stride=grad_result.stride(0),
        num_classes=num_classes,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        log_target=log_target,
    )

    # Apply the gradient output if necessary
    if torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        return grad_result

    return grad_result * grad_output


class KLLoss(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, y_pred: torch.Tensor, y_true: torch.Tensor, reduction: REDUCTION_MODE_LITERAL = "batchmean", log_target: bool = False, eps: float = 1e-10) -> torch.Tensor:
        ctx.save_for_backward(y_true)
        ctx.reduction_mode = reduction
        ctx.log_target = log_target
        return kl_forward(y_pred, y_true, log_target=log_target, reduction_mode=reduction, epsilon=eps)

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (y_true,) = ctx.saved_tensors
        grad_result = torch.empty_like(y_true)

        derivative = kl_backward(y_true, grad_output, grad_result, ctx.log_target)

        # Apply final reduction adjustment in backward pass
        if ctx.reduction_mode == "batchmean":
            derivative = derivative / y_true.shape[0]
        elif ctx.reduction_mode == "mean":
            derivative = derivative / (y_true.shape[0] * y_true.shape[1])

        return derivative, None, None, None, None
