from functools import wraps
import torch

def force_fp32(apply_to=None, out_fp32=True):
    """Decorator to force outputs to be FP32 in mixed precision training.

    Args:
        apply_to (sequence[str], optional): A list of argument names to which
            the decorator should be applied. Default: None (apply to all).
        out_fp32 (bool): Whether to convert the function output(s) to FP32.
            Default: True.
    """

    def decorator(func):
        @wraps(func)
        def new_func(*args, **kwargs):
            # Check if the model is in mixed precision mode
            if not hasattr(args[0], 'with_fp16') or not args[0].with_fp16:
                # If not using mixed precision, call the function normally
                return func(*args, **kwargs)

            # Cast inputs to FP32 where necessary
            if apply_to is None:
                # Apply FP32 conversion to all inputs
                new_args = [arg.float() if isinstance(arg, torch.Tensor) else arg for arg in args]
            else:
                # Apply FP32 conversion only to specified arguments
                new_args = []
                for i, arg in enumerate(args):
                    if i in apply_to:
                        new_args.append(arg.float() if isinstance(arg, torch.Tensor) else arg)
                    else:
                        new_args.append(arg)

            # Call the original function
            output = func(*new_args, **kwargs)

            # Cast output back to FP32 if necessary
            if out_fp32 and isinstance(output, torch.Tensor):
                return output.float()
            elif isinstance(output, tuple):
                # Handle tuple outputs (e.g., when multiple outputs are returned)
                return tuple(o.float() if isinstance(o, torch.Tensor) else o for o in output)
            else:
                return output

        return new_func

    return decorator
