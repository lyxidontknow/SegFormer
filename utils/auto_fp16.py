from functools import wraps
import torch

def auto_fp16(apply_to=None):
    """Decorator for automatically converting inputs of a function to FP16.

    Args:
        apply_to (sequence, optional): A list of argument names to which
            FP16 conversion should be applied. If not specified, it applies
            to all positional arguments.
    """

    def auto_fp16_wrapper(func):
        @wraps(func)
        def new_func(*args, **kwargs):
            if not hasattr(args[0], 'with_fp16') or not args[0].with_fp16:
                # If the module does not have FP16 enabled, do nothing
                return func(*args, **kwargs)

            # Cast inputs to FP16
            new_args = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    new_args.append(arg.half())
                else:
                    new_args.append(arg)

            # Call the function with FP16 inputs
            return func(*new_args, **kwargs)

        return new_func

    return auto_fp16_wrapper
