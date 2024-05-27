"""Trace a model's forward pass to get intermediate activations."""

import contextlib
import inspect

import torch
import torch.nn as nn


class StopForward(Exception):
    """Raised when the forward pass should stop.

    If the only output needed from running a network is the retained submodule
    then the Trace(submodule, stop=True) will stop execution immediately after
    the retained submodule by raising the StopForward() exception. When Trace
    is used as a context manager, it catches that exception and can be used as
    follows:

    with Trace(net, layername, stop=True) as tr:
        net(inp)
    print(tr.output)
    """

    pass


def get_module(model: nn.Module, name: str):
    """Finds the named module within the given model.

    Args:
        model: The model to search.
        name: The name of the module to find.

    Returns:
        The named module.

    Raises:
        LookupError: If the named module is not found.
    """
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)


def recursive_copy(
    x: torch.Tensor | dict | list | tuple,
    clone: bool | None = None,
    detach: bool | None = None,
    retain_grad: bool | None = None,
):
    """Copies a reference to a tensor, or an object that contains tensors.

    This optionally detaches and clones the tensor(s).
    If retain_grad is true, the original tensors are marked to have grads retained.

    Args:
        x: The tensor or object to copy.
        clone: If True, clone the tensor.
        detach: If True, detach the tensor.
        retain_grad: If True, retain the tensor's gradients.
    """
    if not clone and not detach and not retain_grad:
        return x
    if isinstance(x, torch.Tensor):
        if retain_grad:
            if not x.requires_grad:
                x.requires_grad = True
            x.retain_grad()
        elif detach:
            x = x.detach()
        if clone:
            x = x.clone()
        return x
    # Only dicts, lists, and tuples (and subclasses) can be copied.
    if isinstance(x, dict):
        return type(x)(
            {
                k: recursive_copy(
                    v, clone=clone, detach=detach, retain_grad=retain_grad
                )
                for k, v in x.items()
            }
        )
    elif isinstance(x, list | tuple):
        return type(x)(
            [
                recursive_copy(v, clone=clone, detach=detach, retain_grad=retain_grad)
                for v in x
            ]
        )
    else:
        raise AssertionError(f"Unknown type {type(x)} cannot be broken into tensors.")


def invoke_with_optional_args(fn, *args, **kwargs):
    """Invoke a function with optional arguments.

    This function will pass arguments to the function in the following priority:
    1. Arguments that match by name.
    2. Remaining positional arguments.
    3. Extra keyword arguments that the function can accept.
    4. Extra positional arguments that the function can accept.

    Ordinary python calling conventions are helpful for supporting a function
    that might be revised to accept extra arguments in a newer version, without
    requiring the caller to pass those new arguments.  This function helps support
    function callers that might be revised to supply extra arguments, without
    requiring the callee to accept those new arguments.
    """
    argspec = inspect.getfullargspec(fn)
    pass_args = []
    used_kw = set()
    unmatched_pos = []
    used_pos = 0
    defaulted_pos = len(argspec.args) - (
        0 if not argspec.defaults else len(argspec.defaults)
    )
    # Pass positional args that match name first, then by position.
    for i, n in enumerate(argspec.args):
        if n in kwargs:
            pass_args.append(kwargs[n])
            used_kw.add(n)
        elif used_pos < len(args):
            pass_args.append(args[used_pos])
            used_pos += 1
        else:
            unmatched_pos.append(len(pass_args))
            pass_args.append(
                None if i < defaulted_pos else argspec.defaults[i - defaulted_pos]  # type: ignore
            )
    # Fill unmatched positional args with unmatched keyword args in order.
    if len(unmatched_pos):
        for k, v in kwargs.items():
            if k in used_kw or k in argspec.kwonlyargs:
                continue
            pass_args[unmatched_pos[0]] = v
            used_kw.add(k)
            unmatched_pos = unmatched_pos[1:]
            if len(unmatched_pos) == 0:
                break
        else:
            if unmatched_pos[0] < defaulted_pos:
                unpassed = ", ".join(
                    argspec.args[u] for u in unmatched_pos if u < defaulted_pos
                )
                raise TypeError(f"{fn.__name__}() cannot be passed {unpassed}.")
    # Pass remaining kw args if they can be accepted.
    pass_kw = {
        k: v
        for k, v in kwargs.items()
        if k not in used_kw and (k in argspec.kwonlyargs or argspec.varargs is not None)
    }
    # Pass remaining positional args if they can be accepted.
    if argspec.varargs is not None:
        pass_args += list(args[used_pos:])
    return fn(*pass_args, **pass_kw)


class Trace(contextlib.AbstractContextManager):
    """Trace the nn.Module forward pass to get intermediate activations."""

    def __init__(
        self,
        module: nn.Module,
        layer: str | None = None,
        stop=False,
        retain_input=True,
        retain_output=True,
        retain_grad=False,
        edit_output=None,
        clone=False,
        detach=False,
    ):
        """Replace the nn.Module's forward method with a closure.

        This closure intercepts and tracks the intermediate activations.

        Args:
            module: The nn.Module to trace.
            layer: The name of the layer to retain.
            stop: If True, stop the forward pass after the layer is retained.
            retain_input: If True, retain the input to the layer.
            retain_output: If True, retain the output of the layer.
            retain_grad: If True, retain the gradients of the output.
            edit_output: A function to edit the output by hijacking the forward pass.
            clone: If True, clone the input and output tensors.
            detach: If True, detach the input and output tensors.
        """
        retainer = self
        self.layer = layer

        if layer is not None:
            module = get_module(module, layer)

        def _hook(m, inputs, output):
            if edit_output:
                output = invoke_with_optional_args(
                    edit_output, output=output, layer=self.layer, inputs=inputs
                )
            if retain_input:
                retainer.input = recursive_copy(  # type: ignore
                    inputs[0] if len(inputs) == 1 else inputs,
                    clone=clone,
                    retain_grad=False,
                    detach=detach,
                )
            if retain_output:
                retainer.output = recursive_copy(  # type: ignore
                    output, clone=clone, retain_grad=retain_grad
                )

                # if retain_gran then also insert a copy operations so inplace operations
                # do not error
                if retain_grad:
                    output = recursive_copy(retainer.output, clone=True, detach=False)  # type: ignore
            if stop:
                raise StopForward()
            return output

        self.registered_hook = module.register_forward_hook(_hook)
        self.stop = stop

    def __enter__(self):
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context manager.

        If the forward pass was stopped and the exception was caught, return True.

        Args:
            exc_type: The exception type.
            exc_value: The exception value.
            traceback: The traceback.
        """
        self.close()
        if self.stop and exc_type is StopForward:
            return True

    def close(self):
        """Close the hook."""
        self.registered_hook.remove()
