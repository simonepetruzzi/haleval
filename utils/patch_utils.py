import torch
from torch.utils.hooks import RemovableHandle
from functools import partial
import torch.nn as nn
import traceback


class InspectOutput:
    """
    Takes a model, a list of module names and registers hooks to inspect the output 
    of those modules during the forward pass.
    """

    def __init__(self, model, module_names, move_to_cpu=False, last_position=False):
        self.model = model
        self.module_names = module_names
        self.move_to_cpu = move_to_cpu
        self.last_position = last_position
        self.handles = []
        self.catcher = dict()

    def __enter__(self):
        for module_name, module in self.model.named_modules():
            if module_name in self.module_names:
                handle = inspect_output(module, self.catcher, module_name, move_to_cpu=self.move_to_cpu,
                                        last_position=self.last_position)
                self.handles.append(handle)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self.handles:
            handle.remove()

        if exc_type is not None:
            print("An exception occurred:")
            print(f"Type: {exc_type}")
            print(f"Value: {exc_val}")
            print("Traceback:")
            traceback.print_tb(exc_tb)
            return False
        return True
    

def inspect_hook(module: nn.Module, inputs, outputs, catcher: dict, module_name, move_to_cpu, last_position=False):
    '''
    This function is invoked when triggered by a hook in the forward pass.
    You can capture the last position (final token in a sequence) or the entire output. 
    '''
    if last_position:
        if type(outputs) is tuple:
            catcher[module_name] = outputs[0][:, -1]  # .clone()
        else:
            catcher[module_name] = outputs[:, -1]
        if move_to_cpu:
            catcher[module_name] = catcher[module_name].cpu()
    else:
        if type(outputs) is tuple:
            catcher[module_name] = outputs[0]  # .clone()
        else:
            catcher[module_name] = outputs
        if move_to_cpu:
            catcher[module_name] = catcher[module_name].cpu()
    return outputs


def inspect_output(module: nn.Module, catcher: dict, module_name, move_to_cpu, last_position=False) -> RemovableHandle:
    '''
    Register a forward hook to catch the output of a module during forward pass.
    '''
    hook_instance = partial(inspect_hook, catcher=catcher, module_name=module_name, move_to_cpu=move_to_cpu,
                            last_position=last_position)
    handle = module.register_forward_hook(hook_instance)
    return handle