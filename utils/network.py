# Now we can use this function in a "with" grammer
@contextmanager
def register_embedding_hook(layer, output_arr):
    """
    Add a hook to a pytorch layer to capture output of that layer on forward pass
    """
    handle = layer.register_forward_hook(
        lambda layer, input, output: output_arr.append(output)
    )
    yield
    handle.remove()