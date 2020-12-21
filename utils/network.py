from contextlib import contextmanager

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


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)