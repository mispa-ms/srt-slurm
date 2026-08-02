def pp_aux_split(
    aux_layers: tuple[int, ...],
    start_layer: int,
    end_layer: int,
    is_last_rank: bool,
) -> tuple[int, tuple[int, ...]]:
    """Split EAGLE-3 aux hidden state layers across one pipeline-parallel rank.

    Aux index ``L`` denotes the input to layer ``L``, i.e. the output of layer
    ``L - 1``. A rank owning layers ``[start_layer, end_layer)`` can therefore
    produce aux for ``L in [start_layer, end_layer]``, which overlaps the next
    rank at ``end_layer``. Non-last ranks drop that boundary entry; the next
    rank reproduces the identical tensor from the hidden state it receives.

    Args:
        aux_layers: Aux hidden state layer indices for the whole model.
        start_layer: First layer index owned by this rank, inclusive.
        end_layer: Last layer index owned by this rank, exclusive.
        is_last_rank: Whether this rank is the last pipeline stage.

    Returns:
        ``(recv_count, local_layers)``. ``recv_count`` is how many aux tensors
        arrive from earlier stages; ``local_layers`` are the aux indices this
        rank collects itself, in ascending order.
    """
    ordered = tuple(sorted(aux_layers))
    recv_count = sum(1 for layer in ordered if layer < start_layer)
    if is_last_rank:
        local = tuple(x for x in ordered if start_layer <= x <= end_layer)
    else:
        local = tuple(x for x in ordered if start_layer <= x < end_layer)
    return recv_count, local
