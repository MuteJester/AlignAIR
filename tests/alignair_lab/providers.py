"""Record providers for the cache writer's tests; spawned writer processes import them by module name.

GenAIRR draws record k of the stream of seed s from seed s + k and names it ``seq<k>``, so record k of seed s + a is
record k + a of seed s under another name: a stream can be written in segments, each drawn from its own seed."""
from itertools import count


def shiftable(dataconfig, parameters, seed):
    """Records that depend on seed + k only, named as GenAIRR names them."""
    for k in count():
        yield {'sequence_id': f'seq{k}', 'value': seed + k}


def unshiftable(dataconfig, parameters, seed):
    """Records that depend on the stream's seed and k apart: a segment drawn from a later seed is another stream."""
    for k in count():
        yield {'sequence_id': f'seq{k}', 'value': [seed, k]}


def renamed(dataconfig, parameters, seed):
    """Shiftable records under names other than seq<k>, which a segment could not renumber."""
    for k in count():
        yield {'sequence_id': f'read-{k}', 'value': seed + k}
