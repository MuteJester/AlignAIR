"""The GenAIRR seeds of a training run's record streams: each stream draws from a reserved block of seeds."""
from __future__ import annotations

import hashlib
import json

# GenAIRR draws record k of the stream of seed s from seed s + k, so streams whose seeds lie close together repeat each
# other's draws a few records apart (and the same seed under other parameters repeats the V/D/J draw). Every record
# stream of a training run therefore draws from a reserved block of SEED_BLOCK_RECORDS consecutive seeds. The block is
# placed by a SHA-256 of what the stream is for (``seed_block``), in [2**62, 2**63): far above the seeds a user or a
# benchmark picks ([0, 2**31)) and below GenAIRR's limit (2**63 + 1 works, 2**64 - 1 overflows). Blocks are aligned, so
# two blocks either coincide or are disjoint. Each purpose has its own quarter of the range, so training, validation and
# probe blocks never meet. Within one call of ``training_records`` the streams are checked to draw distinct blocks.
# Across calls of one purpose, such as a resume at another start_step against the original run, disjointness is only
# probable: two calls of 28 streams share no block with probability about 1 - 28 * 28 / 2**28 (1 - 3e-6).
STREAM_PURPOSES = ('training', 'validation', 'probe')
SEED_BLOCK_RECORDS = 2 ** 32
_FIRST_BLOCK_SEED = 2 ** 62
_BLOCKS_PER_PURPOSE = 2 ** 28              # 4 quarters of 2**28 blocks of 2**32 seeds: [2**62, 2**63)


def seed_block(seed: int, purpose: str, start_step: int, panel: int, component: int, stream: str) -> int:
    """The first GenAIRR seed of the reserved block of one record stream: run ``seed``, stream ``purpose``
    (``STREAM_PURPOSES``), the step ``start_step`` a training stream starts at (a resumed run's stream), the
    stream's reference ``panel`` and curriculum ``component`` positions, and its name among the component's streams
    (``CurriculumStreamPolicy.substreams_for``)."""
    key = json.dumps(['alignair.training_records', seed, purpose, start_step, panel, component, stream]).encode()
    position = int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') % _BLOCKS_PER_PURPOSE
    return _FIRST_BLOCK_SEED + (STREAM_PURPOSES.index(purpose) * _BLOCKS_PER_PURPOSE + position) * SEED_BLOCK_RECORDS
