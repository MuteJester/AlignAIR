from alignair.gym.curriculum import Curriculum


def test_curriculum_ramps_and_describes():
    c = Curriculum()
    easy = c.params(0.0)
    hard = c.params(1.0)
    # harder at p=1: higher mutation rate cap, more trim, indels, seq errors
    assert hard["mutation_rate"] >= easy["mutation_rate"]
    assert hard["end_loss_5"][1] >= easy["end_loss_5"][1]
    assert hard["indel_count"][1] >= easy["indel_count"][1]
    assert hard["seq_error_rate"] >= easy["seq_error_rate"]
    d = c.describe(0.5)
    assert isinstance(d, str) and "stage" in d.lower()


def test_curriculum_stage_index():
    c = Curriculum(stages=5)
    assert c.stage(0.0) == 0
    assert c.stage(1.0) == 4
