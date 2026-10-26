"""Exercise public APIs from an installed distribution outside the source checkout.

Run with the wheel environment's Python and --forbid-source pointing at src/.
This helper deliberately has no imports from tests.* or pytest.
"""
import argparse
import csv
import importlib.util
from pathlib import Path
import tempfile
import sysconfig
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forbid-source", type=Path, required=True)
    args = parser.parse_args()

    import alignair
    import GenAIRR.data as gd
    import torch
    from alignair import Aligner, load_model
    from alignair.cli.main import build_parser
    from alignair.core import AlignAIR
    from alignair.core.config import AlignAIRConfig
    from alignair.errors import ModelFormatError, RegistryError
    from alignair import model_file

    origin = Path(alignair.__file__).resolve()
    assert not origin.is_relative_to(args.forbid_source.resolve()), origin
    installed_roots = {Path(sysconfig.get_path(name)).resolve() for name in ("purelib", "platlib")}
    assert any(origin.is_relative_to(root) for root in installed_roots), origin
    torch.set_num_threads(2)
    torch.manual_seed(0)
    cfg = AlignAIRConfig.from_dataconfigs(gd.HUMAN_IGH_OGRDB, max_seq_length=128)
    model = AlignAIR(cfg).eval()
    with tempfile.TemporaryDirectory(prefix="alignair-installed-") as directory:
        root = Path(directory)
        artifact = root / "model.alignair"
        model_file.save_model(str(artifact), model, dataconfigs=["HUMAN_IGH_OGRDB"],
                              training={"steps": 0}, include_trusted_pickle=False)
        header = model_file.read_metadata(artifact)
        assert "reference_json" in header["sections"]
        assert not any(name.startswith("dataconfig/") or name == "train_state"
                       for name in header["sections"])
        restored, reference = load_model(str(artifact), device="cpu")
        assert reference.gene("V").names
        result = Aligner.from_model(restored, reference, device="cpu").predict(["ACGT" * 20])
        assert len(result) == 1
        output = root / "predictions.tsv"
        result.write_airr(str(output))
        with output.open() as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        assert len(rows) == 1
        assert "sequence_id" in rows[0] and "v_call" in rows[0]
        broken = root / "broken.alignair"
        broken.write_bytes(b"not a model")
        try:
            model_file.read_metadata(broken)
        except ModelFormatError as exc:
            assert exc.code == "ALIGNAIR_MODEL_FORMAT"
        else:
            raise AssertionError("corrupt installed-model input was accepted")
        parsed = build_parser().parse_args(["doctor"])
        assert parsed is not None
        from huggingface_hub import constants
        class UnexpectedNetwork(BaseException):
            """Cannot be swallowed by an SDK's ordinary exception handling."""
        with (patch.object(constants, "HF_HUB_CACHE", str(root / "empty-hub")),
              patch("socket.getaddrinfo", side_effect=UnexpectedNetwork("offline DNS")),
              patch("socket.socket.connect", side_effect=UnexpectedNetwork("offline connection"))):
            try:
                Aligner.from_pretrained("hf://AlignAIR/qa-offline-miss", offline=True, device="cpu")
            except RegistryError as exc:
                assert exc.context["offline"] is True
                assert isinstance(exc.__cause__, FileNotFoundError)
            else:
                raise AssertionError("offline loading accepted a missing cache artifact")
        # The research harness is intentionally source-only. The runtime's
        # shipped self-evaluator must work without importing that harness.
        assert importlib.util.find_spec("alignair_benchmark") is None
        from alignair.evaluate import run_benchmark
        evaluation = run_benchmark(restored, reference, gd.HUMAN_IGH_OGRDB,
                                   n=2, seed=101, strata_names=["clean"], device="cpu", batch_size=2)
        assert evaluation["clean"]["n"] == 2
        assert 0 <= evaluation["clean"]["v_call_acc"] <= 1
        assert 0 <= evaluation["clean"]["productive_acc"] <= 1
    print(f"Installed distribution contracts passed: {origin}")


if __name__ == "__main__":
    main()
