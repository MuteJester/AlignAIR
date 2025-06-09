import subprocess
import os
import shutil
import pytest
import pandas as pd

# --- Paths and Config ---
TESTS_DIR = os.path.abspath(os.path.dirname(__file__))
DOCKER_IMAGE_TAGGED = "thomask90/alignair:latest"

HEAVY_INPUT_FILE = os.path.join(TESTS_DIR, "sample_HeavyChain_dataset.csv")
HEAVY_GOLDEN_FILE = os.path.join(TESTS_DIR, "heavychain_predict_validation.csv")

LIGHT_INPUT_FILE = os.path.join(TESTS_DIR, "sample_LightChain_dataset.csv")
LIGHT_GOLDEN_FILE = os.path.join(TESTS_DIR, "lightchain_predict_validation.csv")

OUTPUT_DIR = os.path.join(TESTS_DIR, "docker_output")
DOCKER_HEAVY_MODEL_PATH = "/app/pretrained_models/IGH_S5F_576"
DOCKER_LIGHT_MODEL_PATH = "/app/pretrained_models/IGL_S5F_576"
DOCKER_INPUT_HEAVY = "test_heavy.csv"
DOCKER_INPUT_LIGHT = "test_light.csv"
DOCKER_OUTPUT_PATH = "./"  # Save into the mounted /data directory (which is OUTPUT_DIR)

@pytest.fixture(autouse=True)
def clean_output():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def compare_csv(output_csv, golden_csv):
    assert os.path.isfile(output_csv), f"Expected output file not found: {output_csv}"
    assert os.path.isfile(golden_csv), f"Validation (golden) file not found: {golden_csv}"
    df = pd.read_csv(output_csv)
    validation = pd.read_csv(golden_csv)
    assert df.shape == validation.shape, f"Output shape {df.shape} != Validation shape {validation.shape}"
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            assert df.iloc[i, j] == validation.iloc[i, j], \
                f"Mismatch at row {i}, col {j}: {df.iloc[i, j]} != {validation.iloc[i, j]}"
    assert not df.empty, "Output CSV is empty"

def test_alignair_cli_predict_heavy():
    # Copy test input to mounted location
    input_mounted = os.path.join(OUTPUT_DIR, DOCKER_INPUT_HEAVY)
    shutil.copy(HEAVY_INPUT_FILE, input_mounted)

    docker_cmd = [
        "docker", "run", "--rm",
        "-v", f"{OUTPUT_DIR}:/tests",
        DOCKER_IMAGE_TAGGED,
        "python", "app.py", "run",
        f"--model-checkpoint={DOCKER_HEAVY_MODEL_PATH}",
        f"--save-path={DOCKER_OUTPUT_PATH}",
        "--chain-type=heavy",
        f"--sequences=./tests/{DOCKER_INPUT_HEAVY}"
    ]

    print("[INFO] Running docker CLI for heavy chain prediction")
    result = subprocess.run(docker_cmd, capture_output=True, text=True)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    assert result.returncode == 0, "Docker CLI failed!"

    # Your app saves output as "<basename>_alignairr_results.csv"
    output_csv = os.path.join(OUTPUT_DIR, "test_heavy_alignairr_results.csv")
    compare_csv(output_csv, HEAVY_GOLDEN_FILE)

def test_alignair_cli_predict_light():
    # Copy test input to mounted location
    input_mounted = os.path.join(OUTPUT_DIR, DOCKER_INPUT_LIGHT)
    shutil.copy(LIGHT_INPUT_FILE, input_mounted)

    docker_cmd = [
        "docker", "run", "--rm",
        "-v", f"{OUTPUT_DIR}:/tests",
        DOCKER_IMAGE_TAGGED,
        "python", "app.py", "run",
        f"--model-checkpoint={DOCKER_LIGHT_MODEL_PATH}",
        f"--save-path={DOCKER_OUTPUT_PATH}",
        "--chain-type=light",
        f"--sequences=./tests/{DOCKER_INPUT_LIGHT}"
    ]

    print("[INFO] Running docker CLI for light chain prediction")
    result = subprocess.run(docker_cmd, capture_output=True, text=True)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    assert result.returncode == 0, "Docker CLI failed!"

    output_csv = os.path.join(OUTPUT_DIR, "test_light_alignairr_results.csv")
    compare_csv(output_csv, LIGHT_GOLDEN_FILE)
