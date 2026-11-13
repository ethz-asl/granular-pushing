"""
Run the MPM simulation (mpm_sim.py) in a separate process.

A subprocess is needed because the simulation rewrites sys.argv for Newton's argument parser.
"""
import subprocess
import sys
from pathlib import Path

def run_mpm_simulation(config_file: str = None, headless: bool = True, config_overrides: dict = None):
    """
    Run mpm_sim.py (next to this file) as a subprocess and wait for it.

    A failed simulation is reported, not raised.

    Args:
        config_file: Optional path to a YAML config file (maps to --config).
        headless: If True, uses '--viewer null'. If False, uses '--viewer gl'.
        config_overrides: Dictionary of key-value pairs to override (maps to --set key=value).
    """
    print("=== Running MPM simulation (subprocess) ===")

    # 1. Locate target script (always same folder as this file)
    current_dir = Path(__file__).parent.resolve()
    target_script = current_dir / "mpm_sim.py"
    
    if not target_script.exists():
        sys.exit(f"Error: Could not find {target_script}")

    # 2. Build Base Command
    cmd = [sys.executable, str(target_script)]

    # 3. Handle Viewer / Headless
    if headless:
        cmd.extend(["--viewer", "null"])
    else:
        cmd.extend(["--viewer", "gl"])

    # 4. Handle Config File
    if config_file:
        cmd.extend(["--config", str(config_file)])

    # 5. Handle Overrides
    if config_overrides:
        cmd.append("--set")
        for key, value in config_overrides.items():
            cmd.append(f"{key}={value}")

    # 6. Execute
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print("\nSuccess.")
    except subprocess.CalledProcessError as e:
        print(f"\nFailed with error code {e.returncode}")
