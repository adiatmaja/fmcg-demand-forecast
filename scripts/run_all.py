"""Run the full FMCG forecasting pipeline: data generation → demand → sales."""
import argparse
import logging
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run complete FMCG forecasting pipeline")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--skip-generate", action="store_true", help="Skip data generation")
    parser.add_argument("--skip-demand", action="store_true", help="Skip demand pipeline")
    parser.add_argument("--skip-sales", action="store_true", help="Skip sales pipeline")
    parser.add_argument("--sales-mode", choices=["individual", "merged", "both"], default="both")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger(__name__)
    scripts_dir = Path(__file__).parent

    def run(script: str, extra_args: list[str] | None = None) -> None:
        cmd = [sys.executable, str(scripts_dir / script), "--config", args.config]
        if extra_args:
            cmd.extend(extra_args)
        logger.info("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            logger.error("Script %s failed with return code %d", script, result.returncode)
            sys.exit(result.returncode)

    if not args.skip_generate:
        logger.info("=== Phase 1: Data Generation ===")
        run("generate_data.py")

    if not args.skip_demand:
        logger.info("=== Phase 2: Demand Forecasting ===")
        run("run_demand.py")

    if not args.skip_sales:
        logger.info("=== Phase 3: Sales Forecasting ===")
        run("run_sales.py", ["--mode", args.sales_mode])

    logger.info("=== Pipeline complete ===")


if __name__ == "__main__":
    main()
