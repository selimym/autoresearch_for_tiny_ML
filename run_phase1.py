"""
run_phase1.py — Launch Phase 1 ShinkaEvolve architecture search.

Usage:
  uv run python run_phase1.py --config shinka_phase1.yaml
"""
import argparse, yaml
from shinka.core import ShinkaEvolveRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig


def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    eval_program_path = config.pop("eval_program_path", "shinka_evaluate.py")

    evo_config = EvolutionConfig(**config["evo_config"])
    job_config = LocalJobConfig(
        eval_program_path=eval_program_path,
        time="00:15:00",
    )
    db_config = DatabaseConfig(**config["db_config"])

    runner = ShinkaEvolveRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        max_evaluation_jobs=config.get("max_evaluation_jobs", 1),
        max_proposal_jobs=config.get("max_proposal_jobs", 2),
        max_db_workers=config.get("max_db_workers", 2),
        verbose=True,
    )
    runner.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="shinka_phase1.yaml")
    args = parser.parse_args()
    main(args.config)
