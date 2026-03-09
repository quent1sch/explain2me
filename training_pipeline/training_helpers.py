# training helpers

import wandb

def init_wandb(config, resume_checkpoint):
    """
    Initialize Weights & Biases safely.

    Logic:
    - If training resumes AND run_id provided → resume that run
    - If training resumes AND no run_id → resume latest run automatically
    - If not resuming → start fresh run
    """

    project = config["wandb"]["project"]
    run_id = config["wandb"].get("run_id")

    try:
        if resume_checkpoint:
            if run_id:
                print(f"Resuming W&B run with ID: {run_id}")
                wandb.init(
                    project=project,
                    id=run_id,
                    resume="must",
                )
            else:
                print("Resuming latest W&B run automatically.")
                wandb.init(
                    project=project,
                    resume="allow",
                )
        else:
            print("Starting new W&B run.")
            wandb.init(
                project=project,
            )

    except Exception as e:
        print("W&B initialization failed, starting fresh run.")
        print(e)
        wandb.init(project=project)

    print("W&B run id:", wandb.run.id)