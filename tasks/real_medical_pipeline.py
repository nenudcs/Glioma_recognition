"""Pipeline factory for trained MedicalNet/nnUNet checkpoints.

This is intentionally separate from ``tasks.real_pipeline`` so the repository's
contract tests and dummy baseline remain runnable without MedicalNet or nnUNet.
Enable it with:

    export COMPETITION_PIPELINE_FACTORY=tasks.real_medical_pipeline:build_pipeline
    export GOAL3_BACKEND=medicalnet
    export GOAL3_CHECKPOINT=/.../goal3_medicalnet.pt
    export GOAL4_BACKEND=medicalnet
    export GOAL4_CHECKPOINT=/.../goal4_medicalnet.pt
    export GOAL5_BACKEND=nnunet
    export GOAL5_NNUNET_MODEL_FOLDER=/.../nnUNet_results/...
"""

from __future__ import annotations

from pipeline.inference import InferencePipeline, StudyTaskBinding
from tasks.goal1_authenticity.config import Goal1Config
from tasks.goal1_authenticity.task import Goal1AuthenticityTask
from tasks.goal2_duplicate.config import Goal2DuplicateConfig
from tasks.goal2_duplicate.inference import DuplicateProbe
from tasks.goal2_duplicate.task import Goal2DuplicateRecorder
from tasks.goal2_stitched.config import Goal2StitchedConfig
from tasks.goal2_stitched.gating import GatedStudyTask, gated_fields
from tasks.goal2_stitched.task import Goal2StitchedTask
from tasks.goal3.task import Goal3Task
from tasks.goal4.task import Goal4Task
from tasks.goal5.task import Goal5Task


def build_pipeline(
    goal1_config: Goal1Config | None = None,
    goal2_config: Goal2StitchedConfig | None = None,
    duplicate_config: Goal2DuplicateConfig | None = None,
) -> InferencePipeline:
    duplicate = duplicate_config or Goal2DuplicateConfig.from_env()
    probe = DuplicateProbe(duplicate) if duplicate.enabled else None
    goal3 = Goal3Task()
    goal4 = Goal4Task()
    goal5 = Goal5Task()
    return InferencePipeline(
        study_tasks=(
            StudyTaskBinding("goal1", Goal1AuthenticityTask(goal1_config)),
            StudyTaskBinding("goal2_stitched", Goal2StitchedTask(goal2_config, duplicate_probe=probe)),
            StudyTaskBinding("goal3", GatedStudyTask(goal3, "goal3", goal3.predict, gated_fields=gated_fields())),
            StudyTaskBinding("goal5", GatedStudyTask(goal5, "goal5", goal5.predict, gated_fields=gated_fields())),
            StudyTaskBinding("goal4", GatedStudyTask(goal4, "goal4", goal4.predict, gated_fields=gated_fields())),
        ),
        duplicate_task=Goal2DuplicateRecorder(duplicate, probe=probe),
    )
