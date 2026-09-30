from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from data.structures import CompetitionDataset, Series, Study
from pipeline.context import PipelineContext
from pipeline.inference import InferencePipeline


class ContractTest(unittest.TestCase):
    def test_all_dummy_tasks_fill_the_frozen_context(self) -> None:
        series = Series(
            series_uid="T1CE",
            modality="T1 enhanced",
            image=np.zeros((2, 3, 4), dtype=np.float32),
            affine=np.eye(4),
            source_path=Path("T1CE.nii.gz"),
        )
        study = Study("ACC001", (series,))
        dataset = CompetitionDataset(Path("dataset"), (study, Study("ACC002", (series,))))

        contexts, duplicates = InferencePipeline().run(dataset)
        context: PipelineContext = contexts["ACC001"]

        self.assertIsNotNone(context.goal1)
        self.assertIsNotNone(context.goal2_stitched)
        self.assertIsNotNone(context.goal3)
        self.assertIsNotNone(context.goal4)
        self.assertIsNotNone(context.goal5)
        self.assertEqual(1, len(duplicates.pairs))


if __name__ == "__main__":
    unittest.main()

