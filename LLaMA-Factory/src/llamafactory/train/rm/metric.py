# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from ...extras.misc import numpify


if TYPE_CHECKING:
    from transformers import EvalPrediction


@dataclass
class ComputeAccuracy:
    r"""Compute reward accuracy and support `batch_eval_metrics`."""

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"accuracy": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        chosen_scores, rejected_scores = numpify(eval_preds.predictions[0]), numpify(eval_preds.predictions[1])
        if not chosen_scores.shape:
            self.score_dict["accuracy"].append(chosen_scores > rejected_scores)
        else:
            for i in range(len(chosen_scores)):
                self.score_dict["accuracy"].append(chosen_scores[i] > rejected_scores[i])

        if compute_result:
            return self._dump()


@dataclass
class ComputeVQAAccuracy:
    
    tokenizer: "PreTrainedTokenizer"
    
    r"""Compute VQA reward accuracy and support `batch_eval_metrics`."""

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"accuracy": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        for pred, label in zip(decoded_preds, decoded_labels):
            self.score_dict["accuracy"].append(pred.strip().lower() == label.strip().lower())

        if compute_result:
            return self._dump()