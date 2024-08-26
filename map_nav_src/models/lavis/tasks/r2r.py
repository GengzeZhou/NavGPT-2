"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os

from lavis.common.dist_utils import main_process
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask


@registry.register_task("r2r_captioning")
class R2RCaptionTask(BaseTask):
    def __init__(self, num_beams, max_len, min_len, evaluate, report_metric=True):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.evaluate = evaluate

        self.report_metric = report_metric

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.num_beams
        max_len = run_cfg.max_len
        min_len = run_cfg.min_len
        evaluate = run_cfg.evaluate

        report_metric = run_cfg.get("report_metric", True)

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            report_metric=report_metric,
        )

    def valid_step(self, model, samples):
        results = []

        # run_cfg = slf.cfg.run_cfg
        llm_thoughts = model.generate(
            samples,
            use_nucleus_sampling=False,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len,
        )

        sample_ids = samples["sample_ids"]
        target_outputs = samples["text_output"]
        for llm_thought, target_output, sample_id in zip(llm_thoughts, target_outputs, sample_ids):
            results.append(
                {
                    "llm_thought": llm_thought,
                    "target_output": target_output,
                    "sample_id": sample_id,
                }
            )

        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="sample_id",
        )

        if self.report_metric:
            metrics = self._report_metrics(
                eval_result=val_result, split_name=split_name
            )
        else:
            metrics = {"agg_metrics": 0.0}

        return metrics

    @main_process
    def _report_metrics(self, eval_result, split_name):

        # TODO better way to define this
        coco_val = coco_caption_eval(eval_result)

        agg_metrics = coco_val.eval["CIDEr"] + coco_val.eval["Bleu_4"]
        log_stats = {split_name: {k: v for k, v in coco_val.eval.items()}}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        coco_res = {k: v for k, v in coco_val.eval.items()}
        coco_res["agg_metrics"] = agg_metrics

        return coco_res

def coco_caption_eval(results):
    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(results)

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")

    return coco_eval

# TODO better structure for this.
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

class COCOEvalCap:
    def __init__(self, results):
        self.evalSamples = []
        self.eval = {}
        self.sampleToEval = {}
        self.results = results

    def evaluate(self):
        gts = {}
        res = {}
        for sample in self.results:
            sampleId = sample['sample_id']
            gts[sampleId] = [{'caption':sample['target_output']}]
            res[sampleId] = [{'caption':sample['llm_thought']}]

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setSampleToEvalSamples(scs, gts.keys(), m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setSampleToEvalSamples(scores, gts.keys(), method)
                print("%s: %0.3f"%(method, score))
        self.setEvalSamples()

    def setEval(self, score, method):
        self.eval[method] = score

    def setSampleToEvalSamples(self, scores, sampleIds, method):
        for sampleId, score in zip(sampleIds, scores):
            if not sampleId in self.sampleToEval:
                self.sampleToEval[sampleId] = {}
                self.sampleToEval[sampleId]["sample_id"] = sampleId
            self.sampleToEval[sampleId][method] = score

    def setEvalSamples(self):
        self.evalSamples = [eval for sampleId, eval in self.sampleToEval.items()]