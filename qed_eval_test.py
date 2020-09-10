# Lint as: python3
"""Tests for google3.third_party.py.language.google.qed.qed_eval."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import qed_eval
from absl.testing import absltest

example_1 = """
{
    "example_id": -6560319052930436991,
    "title_text": "Flight (Grey's Anatomy)",
    "url": "https://en.wikipedia.org//w/index.php?title=Flight_(Grey%27s_Anatomy)&amp;oldid=804214813",
    "question_text": "who died in the plane crash greys anatomy",
    "paragraph_text": "`` Flight '' is the twenty - fourth and final episode of the eighth season of the American television medical drama Grey 's Anatomy , and the show 's 172nd episode overall . It was written by series creator Shonda Rhimes , and directed by Rob Corn . The episode was originally broadcast on the American Broadcasting Company ( ABC ) in the United States on May 17 , 2012 . In the episode , six doctors from Seattle Grace Mercy West Hospital who are victims of an aviation accident fight to stay alive , but Dr. Lexie Grey ( Chyler Leigh ) ultimately dies . Other storylines occur in Seattle where Dr. Richard Webber ( James Pickens , Jr . ) plans his annual dinner for the departing residents , Dr. Owen Hunt ( Kevin McKidd ) fires Dr. Teddy Altman ( Kim Raver ) , and Dr. Miranda Bailey ( Chandra Wilson ) gets engaged .",
    "sentence_starts": [
        0,
        174,
        250,
        372,
        556
    ],
    "original_nq_answers": [
        [
            {
                "start": 506,
                "end": 520,
                "string": "Dr. Lexie Grey"
            }
        ],
        [
            {
                "start": 506,
                "end": 537,
                "string": "Dr. Lexie Grey ( Chyler Leigh )"
            }
        ],
        [
            {
                "start": 506,
                "end": 520,
                "string": "Dr. Lexie Grey"
            },
            {
                "start": 523,
                "end": 535,
                "string": "Chyler Leigh"
            }
        ]
    ],
    "annotation": {
        "referential_equalities": [
            {
                "question_reference": {
                    "start": 12,
                    "end": 27,
                    "string": "the plane crash"
                },
                "sentence_reference": {
                    "start": 459,
                    "end": 479,
                    "bridge": false,
                    "string": "an aviation accident"
                }
            },
            {
                "question_reference": {
                    "start": 28,
                    "end": 41,
                    "string": "greys anatomy"
                },
                "sentence_reference": {
                    "start": -1,
                    "end": -1,
                    "bridge": "of",
                    "string": ""
                }
            }
        ],
        "answer": [
            {
                "sentence_reference": {
                    "start": 506,
                    "end": 520,
                    "bridge": false,
                    "string": "Dr. Lexie Grey"
                },
                "paragraph_reference": {
                    "start": 506,
                    "end": 520,
                    "string": "Dr. Lexie Grey"
                }
            }
        ],
        "explanation_type": "single_sentence",
        "selected_sentence": {
            "start": 372,
            "end": 556,
            "string": "In the episode , six doctors from Seattle Grace Mercy West Hospital who are victims of an aviation accident fight to stay alive , but Dr. Lexie Grey ( Chyler Leigh ) ultimately dies . "
        }
    }
}"""

example_2 = """
{
    "example_id": -4340755100872459608,
    "title_text": "Health (gaming)",
    "url": "https://en.wikipedia.org//w/index.php?title=Health_(gaming)&amp;oldid=819315199",
    "question_text": "what does hp mean in war and order",
    "paragraph_text": "Health or vitality is an attribute assigned to entities , such as the player character , enemies and objects within a role - playing or video game , that indicates its state in combat . Health is usually measured in hit points or health points , shortened to HP . When the HP of a player character reaches zero , the player may lose a life or their character might become incapacitated or die . When the HP of an enemy reaches zero , it may be defeated or die and the player is usually rewarded in some way .",
    "sentence_starts": [
        0,
        186,
        264,
        395
    ],
    "original_nq_answers": [
        [
            {
                "start": 216,
                "end": 243,
                "string": "hit points or health points"
            }
        ]
    ],
    "annotation": {
        "referential_equalities": [
            {
                "question_reference": {
                    "start": 10,
                    "end": 12,
                    "string": "hp"
                },
                "sentence_reference": {
                    "start": 259,
                    "end": 261,
                    "bridge": false,
                    "string": "HP"
                }
            }
        ],
        "answer": [
            {
                "sentence_reference": {
                    "start": 216,
                    "end": 243,
                    "bridge": false,
                    "string": "hit points or health points"
                },
                "paragraph_reference": {
                    "start": 216,
                    "end": 243,
                    "string": "hit points or health points"
                }
            }
        ],
        "explanation_type": "single_sentence",
        "selected_sentence": {
            "start": 186,
            "end": 264,
            "string": "Health is usually measured in hit points or health points , shortened to HP . "
        }
    }
}"""


class QedEvalTest(absltest.TestCase):

  def setUp(self):
    super(QedEvalTest, self).setUp()
    self._annotation_jsonlines = [json.loads(example_1), json.loads(example_2)]
    annot_elems = [
        qed_eval.load_single_line(l) for l in self._annotation_jsonlines
    ]
    self.annotation_dict = {elem.example_id: elem for elem in annot_elems}

  def get_span(self, text, span):
    return {"start": span[0], "end": span[1], "string": text[span[0]:span[1]]}

  def set_answer(self, example, answers):
    output_answers = example["annotation"]["answer"]
    output_answers.clear()
    for answer in answers:
      output_answers.append({
          "paragraph_reference":
              self.get_span(example["paragraph_text"], answer)
      })

  def set_refs(self, example, refs):
    refs_output = example["annotation"]["referential_equalities"]
    refs_output.clear()
    for ref in refs:
      question_span, sentence_span = ref
      refs_output.append({
          "question_reference":
              self.get_span(example["question_text"], question_span),
          "sentence_reference":
              self.get_span(example["paragraph_text"], sentence_span)
      })

  def test_strict_accuracy_on_correct(self):
    prediction_jsonlines = [json.loads(example_1), json.loads(example_2)]
    self.set_answer(prediction_jsonlines[0], [(506, 520)])  # correct answer
    self.set_refs(
        prediction_jsonlines[0],
        [
            ((12, 27), (459, 479)),  # two correct refs
            ((28, 41), (-1, -1))
        ])
    self.set_answer(prediction_jsonlines[1], [(216, 243)])  # correct answer
    self.set_refs(prediction_jsonlines[1],
                  [((10, 12), (259, 261))])  # one correct ref

    pred_elems = [qed_eval.load_single_line(l) for l in prediction_jsonlines]
    prediction_dict = {elem.example_id: elem for elem in pred_elems}
    score_dict = qed_eval.compute_scores(
        self.annotation_dict, prediction_dict, strict=True)

    self.assertEqual(score_dict["exact_match_accuracy"], 1.0)
    self.assertEqual(score_dict["pair"][0], 1.0)
    self.assertEqual(score_dict["pair"][1], 1.0)
    self.assertEqual(score_dict["question_mention"][0], 1.0)
    self.assertEqual(score_dict["question_mention"][1], 1.0)
    self.assertEqual(score_dict["context_mention"][0], 1.0)
    self.assertEqual(score_dict["context_mention"][1], 1.0)
    self.assertEqual(score_dict["all_mention"][0], 1.0)
    self.assertEqual(score_dict["all_mention"][1], 1.0)
    self.assertEqual(score_dict["answer_accuracy"], 1.0)

  def test_strict_accuracy(self):
    prediction_jsonlines = [json.loads(example_1), json.loads(example_2)]
    self.set_answer(prediction_jsonlines[0], [(506, 520)])  # correct answer
    self.set_refs(prediction_jsonlines[0],
                  [((28, 41), (-1, -1))])  # one correct ref, one missing
    self.set_answer(prediction_jsonlines[1], [(217, 243)])  # wrong answer
    self.set_refs(prediction_jsonlines[1],
                  [((10, 12), (259, 261))])  # one correct ref

    pred_elems = [qed_eval.load_single_line(l) for l in prediction_jsonlines]
    prediction_dict = {elem.example_id: elem for elem in pred_elems}
    score_dict = qed_eval.compute_scores(
        self.annotation_dict, prediction_dict, strict=True)

    self.assertEqual(score_dict["exact_match_accuracy"], 0.5)
    self.assertEqual(score_dict["pair"][0], 1.0)
    self.assertEqual(score_dict["pair"][1], 2.0 / 3.0)
    self.assertEqual(score_dict["question_mention"][0], 1.0)
    self.assertEqual(score_dict["question_mention"][1], 2.0 / 3.0)
    self.assertEqual(score_dict["context_mention"][0], 1.0)
    self.assertEqual(score_dict["context_mention"][1], 1.0 / 2.0)
    self.assertEqual(score_dict["all_mention"][0], 1.0)
    self.assertEqual(score_dict["all_mention"][1], 3.0 / 5.0)
    self.assertEqual(score_dict["answer_accuracy"], 1.0 / 2.0)

  def test_non_strict_accuracy(self):
    prediction_jsonlines = [json.loads(example_1), json.loads(example_2)]
    self.set_answer(prediction_jsonlines[0], [(506, 520)])  # correct answer
    self.set_refs(
        prediction_jsonlines[0],
        [
            ((15, 27), (462, 479)),  # one correct ref (non strict)
            ((28, 41), (-1, -1))
        ])  # one correct ref
    self.set_answer(prediction_jsonlines[1],
                    [(217, 243)])  # correct answer (non strict)
    self.set_refs(prediction_jsonlines[1],
                  [((10, 12), (259, 261))])  # one correct ref

    pred_elems = [qed_eval.load_single_line(l) for l in prediction_jsonlines]
    prediction_dict = {elem.example_id: elem for elem in pred_elems}

    score_dict = qed_eval.compute_scores(
        self.annotation_dict, prediction_dict, strict=False)
    print(score_dict)
    self.assertEqual(score_dict["exact_match_accuracy"], 1.0)
    self.assertEqual(score_dict["pair"][0], 1.0)
    self.assertEqual(score_dict["pair"][1], 1.0)
    self.assertEqual(score_dict["question_mention"][0], 1.0)
    self.assertEqual(score_dict["question_mention"][1], 1.0)
    self.assertEqual(score_dict["context_mention"][0], 1.0)
    self.assertEqual(score_dict["context_mention"][1], 1.0)
    self.assertEqual(score_dict["all_mention"][0], 1.0)
    self.assertEqual(score_dict["all_mention"][1], 1.0)
    self.assertEqual(score_dict["answer_accuracy"], 1.0)

    score_dict = qed_eval.compute_scores(
        self.annotation_dict, prediction_dict, strict=True)
    print(score_dict)
    self.assertEqual(score_dict["exact_match_accuracy"], 0.5)
    self.assertEqual(score_dict["pair"][0], 2.0 / 3.0)
    self.assertEqual(score_dict["pair"][1], 2.0 / 3.0)
    self.assertEqual(score_dict["question_mention"][0], 2.0 / 3.0)
    self.assertEqual(score_dict["question_mention"][1], 2.0 / 3.0)
    self.assertEqual(score_dict["context_mention"][0], 0.5)
    self.assertEqual(score_dict["context_mention"][1], 0.5)
    self.assertEqual(score_dict["all_mention"][0], 3.0 / 5.0)
    self.assertEqual(score_dict["all_mention"][1], 3.0 / 5.0)
    self.assertEqual(score_dict["answer_accuracy"], 1.0 / 2.0)

  def test_non_strict_accuracy_not_enough_overlap(self):
    prediction_jsonlines = [json.loads(example_1), json.loads(example_2)]
    self.set_answer(prediction_jsonlines[0], [(500, 510)])  # correct answer
    self.set_refs(
        prediction_jsonlines[0],
        [
            ((16, 27), (462, 481)),  # one wrong ref (overlap 0.88)
            ((30, 45), (0, 0))
        ])  # one wrong ref
    self.set_answer(prediction_jsonlines[1], [(230, 250)])  # correct answer
    self.set_refs(prediction_jsonlines[1],
                  [((9, 12), (259, 262))])  # one wrong ref

    pred_elems = [qed_eval.load_single_line(l) for l in prediction_jsonlines]
    prediction_dict = {elem.example_id: elem for elem in pred_elems}

    score_dict = qed_eval.compute_scores(
        self.annotation_dict, prediction_dict, strict=False)
    print(score_dict)
    self.assertEqual(score_dict["exact_match_accuracy"], 0.0)
    self.assertEqual(score_dict["pair"][0], 0.0)
    self.assertEqual(score_dict["pair"][1], 0.0)
    self.assertEqual(score_dict["question_mention"][0], 0.0)
    self.assertEqual(score_dict["question_mention"][1], 0.0)
    self.assertEqual(score_dict["context_mention"][0], 0.0)
    self.assertEqual(score_dict["context_mention"][1], 0.0)
    self.assertEqual(score_dict["all_mention"][0], 0.0)
    self.assertEqual(score_dict["all_mention"][1], 0.0)
    self.assertEqual(score_dict["answer_accuracy"], 0.0)

    score_dict = qed_eval.compute_scores(
        self.annotation_dict, prediction_dict, strict=True)
    print(score_dict)
    self.assertEqual(score_dict["exact_match_accuracy"], 0.0)
    self.assertEqual(score_dict["pair"][0], 0.0)
    self.assertEqual(score_dict["pair"][1], 0.0)
    self.assertEqual(score_dict["question_mention"][0], 0.0)
    self.assertEqual(score_dict["question_mention"][1], 0.0)
    self.assertEqual(score_dict["context_mention"][0], 0.0)
    self.assertEqual(score_dict["context_mention"][1], 0.0)
    self.assertEqual(score_dict["all_mention"][0], 0.0)
    self.assertEqual(score_dict["all_mention"][1], 0.0)
    self.assertEqual(score_dict["answer_accuracy"], 0.0)

  def test_accuracy_for_alternative_answers(self):
    prediction_jsonlines = [json.loads(example_1), json.loads(example_2)]
    self.set_answer(prediction_jsonlines[0],
                    [(506, 537)])  # correct answer (alternative answer)
    self.set_answer(prediction_jsonlines[1], [(216, 243)])  # correct answer

    pred_elems = [qed_eval.load_single_line(l) for l in prediction_jsonlines]
    prediction_dict = {elem.example_id: elem for elem in pred_elems}
    score_dict = qed_eval.compute_scores(
        self.annotation_dict, prediction_dict, strict=True)

    self.assertEqual(score_dict["answer_accuracy"], 1.0)

  def test_accuracy_for_alternative_answers_with_multiple_spans(self):
    prediction_jsonlines = [json.loads(example_1), json.loads(example_2)]
    self.set_answer(prediction_jsonlines[0],
                    [(524, 536), (505, 519)])  # correct alternative, non strict
    self.set_answer(prediction_jsonlines[1], [(216, 243)])  # correct answer

    pred_elems = [qed_eval.load_single_line(l) for l in prediction_jsonlines]
    prediction_dict = {elem.example_id: elem for elem in pred_elems}
    score_dict = qed_eval.compute_scores(
        self.annotation_dict, prediction_dict, strict=True)
    self.assertEqual(score_dict["answer_accuracy"], 0.5)

    score_dict = qed_eval.compute_scores(
        self.annotation_dict, prediction_dict, strict=False)
    self.assertEqual(score_dict["answer_accuracy"], 1.0)


if __name__ == "__main__":
  absltest.main()
