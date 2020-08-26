"""Tests for qed_eval.py"""

import qed_eval
from absl.testing import absltest


class QedEvalTest(absltest.TestCase):

  def setUp(self):
    super(QedEvalTest, self).setUp()
    self._annotation_jsonlines = [{
        "example_id": -6560319052930436991,
        "paragraph_text":
            "`` Flight '' is the twenty - fourth and final episode of the "
            "eighth season of the American television medical drama Grey 's "
            "Anatomy , and the show 's 172nd episode overall . It was written "
            "by series creator Shonda Rhimes , and directed by Rob Corn . The "
            "episode was originally broadcast on the American Broadcasting "
            "Company ( ABC ) in the United States on May 17 , 2012 . In the "
            "episode , six doctors from Seattle Grace Mercy West Hospital who "
            "are victims of an aviation accident fight to stay alive , but Dr."
            " Lexie Grey ( Chyler Leigh ) ultimately dies . Other storylines "
            "occur in Seattle where Dr. Richard Webber ( James Pickens , Jr . "
            ") plans his annual dinner for the departing residents , Dr. Owen "
            "Hunt ( Kevin McKidd ) fires Dr. Teddy Altman ( Kim Raver ) , and "
            "Dr. Miranda Bailey ( Chandra Wilson ) gets engaged .",
        "question_text": "who died in the plane crash greys anatomy",
        "title_text": "Flight (Grey's Anatomy)",
        "answer_spans": [[506, 520]],
        "answer_text": ["Dr. Lexie Grey"],
        "annotation": [{
            "context_entity_text": "an aviation accident",
            "context_entity_span": [459, 479],
            "question_entity_text": "the plane crash",
            "question_entity_span": [12, 27]
        }, {
            "context_entity_text": "",
            "context_entity_span": [-1, -1],
            "question_entity_text": "greys anatomy",
            "question_entity_span": [28, 41]
        }],
        "answer_type": "single_sentence"
    }, {
        "example_id": -4340755100872459608,
        "paragraph_text":
            "Health or vitality is an attribute assigned to entities , such as"
            " the player character , enemies and objects within a role - "
            "playing or video game , that indicates its state in combat . "
            "Health is usually measured in hit points or health points , "
            "shortened to HP . When the HP of a player character reaches zero "
            ", the player may lose a life or their character might become "
            "incapacitated or die . When the HP of an enemy reaches zero , it "
            "may be defeated or die and the player is usually rewarded in some"
            " way .",
        "question_text": "what does hp mean in war and order",
        "title_text": "Health (gaming)",
        "answer_spans": [[216, 243]],
        "answer_text": ["hit points or health points"],
        "annotation": [{
            "context_entity_text": "HP",
            "context_entity_span": [259, 261],
            "question_entity_text": "hp",
            "question_entity_span": [10, 12]
        }],
        "answer_type": "single_sentence"
    }]
    annot_elems = [
        qed_eval.load_single_line(l) for l in self._annotation_jsonlines
    ]
    self.annotation_dict = {elem.example_id: elem for elem in annot_elems}

  def test_strict_accuracy(self):
    prediction_jsonlines = [{
        "example_id": -6560319052930436991,
        "paragraph_text":
            "`` Flight '' is the twenty - fourth and final episode of the "
            "eighth season of the American television medical drama Grey 's "
            "Anatomy , and the show 's 172nd episode overall . It was written "
            "by series creator Shonda Rhimes , and directed by Rob Corn . The "
            "episode was originally broadcast on the American Broadcasting "
            "Company ( ABC ) in the United States on May 17 , 2012 . In the "
            "episode , six doctors from Seattle Grace Mercy West Hospital who "
            "are victims of an aviation accident fight to stay alive , but Dr."
            " Lexie Grey ( Chyler Leigh ) ultimately dies . Other storylines "
            "occur in Seattle where Dr. Richard Webber ( James Pickens , Jr . "
            ") plans his annual dinner for the departing residents , Dr. Owen "
            "Hunt ( Kevin McKidd ) fires Dr. Teddy Altman ( Kim Raver ) , and "
            "Dr. Miranda Bailey ( Chandra Wilson ) gets engaged .",
        "question_text": "who died in the plane crash greys anatomy",
        "title_text": "Flight (Grey's Anatomy)",
        "answer_spans": [[506, 520]],
        "answer_text": ["Dr. Lexie Grey"],
        "annotation": [{
            "context_entity_text": "",
            "context_entity_span": [-1, -1],
            "question_entity_text": "greys anatomy",
            "question_entity_span": [28, 41]
        }],
        "answer_type": "single_sentence"
    }, {
        "example_id": -4340755100872459608,
        "paragraph_text":
            "Health or vitality is an attribute assigned to entities , such as"
            " the player character , enemies and objects within a role - "
            "playing or video game , that indicates its state in combat . "
            "Health is usually measured in hit points or health points , "
            "shortened to HP . When the HP of a player character reaches zero "
            ", the player may lose a life or their character might become "
            "incapacitated or die . When the HP of an enemy reaches zero , it "
            "may be defeated or die and the player is usually rewarded in some"
            " way .",
        "question_text": "what does hp mean in war and order",
        "title_text": "Health (gaming)",
        "answer_spans": [[216, 243]],
        "answer_text": ["hit points or health points"],
        "annotation": [{
            "context_entity_text": "HP",
            "context_entity_span": [259, 261],
            "question_entity_text": "hp",
            "question_entity_span": [10, 12]
        }],
        "answer_type": "single_sentence"
    }]
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

  def test_non_strict_accuracy(self):
    prediction_jsonlines = [{
        "example_id": -6560319052930436991,
        "paragraph_text":
            "`` Flight '' is the twenty - fourth and final episode of the "
            "eighth season of the American television medical drama Grey 's "
            "Anatomy , and the show 's 172nd episode overall . It was written "
            "by series creator Shonda Rhimes , and directed by Rob Corn . The "
            "episode was originally broadcast on the American Broadcasting "
            "Company ( ABC ) in the United States on May 17 , 2012 . In the "
            "episode , six doctors from Seattle Grace Mercy West Hospital who "
            "are victims of an aviation accident fight to stay alive , but Dr."
            " Lexie Grey ( Chyler Leigh ) ultimately dies . Other storylines "
            "occur in Seattle where Dr. Richard Webber ( James Pickens , Jr . "
            ") plans his annual dinner for the departing residents , Dr. Owen "
            "Hunt ( Kevin McKidd ) fires Dr. Teddy Altman ( Kim Raver ) , and "
            "Dr. Miranda Bailey ( Chandra Wilson ) gets engaged .",
        "question_text": "who died in the plane crash greys anatomy",
        "title_text": "Flight (Grey's Anatomy)",
        "answer_spans": [[506, 520]],
        "answer_text": ["Dr. Lexie Grey"],
        "annotation": [{
            "context_entity_text": "aviation accident",
            "context_entity_span": [462, 479],
            "question_entity_text": "plane crash",
            "question_entity_span": [16, 27]
        }, {
            "context_entity_text": "",
            "context_entity_span": [-1, -1],
            "question_entity_text": "greys anatomy",
            "question_entity_span": [28, 41]
        }],
        "answer_type": "single_sentence"
    }, {
        "example_id": -4340755100872459608,
        "paragraph_text":
            "Health or vitality is an attribute assigned to entities , such as"
            " the player character , enemies and objects within a role - "
            "playing or video game , that indicates its state in combat . "
            "Health is usually measured in hit points or health points , "
            "shortened to HP . When the HP of a player character reaches zero "
            ", the player may lose a life or their character might become "
            "incapacitated or die . When the HP of an enemy reaches zero , it "
            "may be defeated or die and the player is usually rewarded in some"
            " way .",
        "question_text": "what does hp mean in war and order",
        "title_text": "Health (gaming)",
        "answer_spans": [[216, 243]],
        "answer_text": ["hit points or health points"],
        "annotation": [{
            "context_entity_text": "HP",
            "context_entity_span": [259, 261],
            "question_entity_text": "hp",
            "question_entity_span": [10, 12]
        }],
        "answer_type": "single_sentence"
    }]
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


if __name__ == "__main__":
  absltest.main()
