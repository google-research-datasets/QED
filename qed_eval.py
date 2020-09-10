r"""Methods for evaluating QED annotations.

This script is meant to be run in python3. All offsets are unicode char offsets.
All start char offsets are inclusive and end char offsets are exclusive.
All strings being operated on are assumed to be of type Text.

   Example Usage:
   qed_eval \
     --annotation= \
     --prediction=

   Both files should be in the following format:
   {"example_id": -3290814144789249484,
    "paragraph_text": "The first Nobel Prize in Physics was awarded in 1901
        to Wilhelm Conrad R\u00f6ntgen , of Germany , who received 150,782 SEK ,
        which is equal to 7,731,004 SEK in December 2007 . John Bardeen is the
        only laureate to win the prize twice -- in 1956 and 1972 .
        Maria Sk\u0142odowska - Curie also won two Nobel Prizes , for physics in
        1903 and chemistry in 1911 . William Lawrence Bragg was , until October
        2014 , the youngest ever Nobel laureate ; he won the prize in 1915 at
        the age of 25 . Two women have won the prize : Curie and Maria Goeppert
        - Mayer ( 1963 ) . As of 2017 , the prize has been awarded to 206
        individuals . There have been six years in which the Nobel Prize in
        Physics was not awarded ( 1916 , 1931 , 1934 , 1940 -- 1942 ) .",
    "question_text": "who got the first nobel prize in physics",
    "title_text": "List of Nobel laureates in Physics",
    "annotation": {
        "referential_equalities": [{
            "question_reference": {
                "start": 8, "end": 40,
                "string": "the first nobel prize in physics"
            },
            "sentence_reference": {
                "start": 0, "end": 32, "bridge": false,
                "string": "The first Nobel Prize in Physics"
            }}],
        "answer": [{
            "sentence_reference": {
                "start": 56, "end": 78, "bridge": false,
                "string": "Wilhelm Conrad R\u00f6ntgen"
            },
            "paragraph_reference": {
                "start": 56, "end": 78,
                "string": "Wilhelm Conrad R\u00f6ntgen"
            }}]
        "explanation_type": "single_sentence",
        "selected_sentence": {
            "start": 0,
            "end": 172,
            "string": "The first Nobel Prize in Physics was awarded in 1901 to
                Wilhelm Conrad R\u00f6ntgen , of Germany , who received 150,782
                SEK , which is equal to 7,731,004 SEK in December 2007 . "
        }
    }}


  The output score dict will contain:
  {
      'exact_match_accuracy': ,
      'question_mention':
          (question_mention_p, question_mention_r, question_mention_f1),
      'context_mention':
          (context_mention_p, context_mention_r, context_mention_f1),
      'all_mention': (mention_p, mention_r, mention_f1),
      'pair': (pair_p, pair_r, pair_f1)
  }
"""

import json
import re
import string
from typing import Any, Collection, List, Mapping, Text, Tuple, Union

from absl import app
from absl import flags
from absl import logging
import attr

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'prediction', 'qed-dev.jsonlines',
    'Path to prediction jsonl file.')
flags.DEFINE_string(
    'annotation', 'qed-dev.jsonlines',
    'Path to annotation jsonl file.')
flags.DEFINE_bool(
    'strict', False, 'Whether to enforce strict match'
    'if false, entity mentions are considered equal'
    'if their mention span overlap AND their mention'
    'span matches after normalization')

MIN_F1_FOR_NON_STRICT_OVERLAP = 0.9


def normalize_text(text: Text) -> Text:
  """Lowercases text and removes punctuation, articles and extra whitespace."""

  def remove_articles(s):
    return re.sub(r'\b(a|an|the)\b', ' ', s)

  def replace_punctuation(s):
    to_replace = set(string.punctuation)
    return ''.join('' if ch in to_replace else ch for ch in s)

  def white_space_fix(s):
    return ' '.join(s.split())

  text = text.lower()
  text = replace_punctuation(text)
  text = remove_articles(text)
  text = white_space_fix(text)
  return text


@attr.s(frozen=True)
class Entity:
  """Entity in either document or query."""

  # Inclusive start char offset of this entity mention. -1 refers to the start
  # of the answering sentence. The answering sentence is given in the data
  # as example["annotation"]["selected_sentence"].
  start_offset = attr.ib(type=int)

  # Exclusive end char offset of this entity mention. -1 refers to the entire
  # answering sentence.
  end_offset = attr.ib(type=int)
  # type must be either context or query.
  type = attr.ib(type=Text)
  # entity mention text.
  text = attr.ib(type=Text)
  normalized_text = attr.ib(type=Text)

  def __hash__(self):
    return hash((self.start_offset, self.end_offset, self.type))

  def __eq__(self, other):
    return (self.start_offset == other.start_offset and
            self.end_offset == other.end_offset and self.type == other.type)


@attr.s
class QEDExample:
  """A single training/test example."""
  example_id = attr.ib(type=int)
  title = attr.ib(type=Text)
  question = attr.ib(type=Text)
  answer = attr.ib(type=List[Entity])
  nq_answers = attr.ib(type=List[List[Entity]])
  # the first entity is query entity, the second is document entity.
  aligned_nps = attr.ib(type=List[Tuple[Entity, Entity]])
  # either single_sentence or multi_sentence.
  explanation_type = attr.ib(type=Text)


def load_answer(answer: List[Mapping[Text, Any]]) -> List[Entity]:
  """Loads annotated QED answer, potentially composed of multiple spans."""
  output_answer = []
  for a in answer:
    output_answer.append(
        Entity(
            text=a['paragraph_reference']['string'],
            normalized_text=normalize_text(a['paragraph_reference']['string']),
            start_offset=a['paragraph_reference']['start'],
            end_offset=a['paragraph_reference']['end'],
            type='context'))
  return output_answer


def load_nq_answers(
    answer_list: List[List[Mapping[Text, Any]]]) -> List[List[Entity]]:
  """Loads annotated NQ answers, each potentially composed of multiple spans."""
  output_answer_list = []
  for answer in answer_list:
    output_answer = []
    for a in answer:
      output_answer.append(
          Entity(
              text=a['string'],
              normalized_text=normalize_text(a['string']),
              start_offset=a['start'],
              end_offset=a['end'],
              type='context'))
    output_answer_list.append(output_answer)
  return output_answer_list


def load_aligned_entities(alignment_dict: List[Mapping[Text, Any]],
                          question_text: Text,
                          context_text: Text) -> List[Tuple[Entity, Entity]]:
  """Loads aligned entities from json."""
  aligned_nps = []
  for single_np_alignment in alignment_dict:
    q_entity_text = single_np_alignment['question_reference']['string']
    q_entity_offset = (single_np_alignment['question_reference']['start'],
                       single_np_alignment['question_reference']['end'])
    c_entity_text = single_np_alignment['sentence_reference']['string']
    c_entity_offset = (single_np_alignment['sentence_reference']['start'],
                       single_np_alignment['sentence_reference']['end'])
    if q_entity_text != question_text[q_entity_offset[0]:q_entity_offset[1]]:
      logging.error(
          'Question entity offset not proper. from text: %s, from byte offset %s',
          q_entity_text, question_text[q_entity_offset[0]:q_entity_offset[1]])
      raise ValueError()

    question_entity = Entity(
        text=question_text[q_entity_offset[0]:q_entity_offset[1]],
        normalized_text=normalize_text(q_entity_text),
        start_offset=q_entity_offset[0],
        end_offset=q_entity_offset[1],
        type='question')
    if c_entity_offset[0] != -1:
      if c_entity_text != context_text[c_entity_offset[0]:c_entity_offset[1]]:
        logging.error(
            'Context entity offset not proper. from text: %s, from byte offset %s',
            c_entity_text, context_text[c_entity_offset[0]:c_entity_offset[1]])
        raise ValueError()
      doc_entity = Entity(
          text=context_text[c_entity_offset[0]:c_entity_offset[1]],
          normalized_text=normalize_text(c_entity_text),
          start_offset=c_entity_offset[0],
          end_offset=c_entity_offset[1],
          type='context')
    else:  # this is a bridging linguistic context instance.
      doc_entity = Entity(
          text='',
          start_offset=-1,
          end_offset=-1,
          type='context',
          normalized_text='')
    aligned_nps.append((question_entity, doc_entity))
  return aligned_nps


def load_single_line(elem: Mapping[Text, Any]) -> QEDExample:
  """Loads a QEDExample from json."""
  return QEDExample(
      example_id=elem['example_id'],
      title=elem['title_text'],
      question=elem['question_text'],
      answer=load_answer(elem['annotation'].get('answer', [])),
      nq_answers=load_nq_answers(elem['original_nq_answers']),
      aligned_nps=load_aligned_entities(
          elem['annotation'].get('referential_equalities', []),
          elem['question_text'],
          elem['paragraph_text']),
      explanation_type=elem['annotation']['explanation_type'])


def load_data(fname: Text) -> Mapping[int, QEDExample]:
  """Loads jsonl data and outputs a dict mapping example_id to QEDExample."""
  output_dict = {}
  incorrectly_formatted = 0
  with open(fname) as f:
    for line in f:
      try:
        elem = json.loads(line)
        example = load_single_line(elem)
        if example.explanation_type == 'single_sentence':
          output_dict[example.example_id] = example
      except ValueError:
        incorrectly_formatted += 1
  logging.info('%d examples not correctly formatted and skipped.',
               incorrectly_formatted)
  return output_dict


def overlap(ent1: Entity, ent2: Entity) -> bool:
  """Returns whether two entities overlap at least with 90% F1."""
  if (ent1.start_offset == -1 or ent1.end_offset == -1 or
      ent2.start_offset == -1 or ent2.end_offset == -1):
    return (ent1.start_offset, ent1.end_offset, ent2.start_offset,
            ent2.end_offset) == (-1, -1, -1, -1)

  # Compute F1 as follows:
  #   F1 = tp / (tp + (fp + fn) / 2)
  #   [------ ent1 --------]
  #             [------- ent2 -----]
  #   [-- fn --][--- tp ---][- fp -]
  tp = abs(ent1.end_offset - ent2.start_offset)
  fn = abs(ent2.start_offset - ent1.start_offset)
  fp = abs(ent2.end_offset - ent1.end_offset)
  f1 = tp / (tp + (fp + fn) / 2) if tp else 0.0
  return f1 >= MIN_F1_FOR_NON_STRICT_OVERLAP


def compute_mention_score(annotation: Collection[Entity],
                          prediction: Collection[Entity],
                          strict: bool) -> Tuple[float, float, float]:
  """Computes mention identification performance."""
  if strict:
    annot_entities = set([ent for ent in annotation if ent.start_offset != -1])
    pred_entities = set([ent for ent in prediction if ent.start_offset != -1])
    tp = len(annot_entities & pred_entities)
    tn = len(annot_entities - pred_entities)
    fn = len(pred_entities - annot_entities)
  else:
    tp, tn, fn = 0, 0, 0
    for annot_entity in annotation:
      found = False
      for pred_entity in prediction:
        if pred_entity.normalized_text == annot_entity.normalized_text:
          if overlap(pred_entity, annot_entity):
            found = True
            break
      if found:
        tp += 1
      else:
        tn += 1
    fn = len(prediction) - tp
  return tp, tn, fn


def compute_alignment_score(annotation: QEDExample, prediction: QEDExample,
                            strict: bool) -> Tuple[float, float, float]:
  """Computes the alignment match score."""
  if strict:
    annot_pairs = set(annotation.aligned_nps)
    pred_pairs = set(prediction.aligned_nps)
    tp = len(annot_pairs & pred_pairs)
    tn = len(annot_pairs - pred_pairs)
    fn = len(pred_pairs - annot_pairs)
  else:
    tp, tn, fn = 0, 0, 0
    for annot_q_ent, annot_doc_ent in annotation.aligned_nps:
      found = False
      for pred_q_ent, pred_doc_ent in prediction.aligned_nps:
        if pred_q_ent.normalized_text == annot_q_ent.normalized_text:
          if annot_doc_ent.normalized_text == pred_doc_ent.normalized_text:
            if overlap(pred_q_ent, annot_q_ent):
              if overlap(pred_doc_ent, annot_doc_ent):
                found = True
                break
      if found:
        tp += 1
      else:
        tn += 1
    fn = len(prediction.aligned_nps) - tp
  return tp, tn, fn


def compute_prf1(tp, tn, fn) -> Tuple[float, float, float]:
  """Computes precistion, recall and f1 from true/false positives/negatives."""
  if tp > 0:
    p, r = tp / (tp + fn), tp / (tp + tn)
    f1 = 2 * p * r / (p + r)
  else:
    p, r, f1 = 0.0, 0.0, 0.0
  return p, r, f1


def is_permutation_matrix(matrix: List[List[bool]]) -> bool:
  """Returns whether the given boolean matrix is a permutation matrix."""
  return (all(sum(v) == 1 for v in matrix) and
          sum(any(v) for v in matrix) == len(matrix))


def compute_answer_accuracy(annotation: QEDExample, prediction: QEDExample,
                            strict: bool) -> float:
  """Checks whether the predicted answer matches any of the annotated ones."""
  for annot_answer in [annotation.answer] + annotation.nq_answers:
    all_matches = []
    for a in annot_answer:
      all_matches.append([])
      for p in prediction.answer:
        if strict:
          all_matches[-1].append(a == p)
        else:
          all_matches[-1].append(overlap(a, p))

    # The all_matches matrix should basically a permutation matrix.
    if is_permutation_matrix(all_matches):
      return 1.0

  return 0.0


def compute_scores(
    annotation_dict: Mapping[int,
                             QEDExample], prediction_dict: Mapping[int,
                                                                   QEDExample],
    strict: bool) -> Mapping[Text, Union[float, Tuple[float, float, float]]]:
  """Compute scores."""
  score_dict = {}
  total_q_tp, total_q_tn, total_q_fn = 0, 0, 0
  total_c_tp, total_c_tn, total_c_fn = 0, 0, 0
  total_pair_tp, total_pair_tn, total_pair_fn = 0, 0, 0
  completely_correct_example_count = 0.0
  total_correct_answers = 0
  total_answers = 0
  for example_id in annotation_dict:
    if example_id not in prediction_dict:
      logging.info('Missing prediction for id %d', example_id)
    else:
      prediction = prediction_dict[example_id]
      annotation = annotation_dict[example_id]
      q_tp, q_tn, q_fn = compute_mention_score(
          [nps[0] for nps in annotation.aligned_nps],
          [nps[0] for nps in prediction.aligned_nps], strict)
      c_tp, c_tn, c_fn = compute_mention_score(
          [nps[1] for nps in annotation.aligned_nps],
          [nps[1] for nps in prediction.aligned_nps], strict)
      pair_tp, pair_tn, pair_fn = compute_alignment_score(
          annotation, prediction, strict)
      total_correct_answers += compute_answer_accuracy(annotation, prediction,
                                                       strict)
      total_answers += 1
      if pair_tn + pair_fn == 0:
        completely_correct_example_count += 1
      total_q_tp += q_tp
      total_q_tn += q_tn
      total_q_fn += q_fn
      total_c_tp += c_tp
      total_c_tn += c_tn
      total_c_fn += c_fn
      total_pair_tp += pair_tp
      total_pair_tn += pair_tn
      total_pair_fn += pair_fn
  question_mention_p, question_mention_r, question_mention_f1 = compute_prf1(
      total_q_tp, total_q_tn, total_q_fn)
  context_mention_p, context_mention_r, context_mention_f1 = compute_prf1(
      total_c_tp, total_c_tn, total_c_fn)
  mention_p, mention_r, mention_f1 = compute_prf1(total_q_tp + total_c_tp,
                                                  total_q_tn + total_c_tn,
                                                  total_q_fn + total_c_fn)
  pair_p, pair_r, pair_f1 = compute_prf1(total_pair_tp, total_pair_tn,
                                         total_pair_fn)
  logging.info('# of examples completely correct: %d',
               completely_correct_example_count)
  score_dict = {
      'exact_match_accuracy':
          completely_correct_example_count / len(annotation_dict),
      'question_mention':
          (question_mention_p, question_mention_r, question_mention_f1),
      'context_mention':
          (context_mention_p, context_mention_r, context_mention_f1),
      'all_mention': (mention_p, mention_r, mention_f1),
      'pair': (pair_p, pair_r, pair_f1),
      'answer_accuracy': (total_correct_answers / total_answers)
  }
  logging.info('Question mention P/R/F1 %.4f %.4f %.4f', question_mention_p,
               question_mention_r, question_mention_f1)
  logging.info('Context mention P/R/F1 %.4f %.4f %.4f', context_mention_p,
               context_mention_r, context_mention_f1)
  logging.info('Both Mention P/R/F1 %.4f %.4f %.4f', mention_p, mention_r,
               mention_f1)
  logging.info('Pair P/R/F1 %.4f %.4f %.4f', pair_p, pair_r, pair_f1)
  return score_dict


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  annotation_dict = load_data(FLAGS.annotation)
  logging.info('%d examples in annotation.', len(annotation_dict))
  prediction_dict = load_data(FLAGS.prediction)
  logging.info('%d examples in predicton.', len(prediction_dict))
  score_dict = compute_scores(annotation_dict, prediction_dict, FLAGS.strict)
  logging.info(score_dict)


if __name__ == '__main__':
  app.run(main)
