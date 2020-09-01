# QED: A Framework and Dataset for Explanations in Question Answering

This data release accompanies the paper:

**QED: A Framework and Dataset for Explanations in Question Answering**<br>
by Matthew Lamm, Jennimaria Palomaki, Chris Alberti, Daniel Andor, Eunsol Choi, Livio Baldini Soares, Michael Collins

## Overview of QED

## Data Description

The QED dataset is split into a training set of 5,000 examples and a validation set of 1,000 examples. Each QED example contains additional annotations for an example in the [Natural Questions](https://ai.google.com/research/NaturalQuestions) dataset. The additional annotation consists mainly of pairs of entity spans representing coreference links between the question and the answering passage in the NQ example. Please see the paper for a full description of the provided annotations.

The QED dataset is distributed in JSON Lines format, with one QED example per file. Two files are provided:
* nq-qed-train.jsonlines (the training set),
* nq-qed-dev.jsonlines (the validation set).

We additionally provide an official evaluation script to be used for appropriately comparing results on the validation set. The evaluation script expects a predicted file with the same format as the provided file, but with the annotation replaced by predicted values.

Each QED example is provided in the dataset as a dictionary with the following fields:
1. example_id (unique identifier matching NQ) - int
1. title_text (title of the NQ wikipedia page) - string
1. question_text (query from NQ) - string
1. paragraph_text (answering passage) - string
1. answer_spans (extractive answer to the question) - list of int pairs
1. answer_text (text for answer_spans) - list of strings
1. sentence_starts (start of each sentence in the passage) - list of ints
1. annotation (main QED contribution) - list of dicts, each with
1. question_entity_span (an entity the question) - int pair
1. question_entity_text (text for question_entity_span) - string
1. context_entity_span (an entity in the passage corresponding to the question entity if one exists) - int pair
1. context_entity_text (text for context_entity_span) - string

All of the input data for this task comes from Wikipedia, which is licensed for public use by the Creative Commons Attribution-ShareAlike 3.0 Unported License (CC BY-SA) and the GNU Free Documentation License (GFDL).

## Evaluation Scripts

## Baseline Results

## Contact

Please make use of github issues to ask questions/hold discussion.
