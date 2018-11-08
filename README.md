# EmoContext

## Description
Project directory for SemEval19 Task 3 : EmoContext. (https://competitions.codalab.org/competitions/19790#participate)

SemEval19 Task 3 : EmoContext is a task to predict the emotion context (happy, sad, angry, others) of a 3 turn conversion.

Two examples of the task are given below:

|Id|Turn1|Turn2|Turn3|Label|
|--|-----|-----|-----|-----|
|156|You are funny|	 LOL I know that. :p| 	😊	 |happy|
|187|	Yeah exactly| 	Like you said, like brother like sister ;)| 	Not in the least|others|

## Method

Embed the conveersion (turn1, turn2, turn3) into a fixed embedding using the glove representation of each word. This embedding is then used to train a long short term memory (LSTM) recurrent neutral network (RNN) for predicting the emotion context.

## Instructions
1. Download glove word embedding from https://nlp.stanford.edu/projects/glove/ and save to /data/ folder.
2. Config src/emocontext/testBaseline.config
3. Run command: 
```python baseline.py -config testBaseline.config```
