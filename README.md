# A Preference-based Feedback Corpus

Tian Lan, Ziao Ma, RongCheng Tu, Chen Xu, Heyan Huang, Xian-ling Mao

Beijing Insititute of Technology

---

#### TL;DR: We introduce a corpus for critique-tuning, which includes pairs of feedbacks for preference learning, like PPO or DPO (RLHF), aiming to improve the alignment between generated feedback with human judgments.

## Introduction

The self-critique capability of large-scale language models is currently a very popular research topic. The impressive self-crituque capabilities of GPT-4's powerful proprietary model stand out. However, in contrast, the current abilities of open-source models, like Llama2 series, are relatively limited. To address this issue, lots of efforts, such as [UltraCM-13B](https://huggingface.co/openbmb/UltraCM-13b) and [CritiqueLLM](https://arxiv.org/abs/2311.18702) have been taken to train open-source models by utilizing data generated by GPT-4 API, aiming to distill its powerful self-critique capabilities. However, the community currently lacks the preference-based self-critique data that consists of a high-quality and a relatively low-quality feedback, leading to the gaps between existing critique-tuned open-source models and human preferences and relevance. Therefore, to fill this gap, we have used GPT-4 to collect preference-based feedback (critique) dataset based on the [Feedback-Collection corpus](https://huggingface.co/datasets/kaist-ai/Feedback-Collection), further enhancing the self-critique capabilities of open-source models.


In addition to the [Feedback-Collection corpus](https://huggingface.co/datasets/kaist-ai/Feedback-Collection), there are other open-source critique-tuning datasets such as [UltraFeedback](https://github.com/OpenBMB/UltraFeedback/tree/main/src) and [Auto-J](https://github.com/GAIR-NLP/auto-j/blob/main/codes/leaderboard/README.md). However, these datasets lack essential scoring rubrics and reference responses, making it challenging to assess the quality of their feedbacks. Unlike these works, Feedback-Collection not only defines strict scoring criteria but also provides reference response, high-quality feedbacks generated by GPT-4, and [critique-tuned 7B/13B open-source models](https://huggingface.co/kaist-ai/prometheus-7b-v1.0). This comprehensive set of information contributes to a more thorough evaluation of critique capabilities.
Thus, we choose to build the preference-based feedback corpus based on Feedback-Collection corpus.

Specifically, we first inference their 7B and 13B critique-tuned LLMs on the training set of Feedback-Collection. Then, we collect generated feedbacks with a score difference of 2 or more compared to the ratings given by GPT-4 feedbacks (the overall scoring range being 1-5 points). This choice was made because these feedbacks exhibit significant score disparities in comparison to those generated by GPT-4, making it relatively easy to assess the quality differences between them.
Finally, we prompt GPT-4 to choose which feedback is better by using Chain-of-Thought, i.e. generating the rationale about these feedbacks. The complete prompt is shown as follow:
```python
'''
###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, a score rubric representing a evaluation criteria, and two generated feedbacks are given.
1. Write a detailed analysis that compare qualities of two feedbacks strictly based on the given score rubric (meta-feedback), not evaluating in general.
2. Each feedback contains the [RESULT] to provide their scores for response, ranging from 1 to 5 (5 is perfect and 1 is very bad).
3. After writing an analysis (meta-feedback), write a preference label indicates which feedback is better. You should refer to the score rubric.
4. The output format should look as follows: \"Meta-Feedback: (write an analysis for two feedbacks) [LABEL] (a label A or B of two feedbacks)\"
5. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{orig_instruction}

###Response to evaluate:
{orig_response}

###Reference Answer (Score 5):
{orig_reference_answer}

###Score Rubrics:
[{orig_criteria}]
Score 1: {orig_score1_description}
Score 2: {orig_score2_description}
Score 3: {orig_score3_description}
Score 4: {orig_score4_description}
Score 5: {orig_score5_description}

###Feedbacks to evaluate: 
---
A: {feedback_a}
---
B: {feedback_b}
---

###Meta-Feedbacks: 
'''
```

Note that it is just the initial phase of our work. In the future, we plan to build upon this foundation by annotating a high-quality preference-based critique test set, and  conduct comprehensive tests on existing critique-tuned open-source models.

#### Road Map

- [ ] Collecting more preference-based feedback samples from existing critique-tuned corpus
- [x] Annotate High-quality Test Set
- [ ] Examine the existing reward models and LLMs on our annotated feedback preference corpus
- [ ] Training Llama2-7B Model with SFT and DPO
- [ ] Reward Model for Feedback Preference

#### Dataset Link

Our preference feedback dataset could be found in huggingface dataset: [FeedbackPreference](https://huggingface.co/datasets/GMFTBY/FeedbackPreference)
You can also find it in `data/processed_feedback_preference.json`.

## Citation 

Please cite our paper if Shepherd contributes in your work:

```bibtex 
@misc{Tian_Feedback_Preference_2023,
author = {Tian, Lan},
month = dec,
title = {{Feedback Preference}},
url = {https://github.com/gmftbyGMFTBY/FeedbackPreference},
year = {2023}
}
```
