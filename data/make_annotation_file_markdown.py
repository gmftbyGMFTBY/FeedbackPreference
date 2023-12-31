import json
import csv
import ipdb


'''convert json to csv file format'''


if __name__ == "__main__":
    with open('test.json') as f:
        data = json.load(f)

    task_description = '''An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, a score rubric representing a evaluation criteria, and two generated feedbacks are given.
1. Write a detailed analysis that compare qualities of two feedbacks strictly based on the given score rubric (meta-feedback), not evaluating in general.
2. Each feedback contains the [RESULT] to provide their scores for response, ranging from 1 to 5 (5 is perfect and 1 is very bad).
3. After writing an analysis (meta-feedback), write a preference label indicates which feedback is better. You should refer to the score rubric.
4. The output format should look as follows: \"Meta-Feedback: (write an analysis for two feedbacks) [LABEL] (a label A or B of two feedbacks)\"
5. Please do not generate any other opening, closing, and explanations.'''
    with open('test.md', 'w', encoding='utf-8') as f:
        f.write('| ' + ' | '.join(['ID', '任务设定', '用户指令', 'Chat模型生成回复', '参考回复（5分）', '打分标准', '待评估反思A', '待评估反思B', 'GPT-4针对两个反思的分析', 'GPT-4的偏好标签']) + ' |\n')
        f.write('|--|--|--|--|--|--|--|--|--|--|\n')
        for index, sample in enumerate(data):
            rubrics = '''[{orig_criteria}]
            Score 1: {orig_score1_description}
            Score 2: {orig_score2_description}
            Score 3: {orig_score3_description}
            Score 4: {orig_score4_description}
            Score 5: {orig_score5_description}'''.format(
                orig_criteria=sample['orig_criteria'],
                orig_score1_description=sample['orig_score1_description'],
                orig_score2_description=sample['orig_score2_description'],
                orig_score3_description=sample['orig_score3_description'],
                orig_score4_description=sample['orig_score4_description'],
                orig_score5_description=sample['orig_score5_description'],
            )
            f.write('| ' + ' | '.join([
                str(index),
                '<div style="width: 300pt">' + task_description.replace('\n', '<br>') + '</div>',
                '<div style="width: 300pt">' + sample['orig_instruction'].replace('\n', '<br>') + '</div>',
                '<div style="width: 300pt">' + sample['orig_response'].replace('\n', '<br>') + '</div>',
                '<div style="width: 400pt">' + sample['orig_reference_answer'].replace('\n', '<br>') + '</div>',
                '<div style="width: 300pt">' + rubrics.replace('\n', '<br>') + '</div>',
                '<div style="width: 300pt">' + sample['meta_feedback_generation']['feedback_a'].replace('\n', '<br>') + '</div>',
                '<div style="width: 300pt">' + sample['meta_feedback_generation']['feedback_b'].replace('\n', '<br>') + '</div>',
                '<div style="width: 300pt">' + sample['meta_feedback_generation']['orig_output'].replace('\n', '<br>') + '</div>',
                '<div style="width: 50pt"> <ul><li>[ ] A</li><li>[ ] B</li></ul> </div>'
            ]) + ' |\n')
