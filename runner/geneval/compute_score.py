import json
import re
from collections import defaultdict
from typing import Dict, List, Any


def normalize_answer(answer: Any) -> str:
    """
    规范化答案，处理大小写、标点符号等
    """
    # 转换为字符串
    if isinstance(answer, (int, float)):
        answer = str(int(answer))
    else:
        answer = str(answer)
    
    # 转换为小写
    answer = answer.lower().strip()
    
    # 移除常见的标点符号（句号、逗号等）
    answer = re.sub(r'[.,;:!?]+', '', answer)
    
    return answer


def compare_answers(correct: Any, predicted: str) -> bool:
    """
    比较正确答案和预测答案，考虑大小写和标点符号
    """
    correct_norm = normalize_answer(correct)
    predicted_norm = normalize_answer(predicted)
    return correct_norm == predicted_norm


def compute_score(
    results_file: str = "asset/geneval/results.jsonl",
    correct_answers_file: str = "evaluation/generation/geneval/correct_answers.jsonl"
) -> Dict[str, Any]:
    """
    计算得分，比较 results.jsonl 中的答案与正确答案
    
    Args:
        results_file: 结果文件路径
        correct_answers_file: 正确答案文件路径
    
    Returns:
        包含总体得分和按类型分类得分的字典
    """
    # 读取正确答案，按顺序存储（因为对于同一个index可能有多个条目）
    # 使用列表存储，保持顺序
    correct_answers_list = []
    with open(correct_answers_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                correct_answers_list.append({
                    'index': data['index'],
                    'correct_answer': data['correct_answer']
                })
    
    # 读取结果文件，也按顺序存储
    results_list = []
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                results_list.append(data)
    
    # 由于两个文件的结构应该是对应的，我们按顺序匹配
    # 但为了更安全，我们使用index和出现顺序来匹配
    # 为每个index维护一个计数器
    index_counters = defaultdict(int)
    correct_answers_by_index = defaultdict(list)
    
    # 将正确答案按index分组
    for item in correct_answers_list:
        correct_answers_by_index[item['index']].append(item['correct_answer'])
    
    # 读取结果并计算得分
    total_correct = 0
    total_count = 0
    
    # 按类型统计
    type_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for data in results_list:
        index = data['index']
        result_type = data['type']
        answers = data['answers']
        
        # 获取该index对应的正确答案
        if index not in correct_answers_by_index:
            print(f"Warning: index {index} not found in correct_answers")
            continue
        
        correct_answers_for_index = correct_answers_by_index[index]
        counter = index_counters[index]
        
        # 如果该index有多个正确答案，按顺序取
        if counter < len(correct_answers_for_index):
            correct_answer = correct_answers_for_index[counter]
            index_counters[index] += 1
        else:
            # 如果超出范围，使用最后一个（处理可能的数量不匹配）
            correct_answer = correct_answers_for_index[-1] if correct_answers_for_index else None
        
        if correct_answer is None:
            continue
        
        # 比较4个答案
        for answer in answers:
            total_count += 1
            type_stats[result_type]['total'] += 1
            
            if compare_answers(correct_answer, answer):
                total_correct += 1
                type_stats[result_type]['correct'] += 1
    
    # 计算总体得分
    overall_score = total_correct / total_count if total_count > 0 else 0.0
    
    # 计算各类型得分
    type_scores = {}
    for result_type, stats in type_stats.items():
        if stats['total'] > 0:
            type_scores[result_type] = {
                'score': stats['correct'] / stats['total'],
                'correct': stats['correct'],
                'total': stats['total']
            }
        else:
            type_scores[result_type] = {
                'score': 0.0,
                'correct': 0,
                'total': 0
            }
    
    return {
        'overall_score': overall_score,
        'overall_correct': total_correct,
        'overall_total': total_count,
        'type_scores': type_scores
    }


def print_scores(scores: Dict[str, Any]):
    """
    打印得分结果
    """
    print("=" * 60)
    print("总体得分:")
    print(f"  正确数: {scores['overall_correct']}/{scores['overall_total']}")
    print(f"  准确率: {scores['overall_score']:.4f} ({scores['overall_score']*100:.2f}%)")
    print()
    
    print("按类型分类得分:")
    for result_type, type_data in sorted(scores['type_scores'].items()):
        print(f"  {result_type}:")
        print(f"    正确数: {type_data['correct']}/{type_data['total']}")
        print(f"    准确率: {type_data['score']:.4f} ({type_data['score']*100:.2f}%)")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, required=True)
    args = parser.parse_args()
    
    scores = compute_score(results_file=args.results_file)
    print_scores(scores)

