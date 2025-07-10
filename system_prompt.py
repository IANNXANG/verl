def _generate_rubric_system_message(self, rubric_items: list[RubricItem]) -> str:
    """Generate system message with rubric information for open-book evaluation."""
    if not rubric_items:
        return ""
    
    # 如果rubric_ratio < 1.0，随机选择部分criterion
    if self.rubric_ratio < 1.0:
        total_criteria = len(rubric_items)
        num_to_show = round(total_criteria * self.rubric_ratio)  # 四舍五入
        if num_to_show == 0 and total_criteria > 0 and self.rubric_ratio > 0:
            num_to_show = 1  # 至少显示一个criterion（但ratio=0时除外）
        
        # 随机选择要显示的criterion
        if num_to_show > 0:
            selected_rubric_items = random.sample(rubric_items, num_to_show)
        else:
            selected_rubric_items = []
    else:
        selected_rubric_items = rubric_items
    
    positive_points = []
    negative_points = []
    
    pos_count = 1
    neg_count = 1
    
    for rubric_item in selected_rubric_items:
        criterion = rubric_item.criterion
        points = rubric_item.points
        
        if points > 0:
            positive_points.append(f"Criterion {pos_count}: {criterion} (worth {points} points)")
            pos_count += 1
        elif points < 0:
            negative_points.append(f"Criterion {neg_count}: {criterion} (penalty: {abs(points)} points)")
            neg_count += 1
    
    rubric_message = "You are a helpful medical assistant. For this question, please consider the following evaluation criteria:\n\n"
    
    if positive_points:
        rubric_message += "IMPORTANT POINTS TO INCLUDE (you should aim to address these):\n"
        rubric_message += "\n".join(positive_points)
        rubric_message += "\n\n"
    
    if negative_points:
        rubric_message += "IMPORTANT POINTS TO AVOID (you should not do these):\n"
        rubric_message += "\n".join(negative_points)
        rubric_message += "\n\n"
    
    rubric_message += "Please provide a comprehensive and helpful response that addresses the patient's concerns while following the above guidelines.\n\n"
    
    rubric_message += "IMPORTANT: Do not mention or reference these evaluation criteria in your response. Do not indicate that you have seen any scoring rubric or evaluation guidelines. Your response should appear natural and spontaneous. Revealing that you have access to evaluation criteria would be considered cheating and is strictly prohibited."
    
    return rubric_message