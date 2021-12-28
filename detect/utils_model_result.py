def post_process_result(model, result, selected_category='None', threshold=0.5):
    output_list = []
    for category_id, category_name in enumerate(model.CLASSES):
        if model.cfg.model.type in {'MaskRCNN', 'QueryInst'}:
            for i, (x1, y1, x2, y2, score) in enumerate(result[0][category_id].tolist()):
                if score < threshold:
                    continue
                if selected_category != 'None':
                    if category_name != selected_category:
                        continue
                mask = result[1][category_id][i]
                data = {
                    'category_id': category_id,
                    'category_name': category_name,
                    'bbox': [min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)],
                    'area': (y2 - y1) * (x2 - x1),
                    'score': score,
                    'mask': mask.tolist()
                }
                output_list.append(data)
        else:
            for x1, y1, x2, y2, score in result[category_id].tolist():
                if score < threshold:
                    continue
                data = {
                    'category_id': category_id,
                    'category_name': category_name,
                    'bbox': [min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)],
                    'area': (y2 - y1) * (x2 - x1),
                    'score': score,
                }
                output_list.append(data)
    return output_list
