import pandas as pd
import numpy as np
import math
import json
import io
import sys
from pathlib import Path
from itertools import combinations
from collections import Counter
from flask import Flask, render_template, request, Response
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy numeric types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def jsonify(data=None, **kwargs):
    """Flask-compatible jsonify that handles numpy types."""
    if data is not None:
        payload = data
        payload.update(kwargs)
    elif kwargs:
        payload = kwargs
    else:
        payload = {}
    return Response(
        json.dumps(payload, cls=NumpyEncoder, ensure_ascii=False),
        mimetype='application/json'
    )

app = Flask(__name__)

# Load dữ liệu
data_path = Path('data/play_tennis.csv')
df = pd.read_csv(data_path)
target_label = 'Play'

feature_cols = [c for c in df.columns if c != target_label]

# ==================== Helper functions ====================
def df_to_table(df_data, title=None, caption=None):
    """Convert DataFrame to structured table dict"""
    return {
        'title': title or '',
        'caption': caption or '',
        'headers': df_data.columns.tolist(),
        'rows': df_data.values.tolist(),
        'dtypes': [str(d) for d in df_data.dtypes]
    }

def list_to_table(data, headers, title=None, caption=None):
    """Convert list of lists to structured table dict"""
    return {
        'title': title or '',
        'caption': caption or '',
        'headers': headers,
        'rows': data,
        'dtypes': ['object'] * len(headers)
    }

def make_cell_result(text_lines=None, tables=None, tree_data=None, extra=None):
    """Create a structured cell result"""
    result = {}
    if text_lines:
        result['text'] = '\n'.join(text_lines)
    if tables:
        result['tables'] = tables
    if tree_data:
        result['tree'] = tree_data
    if extra:
        result['extra'] = extra
    return result


# ==================== CELL 1: Tương quan Pearson ====================
def cell1_correlation(params=None):
    if params is None:
        params = {}
    col_chosen = params.get('col_chosen', 'Outlook')
    
    df_subset = df[[target_label, col_chosen]].copy()
    
    le_target = LabelEncoder()
    le_feature = LabelEncoder()
    df_subset[target_label] = le_target.fit_transform(df_subset[target_label])
    df_subset[col_chosen] = le_feature.fit_transform(df_subset[col_chosen])
    df_subset = df_subset.dropna()
    
    x = df_subset[col_chosen].values
    y = df_subset[target_label].values
    
    # Encode mapping tables
    feature_mapping = list(le_feature.classes_)
    target_mapping = list(le_target.classes_)
    
    # Data after encode table
    encode_data = [[i+1, x[i], y[i]] for i in range(len(x))]
    
    def pearson_correlation(x, y):
        if len(x) != len(y):
            raise ValueError("Độ dài của x và y không bằng nhau!")
        n = len(x)
        if n == 0:
            return 0
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        tu_so = sum((x_i - mean_x) * (y_i - mean_y) for x_i, y_i in zip(x, y))
        sum_sq_x = sum((x_i - mean_x)**2 for x_i in x)
        sum_sq_y = sum((y_i - mean_y)**2 for y_i in y)
        mau_so = math.sqrt(sum_sq_x * sum_sq_y)
        if mau_so == 0:
            return 0
        return tu_so / mau_so
    
    r = pearson_correlation(x, y)
    
    # Kết luận
    abs_r = abs(r)
    if abs_r == 1.0:
        conclusion = "Quan hệ tuyến tính hoàn hảo."
    elif 0.7 <= abs_r < 1.0:
        conclusion = "Quan hệ tuyến tính chặt chẽ."
    elif 0.3 <= abs_r < 0.7:
        conclusion = "Quan hệ tuyến tính trung bình."
    else:
        conclusion = "Rất ít hoặc không có quan hệ tuyến tính."
    
    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))
    tu_so = sum((x_i - mean_x) * (y_i - mean_y) for x_i, y_i in zip(x, y))
    sum_sq_x = sum((x_i - mean_x)**2 for x_i in x)
    sum_sq_y = sum((y_i - mean_y)**2 for y_i in y)
    
    return make_cell_result(
        text_lines=[
            f"Phân tích tương quan Pearson giữa '{col_chosen}' và '{target_label}'",
            "",
            f"Hệ số tương quan Pearson: r = {r:.4f}",
            f"Kết luận: {conclusion}"
        ],
        tables=[
            list_to_table(
                [[str(c), str(v)] for c, v in zip(le_feature.classes_, le_feature.transform(le_feature.classes_))],
                [f'{col_chosen} (gốc)', f'{col_chosen} (mã)'],
                title=f"Bảng mã hóa '{col_chosen}'"
            ),
            list_to_table(
                [[str(c), str(v)] for c, v in zip(le_target.classes_, le_target.transform(le_target.classes_))],
                [f'{target_label} (gốc)', f'{target_label} (mã)'],
                title=f"Bảng mã hóa '{target_label}'"
            ),
            list_to_table(
                encode_data,
                ['Mẫu', f'{col_chosen} (x)', f'{target_label} (y)'],
                title="Dữ liệu sau khi mã hóa"
            ),
        ],
        extra={
            'r': round(r, 6),
            'r_abs': round(abs_r, 6),
            'conclusion': conclusion,
            'col_chosen': col_chosen,
            'n': len(x),
            'mean_x': round(mean_x, 4),
            'mean_y': round(mean_y, 4),
            'tu_so': round(tu_so, 4),
            'sum_sq_x': round(sum_sq_x, 4),
            'sum_sq_y': round(sum_sq_y, 4),
            'mau_so': round(math.sqrt(sum_sq_x * sum_sq_y), 4),
        }
    )


# ==================== CELL 2: Apriori & Association Rules ====================
def cell2_apriori(params=None):
    if params is None:
        params = {}
    min_sup = float(params.get('min_sup', 0.2))
    min_conf = float(params.get('min_conf', 0.7))
    
    trans = [set(f"{c}_{v}" for c, v in row.items()) for _, row in df.iterrows()]
    
    min_sup_count = max(1, int(np.ceil(min_sup * len(trans))))
    
    def generate_candidates(freq_list, k):
        candidates = set()
        for i in range(len(freq_list)):
            for j in range(i + 1, len(freq_list)):
                candidate = freq_list[i] | freq_list[j]
                if len(candidate) == k:
                    candidates.add(candidate)
        return candidates
    
    all_frequent_itemsets = {}
    C1 = {}
    for t in trans:
        for item in t:
            item_set = frozenset([item])
            C1[item_set] = C1.get(item_set, 0) + 1
    
    F1 = {k: v for k, v in C1.items() if v >= min_sup_count}
    all_frequent_itemsets[1] = F1
    
    k = 2
    previous_frequent = list(F1.keys())
    while len(previous_frequent) >= k:
        Ck = generate_candidates(previous_frequent, k)
        if not Ck:
            break
        Fk = {}
        for c in Ck:
            cnt = sum(1 for t in trans if c <= t)
            if cnt >= min_sup_count:
                Fk[c] = cnt
        if not Fk:
            break
        all_frequent_itemsets[k] = Fk
        previous_frequent = list(Fk.keys())
        k += 1
    
    # Tổng hợp all itemsets
    all_itemsets = []
    for k_level, itemsets_dict in sorted(all_frequent_itemsets.items()):
        for itemset, sup in itemsets_dict.items():
            all_itemsets.append((itemset, sup, sup / len(trans)))
    all_itemsets.sort(key=lambda x: (len(x[0]), x[1]), reverse=True)
    
    # Table: Frequent Itemsets
    itemsets_table_rows = []
    for idx, (itemset, sup, sup_ratio) in enumerate(all_itemsets, 1):
        items_str = ", ".join(sorted(itemset))
        itemsets_table_rows.append([idx, len(itemset), items_str, sup, f"{sup}/{len(trans)}", f"{sup_ratio*100:.1f}%"])
    
    # Maximal itemsets
    maximal_itemsets = []
    for itemset, sup, sup_ratio in all_itemsets:
        is_maximal = True
        for other_itemset, _, _ in all_itemsets:
            if itemset < other_itemset:
                is_maximal = False
                break
        if is_maximal:
            maximal_itemsets.append((itemset, sup, sup_ratio))
    
    maximal_table_rows = []
    for idx, (itemset, sup, sup_ratio) in enumerate(maximal_itemsets, 1):
        items_str = ", ".join(sorted(itemset))
        maximal_table_rows.append([idx, len(itemset), items_str, sup, f"{sup/len(trans)*100:.1f}%"])
    
    # Sinh luật kết hợp
    F2_and_above = {}
    for k_level, itemsets_dict in sorted(all_frequent_itemsets.items()):
        if k_level >= 2:
            for itemset, sup in itemsets_dict.items():
                F2_and_above[itemset] = sup
    
    rules = []
    for itemset, sup_xy in F2_and_above.items():
        items_list = list(itemset)
        for X_item in items_list:
            for Y_item in items_list:
                if X_item != Y_item:
                    sup_x = F1.get(frozenset([X_item]), 0)
                    if sup_x > 0:
                        conf = sup_xy / sup_x
                        if conf >= min_conf:
                            lift = (sup_xy / len(trans)) / ((sup_x / len(trans)) * (F1.get(frozenset([Y_item]), 0) / len(trans)))
                            rules.append({
                                'X': X_item,
                                'Y': Y_item,
                                'conf': conf,
                                'sup_xy': sup_xy,
                                'sup_x': sup_x,
                                'sup_ratio': sup_xy / len(trans),
                                'lift': lift
                            })
    
    rules.sort(key=lambda x: (x['conf'], x['sup_xy']), reverse=True)
    
    rules_table_rows = []
    for idx, rule in enumerate(rules, 1):
        rules_table_rows.append([
            idx, rule['X'], rule['Y'],
            f"{rule['conf']:.4f}",
            f"{rule['sup_ratio']*100:.1f}%",
            f"{rule['lift']:.4f}"
        ])
    
    # Filter rules with Y = target
    play_rules = [r for r in rules if r['Y'].startswith(f'{target_label}_')]
    play_rules_rows = []
    for idx, rule in enumerate(play_rules, 1):
        play_rules_rows.append([
            idx, rule['X'], rule['Y'],
            f"{rule['conf']:.4f}",
            f"{rule['sup_ratio']*100:.1f}%",
            f"{rule['lift']:.4f}"
        ])
    
    # Transaction table
    trans_rows = []
    for i, t in enumerate(trans, 1):
        trans_rows.append([i, len(t), ', '.join(sorted(t))])
    
    # F1 table
    f1_rows = []
    for itemset, sup in sorted(F1.items(), key=lambda x: x[1], reverse=True):
        items_str = ''.join(itemset)
        f1_rows.append([items_str, sup, f"{sup/len(trans)*100:.1f}%"])
    
    tables = [
        list_to_table(trans_rows, ['Giao tác', 'Số phần tử', 'Các phần tử'], title="Các giao tác"),
        list_to_table(f1_rows, ['Itemset', 'Support', 'Tỉ lệ'], title=f"F1 - Itemsets phổ biến (min_sup={min_sup})"),
    ]
    
    if itemsets_table_rows:
        tables.append(
            list_to_table(itemsets_table_rows, ['STT', 'Kích thước', 'Itemset', 'Support', 'Support (p/s)', 'Tỉ lệ'], title="Tất cả tập phổ biến")
        )
    
    if maximal_table_rows:
        tables.append(
            list_to_table(maximal_table_rows, ['STT', 'Kích thước', 'Itemset', 'Support', 'Tỉ lệ'], title="Tập phổ biến tối đa")
        )
    
    if rules_table_rows:
        tables.append(
            list_to_table(rules_table_rows, ['STT', 'X', 'Y', 'Confidence', 'Support', 'Lift'], title=f"Luật kết hợp (min_conf={min_conf})")
        )
    
    if play_rules_rows:
        tables.append(
            list_to_table(play_rules_rows, ['STT', 'X', 'Y', 'Confidence', 'Support', 'Lift'], title=f"Luật với Y = '{target_label}'")
        )
    
    text_lines = [
        f"Phân tích Apriori trên Play Tennis",
        f"min_sup = {min_sup} ({min_sup_count}/{len(trans)} giao tác)",
        f"min_conf = {min_conf}",
        f"Tổng số giao tác: {len(trans)}",
        f"Tổng số itemsets phổ biến: {len(all_itemsets)}",
        f"Tổng số itemsets tối đa: {len(maximal_itemsets)}",
        f"Tổng số luật: {len(rules)}",
        f"Số luật với Y='{target_label}': {len(play_rules)}"
    ]
    
    return make_cell_result(
        text_lines=text_lines,
        tables=tables,
        extra={
            'min_sup': min_sup,
            'min_sup_count': min_sup_count,
            'min_conf': min_conf,
            'n_trans': len(trans),
            'n_itemsets': len(all_itemsets),
            'n_maximal': len(maximal_itemsets),
            'n_rules': len(rules),
            'n_play_rules': len(play_rules)
        }
    )


# ==================== CELL 3: Rough Set ====================
def cell3_roughset(params=None):
    X = df.drop(target_label, axis=1)
    y = df[target_label]
    attributes = list(X.columns)
    
    attr_symbols = {attr: f"A{i+1}" for i, attr in enumerate(attributes)}
    symbol_to_attr = {v: k for k, v in attr_symbols.items()}
    
    n = len(df)
    conditions = []
    
    for i in range(n):
        for j in range(n):
            if i < j and y.iloc[i] != y.iloc[j]:
                diff_attrs = {attr_symbols[attr] for attr in attributes if X.iloc[i][attr] != X.iloc[j][attr]}
                if diff_attrs:
                    conditions.append(diff_attrs)
    
    unique_conditions = [list(c) for c in set(tuple(sorted(c)) for c in conditions if c)]
    
    sym_list = list(attr_symbols.values())
    
    def find_reducts(conditions, attributes_syms):
        valid_reducts = []
        for k in range(1, len(attributes_syms) + 1):
            for subset in combinations(attributes_syms, k):
                subset_set = set(subset)
                is_valid = True
                for cond in conditions:
                    if not set(cond).intersection(subset_set):
                        is_valid = False
                        break
                if is_valid:
                    valid_reducts.append(list(subset))
            if valid_reducts:
                break
        return valid_reducts
    
    reducts = find_reducts(unique_conditions, sym_list)
    
    # Conditions table
    cond_rows = []
    for idx, cond in enumerate(unique_conditions, 1):
        cond_str = " ∨ ".join(cond)
        cond_rows.append([idx, cond_str])
    
    # Reducts table
    reduct_rows = []
    if reducts:
        for i, red in enumerate(reducts):
            red_sym_str = ', '.join(red)
            red_attrs = [symbol_to_attr[s] for s in red]
            red_attr_str = ', '.join(red_attrs)
            reduct_rows.append([i+1, f"{{{red_sym_str}}}", red_attr_str])
        chosen_red_sym = reducts[0]
        chosen_red_attr = [symbol_to_attr[s] for s in chosen_red_sym]
    else:
        chosen_red_attr = attributes
    
    # Sinh luật
    def generate_rules(df_full, reduct_attrs, target_col):
        grouped = df_full.groupby(reduct_attrs)
        rules = []
        rule_count = 1
        for name, group in grouped:
            support = len(group)
            if support < 1:
                continue
            majority_decision = group[target_col].mode()[0]
            majority_count = len(group[group[target_col] == majority_decision])
            accuracy = majority_count / support
            if type(name) not in (list, tuple):
                name = (name,)
            condition = " AND ".join([f"{attr}={val}" for attr, val in zip(reduct_attrs, name)])
            rules.append({
                'id': rule_count,
                'condition': condition,
                'decision': majority_decision,
                'accuracy': accuracy * 100,
                'support': support
            })
            rule_count += 1
        return sorted(rules, key=lambda x: (x['accuracy'], x['support']), reverse=True)
    
    rules = generate_rules(df, chosen_red_attr, target_label)
    
    rules_rows = []
    for r in rules:
        rules_rows.append([r['id'], r['condition'], r['decision'], f"{r['accuracy']:.1f}%", r['support']])
    
    text_lines = [
        "Phân tích Rough Set trên Play Tennis",
        f"Thuộc tính điều kiện: {attributes}",
        f"Thuộc tính quyết định: {target_label}",
        f"Số điều kiện phân biệt: {len(unique_conditions)}",
        f"Các reduct tìm được: {len(reducts)}" + (f" -> Chọn: {chosen_red_attr}" if reducts else " -> Không tìm thấy reduct, dùng tất cả attributes"),
        f"Tổng số luật: {len(rules)}"
    ]
    
    tables = [
        list_to_table(cond_rows, ['STT', 'Điều kiện'], title="Ma trận điều kiện phân biệt"),
    ]
    
    if reduct_rows:
        tables.append(
            list_to_table(reduct_rows, ['STT', 'Ký hiệu', 'Thuộc tính'], title="Reducts tìm được")
        )
    
    tables.append(
        list_to_table(rules_rows, ['STT', 'Điều kiện (IF)', 'Quyết định (THEN)', 'Độ chính xác', 'Số mẫu'], title="Luật Rough Set")
    )
    
    return make_cell_result(
        text_lines=text_lines,
        tables=tables,
        extra={
            'n_conditions': len(unique_conditions),
            'n_reducts': len(reducts),
            'chosen_reduct': chosen_red_attr,
            'n_rules': len(rules)
        }
    )


# ==================== CELL 4: Naive Bayes ====================
def cell4_naivebayes(params=None):
    if params is None:
        params = {}
    with_laplace = params.get('laplace', True)
    
    X_raw = df.drop(target_label, axis=1).copy()
    y_raw = df[target_label].copy()
    
    X_encoded = X_raw.copy()
    encoders = {}
    for col in X_encoded.columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        encoders[col] = le
    
    X_vals = X_encoded.values
    y_vals = y_raw.values
    
    np.random.seed(42)
    indices = np.random.permutation(len(X_vals))
    split_idx = int(len(indices) * 0.7)
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    X_train, X_test = X_vals[train_idx], X_vals[test_idx]
    y_train, y_test = y_vals[train_idx], y_vals[test_idx]
    
    classes = np.unique(y_raw)
    
    class NaiveBayesClassifier:
        def __init__(self, smoothing=True):
            self.smoothing = smoothing
            self.alpha = 1.0 if smoothing else 0.0
            self.classes = []
            self.class_probs = {}
            self.feature_probs = {}
        
        def fit(self, X, y):
            self.classes = np.unique(y)
            n_samples, n_features = X.shape
            for c in self.classes:
                self.class_probs[c] = float(np.sum(y == c) / n_samples)
            self.feature_probs = {c: {i: {} for i in range(n_features)} for c in self.classes}
            for i in range(n_features):
                unique_values = np.unique(X[:, i])
                V = len(unique_values)
                for c in self.classes:
                    X_c = X[y == c]
                    n_c = len(X_c)
                    values, counts = np.unique(X_c[:, i], return_counts=True)
                    val_count_dict = dict(zip(values, counts))
                    for val in unique_values:
                        count_val_c = val_count_dict.get(val, 0)
                        prob = (count_val_c + self.alpha) / (n_c + self.alpha * V)
                        # KHÔNG làm trơn: để prob = 0 thật (sẽ thành -inf khi log)
                        # Có làm trơn: prob luôn > 0 nhờ alpha
                        self.feature_probs[c][i][val] = prob
        
        def predict(self, X):
            predictions = []
            for x in X:
                posteriors = {}
                for c in self.classes:
                    prior = np.log(self.class_probs[c])
                    conditional = 0
                    has_zero_prob = False
                    for i, val in enumerate(x):
                        prob = self.feature_probs[c][i].get(val, 0.0)
                        if prob == 0.0:
                            has_zero_prob = True
                        else:
                            conditional += np.log(prob)
                    if has_zero_prob:
                        posteriors[c] = -np.inf  # Lớp này không thể xảy ra
                    else:
                        posteriors[c] = prior + conditional
                # Loại bỏ các lớp có -inf trước khi chọn max
                valid_classes = {c: v for c, v in posteriors.items() if v != -np.inf}
                if valid_classes:
                    best_class = max(valid_classes, key=valid_classes.get)
                else:
                    # Tất cả đều -inf, chọn theo prior
                    best_class = max(posteriors, key=lambda c: np.log(self.class_probs[c]))
                predictions.append(best_class)
            return np.array(predictions)
    
    nb = NaiveBayesClassifier(smoothing=with_laplace)
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    
    # Confusion matrix
    def calculate_confusion_matrix(y_true, y_pred, classes):
        matrix = np.zeros((len(classes), len(classes)), dtype=int)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        for t, p in zip(y_true, y_pred):
            if t in class_to_idx and p in class_to_idx:
                matrix[class_to_idx[t], class_to_idx[p]] += 1
        return matrix
    
    cm = calculate_confusion_matrix(y_test, y_pred, classes)
    accuracy = float(np.sum(y_test == y_pred) / len(y_test))
    
    # Calculate precision, recall, F1-score for each class
    n_classes = len(classes)
    metrics_rows = []
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    
    for i, c in enumerate(classes):
        tp = int(cm[i, i])
        fp = int(np.sum(cm[:, i]) - cm[i, i])
        fn = int(np.sum(cm[i, :]) - cm[i, i])
        tn = int(np.sum(cm) - tp - fp - fn)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1
        
        metrics_rows.append([
            str(c),
            tp, fp, fn, tn,
            f"{precision:.4f}",
            f"{recall:.4f}",
            f"{f1:.4f}"
        ])
    
    macro_precision /= n_classes
    macro_recall /= n_classes
    macro_f1 /= n_classes
    
    # Confusion matrix table (enhanced with TP/FP/FN/TN per row)
    cm_headers = [f"Thực \\ Dự đoán"] + [str(c) for c in classes]
    cm_rows = []
    for i, true_class in enumerate(classes):
        row_data = [str(true_class)]
        for j in range(len(classes)):
            row_data.append(int(cm[i, j]))
        cm_rows.append(row_data)
    
    # Add total row to confusion matrix
    total_row = ['Tổng']
    for j in range(n_classes):
        total_row.append(int(np.sum(cm[:, j])))
    cm_rows.append(total_row)
    
    # Encode tables
    encode_rows = []
    for col, le in encoders.items():
        for orig, enc in zip(le.classes_, le.transform(le.classes_)):
            encode_rows.append([col, orig, enc])
    
    # Prior probabilities
    prior_rows = []
    for c in classes:
        prior_rows.append([str(c), f"{nb.class_probs[c]:.4f}"])
    
    # Posterior probabilities for each test sample
    posterior_rows = []
    for sample_idx in range(len(X_test)):
        x = X_test[sample_idx]
        y_true_val = str(y_test[sample_idx])
        y_pred_val = str(y_pred[sample_idx])
        
        # Compute posteriors
        posteriors = {}
        for c in nb.classes:
            prior = np.log(nb.class_probs[c])
            conditional = 0
            for i, val in enumerate(x):
                prob = nb.feature_probs[c][i].get(val, 1e-10)
                conditional += np.log(prob)
            posteriors[c] = prior + conditional
        
        posterior_str = "; ".join([f"P({c}|x)={posteriors[c]:.4f}" for c in classes])
        posterior_rows.append([sample_idx + 1, x.tolist(), y_true_val, y_pred_val, posterior_str])
    
    # Distribution table
    dist_rows = [
        ['Train', len(X_train), f"{len(X_train)/len(X_vals)*100:.0f}%"],
        ['Test', len(X_test), f"{len(X_test)/len(X_vals)*100:.0f}%"],
    ]
    
    laplace_label = "CÓ làm trơn Laplace" if with_laplace else "KHÔNG làm trơn Laplace"
    
    text_lines = [
        f"Phân loại Naive Bayes ({laplace_label})",
        f"Tập huấn luyện: {len(X_train)} mẫu ({len(X_train)/len(X_vals)*100:.0f}%)",
        f"Tập kiểm tra: {len(X_test)} mẫu ({len(X_test)/len(X_vals)*100:.0f}%)",
        f"Độ chính xác (Accuracy): {accuracy*100:.2f}%",
        f"Precision (trung bình macro): {macro_precision:.4f}",
        f"Recall (trung bình macro): {macro_recall:.4f}",
        f"F1-Score (trung bình macro): {macro_f1:.4f}"
    ]
    
    tables = [
        list_to_table(encode_rows, ['Thuộc tính', 'Giá trị gốc', 'Mã'], title="Bảng mã hóa thuộc tính"),
        list_to_table(dist_rows, ['Tập', 'Số mẫu', 'Tỉ lệ'], title="Phân phối dữ liệu"),
        list_to_table(prior_rows, ['Lớp', 'P(Lớp)'], title="Xác suất tiên nghiệm"),
        list_to_table(cm_rows, cm_headers, title="Ma trận nhầm lẫn (Confusion Matrix)"),
        list_to_table(
            metrics_rows,
            ['Lớp', 'TP', 'FP', 'FN', 'TN', 'Precision', 'Recall', 'F1-Score'],
            title="Chi tiết đánh giá theo từng lớp"
        ),
        list_to_table(posterior_rows, ['Mẫu', 'x', 'y thực', 'y dự đoán', 'Posterior'], title="Chi tiết dự đoán"),
    ]
    
    return make_cell_result(
        text_lines=text_lines,
        tables=tables,
        extra={
            'accuracy': round(accuracy * 100, 2),
            'precision': round(macro_precision, 4),
            'recall': round(macro_recall, 4),
            'f1_score': round(macro_f1, 4),
            'laplace': with_laplace,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'classes': [str(c) for c in classes],
            'cm': cm.tolist()
        }
    )


# ==================== CELL 5: ID3 Decision Tree ====================
def cell5_id3(params=None):
    df_work = df.copy()
    cols = [c for c in df.columns if c != target_label]
    
    def entropy(series):
        total = len(series)
        if total == 0:
            return 0.0
        counts = Counter(series)
        e = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                e -= p * math.log2(p)
        return e
    
    def gini(series):
        total = len(series)
        if total == 0:
            return 0.0
        counts = Counter(series)
        g = 1.0
        for count in counts.values():
            p = count / total
            g -= p ** 2
        return g
    
    def information_gain(data, attr, decision_attr):
        info_decision = entropy(data[decision_attr])
        weighted_info = 0.0
        gain_details = []
        for v, subset in data.groupby(attr):
            w = len(subset) / len(data)
            e = entropy(subset[decision_attr])
            weighted_info += w * e
            gain_details.append((v, len(subset), len(data), round(e, 6)))
        gain = info_decision - weighted_info
        return info_decision, weighted_info, gain, gain_details
    
    def gini_gain(data, attr, decision_attr):
        gini_decision = gini(data[decision_attr])
        weighted_gini = 0.0
        gini_details = []
        for v, subset in data.groupby(attr):
            w = len(subset) / len(data)
            g = gini(subset[decision_attr])
            weighted_gini += w * g
            gini_details.append((v, len(subset), len(data), round(g, 6)))
        gain = gini_decision - weighted_gini
        return gini_decision, weighted_gini, gain, gini_details
    
    class Node:
        def __init__(self, attribute=None, label=None):
            self.attribute = attribute
            self.label = label
            self.children = {}
            self.majority_label = None
            self.sample_count = 0
    
    # ---------- ID3 using Information Gain ----------
    def id3_gain(data, attrs, decision_attr, path_text='Gốc', depth=0):
        node = Node()
        node.sample_count = len(data)
        node.majority_label = data[decision_attr].mode()[0]
        
        step = {
            'depth': depth,
            'path': path_text,
            'samples': len(data),
            'action': '',
            'gain_table': None,
            'children': []
        }
        
        unique_labels = data[decision_attr].unique()
        if len(unique_labels) == 1:
            node.label = unique_labels[0]
            step['action'] = f'Tạo lá: {target_label} = {node.label} (thuần nhất)'
            tree_steps_gain.append(step)
            return node
        
        if not attrs:
            node.label = node.majority_label
            step['action'] = f'Hết thuộc tính -> lá: {target_label} = {node.label} (đa số)'
            tree_steps_gain.append(step)
            return node
        
        gan_rows = []
        info_decision = entropy(data[decision_attr])
        for a in attrs:
            _, weighted_info, g, details = information_gain(data, a, decision_attr)
            gan_rows.append({
                'attr': a,
                'I_split': round(weighted_info, 6),
                'gain': round(g, 6),
                'details': details
            })
        gan_rows.sort(key=lambda x: x['gain'], reverse=True)
        
        best_attr = gan_rows[0]['attr']
        best_gain = gan_rows[0]['gain']
        
        # Build gain table for this step
        gt_rows = []
        for gr in gan_rows:
            gt_rows.append([gr['attr'], round(entropy(data[decision_attr]), 6), gr['I_split'], gr['gain']])
        
        step['gain_table'] = {
            'headers': ['Thuộc tính', 'I(Quyết định)', 'I(Sau chia)', 'Gain'],
            'rows': gt_rows
        }
        
        if best_gain <= 1e-12:
            node.label = node.majority_label
            step['action'] = f'Gain max ~ 0 -> lá: {target_label} = {node.label} (đa số)'
            tree_steps_gain.append(step)
            return node
        
        node.attribute = best_attr
        step['action'] = f'Chọn: {best_attr} (Gain = {best_gain:.6f})'
        step['split_attr'] = best_attr
        
        remaining_attrs = [a for a in attrs if a != best_attr]
        for v in sorted(data[best_attr].unique()):
            subset = data[data[best_attr] == v]
            child_step_label = f"{best_attr} = {v}"
            if subset.empty:
                leaf = Node(label=node.majority_label)
                leaf.sample_count = 0
                node.children[v] = leaf
                step['children'].append({
                    'label': child_step_label,
                    'samples': 0,
                    'action': f'Tạo lá (trống): {target_label} = {node.majority_label}'
                })
            else:
                child_path = f"{path_text} -> {best_attr}={v}"
                child_node = id3_gain(subset, remaining_attrs, decision_attr, child_path, depth+1)
                node.children[v] = child_node
                child_label = child_node.label if child_node.label else f"[{child_node.attribute}]"
                step['children'].append({
                    'label': child_step_label,
                    'samples': len(subset),
                    'action': f'{child_label}'
                })
        
        tree_steps_gain.append(step)
        return node
    
    # ---------- ID3 using Gini Index ----------
    def id3_gini(data, attrs, decision_attr, path_text='Gốc', depth=0):
        node = Node()
        node.sample_count = len(data)
        node.majority_label = data[decision_attr].mode()[0]
        
        step = {
            'depth': depth,
            'path': path_text,
            'samples': len(data),
            'action': '',
            'gini_table': None,
            'children': []
        }
        
        unique_labels = data[decision_attr].unique()
        if len(unique_labels) == 1:
            node.label = unique_labels[0]
            step['action'] = f'Tạo lá: {target_label} = {node.label} (thuần nhất)'
            tree_steps_gini.append(step)
            return node
        
        if not attrs:
            node.label = node.majority_label
            step['action'] = f'Hết thuộc tính -> lá: {target_label} = {node.label} (đa số)'
            tree_steps_gini.append(step)
            return node
        
        gini_rows = []
        for a in attrs:
            gini_decision, weighted_gini, g, details = gini_gain(data, a, decision_attr)
            gini_rows.append({
                'attr': a,
                'G_split': round(weighted_gini, 6),
                'gain': round(g, 6),
                'details': details
            })
        gini_rows.sort(key=lambda x: x['gain'], reverse=True)
        
        best_attr = gini_rows[0]['attr']
        best_gain = gini_rows[0]['gain']
        
        # Build gini table for this step
        gt_rows = []
        for gr in gini_rows:
            gt_rows.append([gr['attr'], round(gini(data[decision_attr]), 6), gr['G_split'], gr['gain']])
        
        step['gini_table'] = {
            'headers': ['Thuộc tính', 'Gini(Quyết định)', 'Gini(Sau chia)', 'Giảm Gini'],
            'rows': gt_rows
        }
        
        if best_gain <= 1e-12:
            node.label = node.majority_label
            step['action'] = f'Giảm Gini max ~ 0 -> lá: {target_label} = {node.label} (đa số)'
            tree_steps_gini.append(step)
            return node
        
        node.attribute = best_attr
        step['action'] = f'Chọn: {best_attr} (Giảm Gini = {best_gain:.6f})'
        step['split_attr'] = best_attr
        
        remaining_attrs = [a for a in attrs if a != best_attr]
        for v in sorted(data[best_attr].unique()):
            subset = data[data[best_attr] == v]
            child_step_label = f"{best_attr} = {v}"
            if subset.empty:
                leaf = Node(label=node.majority_label)
                leaf.sample_count = 0
                node.children[v] = leaf
                step['children'].append({
                    'label': child_step_label,
                    'samples': 0,
                    'action': f'Tạo lá (trống): {target_label} = {node.majority_label}'
                })
            else:
                child_path = f"{path_text} -> {best_attr}={v}"
                child_node = id3_gini(subset, remaining_attrs, decision_attr, child_path, depth+1)
                node.children[v] = child_node
                child_label = child_node.label if child_node.label else f"[{child_node.attribute}]"
                step['children'].append({
                    'label': child_step_label,
                    'samples': len(subset),
                    'action': f'{child_label}'
                })
        
        tree_steps_gini.append(step)
        return node
    
    def tree_to_dict(node):
        if node.label is not None:
            return {
                'type': 'leaf',
                'label': node.label,
                'samples': node.sample_count
            }
        children = {}
        for val, child in node.children.items():
            children[str(val)] = tree_to_dict(child)
        return {
            'type': 'split',
            'attribute': node.attribute,
            'samples': node.sample_count,
            'children': children
        }
    
    # Build both trees
    tree_steps_gain = []
    tree_steps_gini = []
    
    root_gain = id3_gain(df_work, cols, target_label)
    tree_gain_dict = tree_to_dict(root_gain)
    
    root_gini = id3_gini(df_work, cols, target_label)
    tree_gini_dict = tree_to_dict(root_gini)
    
    text_lines = [
        "Xây dựng cây quyết định ID3 với Information Gain (Entropy)",
        f"Số mẫu: {len(df_work)}",
        f"Thuộc tính: {cols}",
        f"Thuộc tính quyết định: {target_label}",
        f"Số bước (Gain): {len(tree_steps_gain)}",
        "",
        "=" * 40,
        "",
        "Xây dựng cây quyết định CART với Gini Index",
        f"Số bước (Gini): {len(tree_steps_gini)}"
    ]
    
    # Comparison table for the top-level split
    comparison_rows = []
    gain_root_attr = tree_gain_dict.get('attribute', 'N/A') if tree_gain_dict.get('type') == 'split' else 'N/A'
    gini_root_attr = tree_gini_dict.get('attribute', 'N/A') if tree_gini_dict.get('type') == 'split' else 'N/A'
    comparison_rows.append(['Information Gain (Entropy)', gain_root_attr, str(len(tree_steps_gain))])
    comparison_rows.append(['Gini Index', gini_root_attr, str(len(tree_steps_gini))])
    
    tables = [
        list_to_table(
            comparison_rows,
            ['Phương pháp', 'Thuộc tính gốc', 'Số bước'],
            title="So sánh ID3 (Gain) vs CART (Gini)"
        )
    ]
    
    return make_cell_result(
        text_lines=text_lines,
        tree_data={
            'gain': tree_gain_dict,
            'gini': tree_gini_dict
        },
        tables=tables,
        extra={
            'steps_gain': tree_steps_gain,
            'steps_gini': tree_steps_gini,
            'columns': cols,
            'target': target_label,
            'n_steps_gain': len(tree_steps_gain),
            'n_steps_gini': len(tree_steps_gini),
            'root_attr_gain': gain_root_attr,
            'root_attr_gini': gini_root_attr
        }
    )


# ==================== CELL 6: K-Means ====================
def cell6_kmeans(params=None):
    if params is None:
        params = {}
    k = int(params.get('k', 2))
    
    X_raw = df.drop(target_label, axis=1).copy()
    X_encoded = pd.get_dummies(X_raw, drop_first=False)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    
    n_samples = X_scaled.shape[0]
    sample_labels = [f'Mẫu {i+1}' for i in range(n_samples)]
    cluster_labels = [f'Cụm {j+1}' for j in range(k)]
    
    # ====== K-Means with verbose iterative output ======
    # Initialize: assign first n/k samples to each cluster
    np.random.seed(42)
    init_assignments = np.random.randint(0, k, n_samples)
    # Ensure at least one sample per cluster
    for j in range(k):
        if np.sum(init_assignments == j) == 0:
            init_assignments[j % n_samples] = j
    
    assignments = init_assignments.copy()
    
    # Build text lines (like the verbose output in the notebook)
    text_lines_list = []
    text_lines_list.append("=" * 60)
    text_lines_list.append("KHỞI TẠO MA TRẬN PHÂN HOẠCH")
    text_lines_list.append("=" * 60)
    
    # Initial partition matrix
    init_mat = np.zeros((k, n_samples), dtype=int)
    for i, c in enumerate(assignments):
        init_mat[c, i] = 1
    
    # Build initial partition matrix as text
    header = "\t".join(sample_labels)
    text_lines_list.append(header)
    for j in range(k):
        row_vals = "\t".join(str(init_mat[j, i]) for i in range(n_samples))
        text_lines_list.append(f"{cluster_labels[j]}\t{row_vals}")
    
    # Run K-Means iterations
    all_iterations_data = []
    n_iters = 0
    
    for iteration in range(1, 101):
        # Compute centroids
        centroids = np.zeros((k, X_scaled.shape[1]))
        for j in range(k):
            members = X_scaled[assignments == j]
            if len(members) > 0:
                centroids[j] = members.mean(axis=0)
        
        text_lines_list.append(f"")
        text_lines_list.append(f"Vector trọng tâm của các cụm:")
        for j in range(k):
            centroid_str = np.round(centroids[j], 5).tolist()
            text_lines_list.append(f"  {cluster_labels[j]}: m{j+1} = {centroid_str}")
        
        # Compute distance matrix
        dist_matrix = np.zeros((n_samples, k))
        for i in range(n_samples):
            for j in range(k):
                dist_matrix[i, j] = np.linalg.norm(X_scaled[i] - centroids[j])
        
        text_lines_list.append(f"")
        text_lines_list.append(f"Khoảng cách Euclide từ các vị trí đến các cụm:")
        dist_header = "\t".join([""] + cluster_labels)
        text_lines_list.append(dist_header)
        for i in range(n_samples):
            row_vals = "\t".join(f"{dist_matrix[i, j]:.5f}" for j in range(k))
            text_lines_list.append(f"{sample_labels[i]}\t{row_vals}")
        
        # Assign new clusters
        new_assignments = np.argmin(dist_matrix, axis=1)
        
        # Build new partition matrix
        new_mat = np.zeros((k, n_samples), dtype=int)
        for i, c in enumerate(new_assignments):
            new_mat[c, i] = 1
        
        text_lines_list.append(f"")
        text_lines_list.append(f"Ma trận phân hoạch (lần lặp {iteration}):")
        text_lines_list.append(header)
        for j in range(k):
            row_vals = "\t".join(str(new_mat[j, i]) for i in range(n_samples))
            text_lines_list.append(f"{cluster_labels[j]}\t{row_vals}")
        
        # Store iteration data for table format
        iter_data = {
            'iteration': iteration,
            'centroids': centroids.copy(),
            'dist_matrix': dist_matrix.copy(),
            'assignments': new_assignments.copy(),
            'partition_matrix': new_mat.copy()
        }
        all_iterations_data.append(iter_data)
        
        # Check convergence
        if np.all(new_assignments == assignments):
            text_lines_list.append(f"")
            text_lines_list.append(f"✅ Các điểm thuộc cụm không thay đổi → Thuật toán dừng lại sau {iteration} lần lặp.")
            n_iters = iteration
            assignments = new_assignments
            break
        
        assignments = new_assignments
        n_iters = iteration
    
    # Final centroids
    centroids = np.zeros((k, X_scaled.shape[1]))
    for j in range(k):
        members = X_scaled[assignments == j]
        if len(members) > 0:
            centroids[j] = members.mean(axis=0)
    
    # Final results
    text_lines_list.append(f"")
    text_lines_list.append("=" * 60)
    text_lines_list.append("KẾT QUẢ CUỐI CÙNG (k-Means):")
    text_lines_list.append("=" * 60)
    
    cluster_sizes = np.bincount(assignments, minlength=k).tolist()
    
    for j in range(k):
        mask = assignments == j
        members_idx = np.where(mask)[0]
        members_str = ", ".join([f"Mẫu {idx+1}" for idx in members_idx])
        centroid_str = np.round(centroids[j], 5).tolist()
        text_lines_list.append(f"  {cluster_labels[j]}: {members_str}")
        text_lines_list.append(f"         Trọng tâm: {centroid_str}")
    
    # Build tables for display
    
    # 0) Initial partition matrix table
    init_mat_header = [''] + sample_labels
    init_mat_rows = []
    for j in range(k):
        row_vals = [cluster_labels[j]]
        for i in range(n_samples):
            row_vals.append(int(init_mat[j, i]))
        init_mat_rows.append(row_vals)
    
    # 1) Per-iteration tables (centroids, distance matrix, partition matrix)
    iter_tables = []
    for iter_idx, iter_data in enumerate(all_iterations_data):
        it = iter_data['iteration']
        
        # Centroids table for this iteration
        centroid_header = ['Cụm'] + [f'Dim {d+1}' for d in range(X_scaled.shape[1])]
        centroid_rows = []
        for j in range(k):
            centroid_rows.append([f'Cụm {j+1}'] + [round(iter_data['centroids'][j][d], 5) for d in range(X_scaled.shape[1])])
        iter_tables.append(list_to_table(centroid_rows, centroid_header, title=f"Lần lặp {it}: Vector trọng tâm"))
        
        # Distance matrix table for this iteration
        dist_header = [''] + cluster_labels
        dist_rows = []
        for i in range(n_samples):
            dist_rows.append([f'Mẫu {i+1}'] + [round(iter_data['dist_matrix'][i][j], 5) for j in range(k)])
        iter_tables.append(list_to_table(dist_rows, dist_header, title=f"Lần lặp {it}: Khoảng cách Euclide"))
        
        # Partition matrix table
        part_header = [''] + sample_labels
        part_rows = []
        for j in range(k):
            row_vals = [cluster_labels[j]]
            for i in range(n_samples):
                row_vals.append(int(iter_data['partition_matrix'][j][i]))
            part_rows.append(row_vals)
        iter_tables.append(list_to_table(part_rows, part_header, title=f"Lần lặp {it}: Ma trận phân hoạch"))
    
    # 2) Final cluster details
    cluster_detail_rows = []
    for j in range(k):
        mask = assignments == j
        indices = np.where(mask)[0]
        for idx in indices:
            sample_num = idx + 1
            feature_values = []
            for col in X_raw.columns:
                feature_values.append(f"{col}={X_raw.iloc[idx][col]}")
            features_str = ", ".join(feature_values)
            target_val = str(df[target_label].iloc[idx])
            cluster_detail_rows.append([j+1, sample_num, features_str, target_val])
    
    # 3) Cluster summary
    cluster_summary_rows = []
    for j in range(k):
        mask = assignments == j
        count = int(np.sum(mask))
        indices = np.where(mask)[0]
        if count > 0:
            target_counts = df[target_label].iloc[indices].value_counts()
            target_dist = "; ".join([f"{val}={cnt}" for val, cnt in target_counts.items()])
            sample_list = (indices + 1).tolist()
            cluster_summary_rows.append([j+1, count, sample_list, target_dist])
    
    # 4) Metrics
    metrics = {}
    inertia = 0.0
    for i in range(n_samples):
        inertia += float(np.linalg.norm(X_scaled[i] - centroids[assignments[i]]) ** 2)
    metrics['inertia'] = round(inertia, 4)
    metrics['n_iters'] = n_iters
    
    if len(np.unique(assignments)) >= 2:
        try:
            metrics['silhouette'] = round(float(silhouette_score(X_scaled, assignments)), 4)
        except:
            metrics['silhouette'] = None
        try:
            metrics['davies_bouldin'] = round(float(davies_bouldin_score(X_scaled, assignments)), 4)
        except:
            metrics['davies_bouldin'] = None
        try:
            metrics['calinski_harabasz'] = round(float(calinski_harabasz_score(X_scaled, assignments)), 4)
        except:
            metrics['calinski_harabasz'] = None
    
    # Elbow WSS
    wss_values = []
    for k_test in range(1, min(6, len(df))):
        try:
            km = KMeans(n_clusters=k_test, random_state=42, n_init=10)
            km.fit(X_scaled)
            wss_values.append({'k': k_test, 'wss': float(km.inertia_)})
        except:
            pass
    
    text_lines = text_lines_list
    
    # Combine all tables: iteration tables first, then summary, then details
    tables = iter_tables + [
        list_to_table(
            cluster_summary_rows,
            ['Cụm', 'Số mẫu', 'Danh sách mẫu', f'Phân phối {target_label}'],
            title="Tổng quan các cụm"
        ),
        list_to_table(
            cluster_detail_rows,
            ['Cụm', 'Mẫu số', 'Giá trị thuộc tính', f'{target_label}'],
            title="Chi tiết từng mẫu trong các cụm"
        ),
    ]
    
    # Metrics table
    metrics_rows = []
    for k_name, v in metrics.items():
        if v is not None:
            metrics_rows.append([k_name, str(v)])
    if metrics_rows:
        tables.append(
            list_to_table(metrics_rows, ['Metric', 'Giá trị'], title="Đánh giá phân cụm")
        )
    
    return make_cell_result(
        text_lines=text_lines,
        tables=tables,
        extra={
            'k': k,
            'cluster_sizes': cluster_sizes,
            'inertia': round(inertia, 4),
            'n_iters': n_iters,
            'metrics': metrics,
            'n_features': X_encoded.shape[1],
            'wss': wss_values,
            'iter_details': [
                {
                    'iteration': d['iteration'],
                    'centroids': d['centroids'].tolist(),
                    'dist_matrix': d['dist_matrix'].tolist(),
                    'partition_matrix': d['partition_matrix'].tolist(),
                    'assignments': d['assignments'].tolist()
                }
                for d in all_iterations_data
            ]
        }
    )


# ==================== CELL 7: Kohonen SOM ====================
def cell7_kohonen(params=None):
    if params is None:
        params = {}
    k = int(params.get('k', 3))
    epochs = int(params.get('epochs', 10))
    alpha0 = float(params.get('alpha', 0.8))
    
    X = df.drop(target_label, axis=1).copy()
    y_true = df[target_label]
    
    X_encoded = pd.get_dummies(X, drop_first=False)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded).astype(np.float32)
    
    num_features = X_scaled.shape[1]
    num_samples = X_scaled.shape[0]
    sample_labels = [f'Mẫu {i+1}' for i in range(num_samples)]
    neuron_labels = [f'Nơron {j+1}' for j in range(k)]
    weight_labels = [f'w{j+1}' for j in range(k)]
    
    def euclidean_distance(a, b):
        return float(np.sqrt(np.sum((a - b) ** 2)))
    
    # Initialize weights from first k samples
    weights = X_scaled[:k].astype(float).copy()
    
    # Build text output (verbose)
    text_lines_list = []
    text_lines_list.append("=" * 60)
    text_lines_list.append("KHỞI TẠO VECTOR TRỌNG SỐ:")
    text_lines_list.append("=" * 60)
    for j in range(k):
        w_str = np.round(weights[j], 5).tolist()
        text_lines_list.append(f"  w{j+1} = {w_str}")
    
    current_alpha = alpha0
    
    # Build per-epoch tables
    iter_tables = []
    init_weight_rows = []
    for j in range(k):
        init_weight_rows.append([f'w{j+1}'] + [round(weights[j][d], 5) for d in range(num_features)])
    iter_tables.append(list_to_table(
        init_weight_rows,
        ['Trọng số'] + [f'Dim {d+1}' for d in range(num_features)],
        title="Khởi tạo vector trọng số"
    ))
    
    all_epochs_data = []
    
    for epoch in range(1, epochs + 1):
        text_lines_list.append(f"")
        text_lines_list.append("=" * 60)
        text_lines_list.append(f"LẦN LẶP {epoch} (α = {round(current_alpha, 6)}):")
        text_lines_list.append("=" * 60)
        
        epoch_sample_rows = []  # For table: sample, distances, winner, update
        
        for i in range(num_samples):
            x = X_scaled[i]
            dists = [euclidean_distance(x, weights[j]) for j in range(k)]
            winner = int(np.argmin(dists))
            
            # Print detailed step
            text_lines_list.append(f"")
            text_lines_list.append(f"  Xét {sample_labels[i]}: x = {x.tolist()}")
            for j in range(k):
                text_lines_list.append(f"    D({weight_labels[j]}) = {round(dists[j], 5)}")
            text_lines_list.append(f"  → Nơron thắng: {weight_labels[winner]}")
            
            # Update winner
            old_w = weights[winner].copy()
            weights[winner] = weights[winner] + current_alpha * (x - weights[winner])
            text_lines_list.append(f"  → Cập nhật {weight_labels[winner]}: {np.round(old_w, 5).tolist()} → {np.round(weights[winner], 5).tolist()}")
            
            epoch_sample_rows.append([
                sample_labels[i],
                x.tolist(),
                "; ".join([f"D({weight_labels[j]})={round(dists[j], 4)}" for j in range(k)]),
                weight_labels[winner]
            ])
        
        # Print weights after epoch
        text_lines_list.append(f"")
        text_lines_list.append(f"  Trọng số sau lần lặp {epoch}:")
        for j in range(k):
            text_lines_list.append(f"    {weight_labels[j]} = {np.round(weights[j], 5).tolist()}")
        
        # Build epoch tables
        # Distances table
        dist_header = [''] + [f'D({weight_labels[j]})' for j in range(k)]
        dist_rows = []
        for i in range(num_samples):
            x = X_scaled[i]
            dists = [euclidean_distance(x, weights[j]) for j in range(k)]
            dist_rows.append([sample_labels[i]] + [round(dists[j], 5) for j in range(k)])
        iter_tables.append(list_to_table(dist_rows, dist_header, title=f"Lần lặp {epoch}: Khoảng cách từ mẫu đến các nơron"))
        
        # Winner table
        win_rows = []
        for i in range(num_samples):
            x = X_scaled[i]
            dists = [euclidean_distance(x, weights[j]) for j in range(k)]
            winner = int(np.argmin(dists))
            win_rows.append([sample_labels[i], weight_labels[winner], round(dists[winner], 5)])
        iter_tables.append(list_to_table(win_rows, ['Mẫu', 'Nơron thắng', 'Khoảng cách'], title=f"Lần lặp {epoch}: Nơron thắng cho từng mẫu"))
        
        # Save epoch data
        epoch_data = {
            'epoch': epoch,
            'alpha': current_alpha,
            'weights': weights.copy(),
            'sample_data': epoch_sample_rows
        }
        all_epochs_data.append(epoch_data)
        
        # Reduce learning rate
        current_alpha = current_alpha / 2
    
    # Final classification
    text_lines_list.append("")
    text_lines_list.append("=" * 60)
    text_lines_list.append("PHÂN CỤM CUỐI CÙNG (Mạng Kohonen):")
    text_lines_list.append("=" * 60)
    text_lines_list.append("")
    text_lines_list.append("Khoảng cách từ các điểm đến các nơron (trọng số cuối):")
    
    # Final distance matrix header
    final_dist_header = "\t".join([""] + [f'{neuron_labels[j]} ({weight_labels[j]})' for j in range(k)])
    text_lines_list.append(final_dist_header)
    
    final_dist_rows = []
    for i in range(num_samples):
        dists = [euclidean_distance(X_scaled[i], weights[j]) for j in range(k)]
        row_str = "\t".join([sample_labels[i]] + [f"{dists[j]:.5f}" for j in range(k)])
        text_lines_list.append(row_str)
        final_dist_rows.append([sample_labels[i]] + [round(dists[j], 5) for j in range(k)])
    
    # Final assignments
    assignments = np.zeros(num_samples, dtype=int)
    for i in range(num_samples):
        dists = [euclidean_distance(X_scaled[i], weights[j]) for j in range(k)]
        assignments[i] = int(np.argmin(dists))
    
    text_lines_list.append("")
    text_lines_list.append("Kết quả:")
    for j in range(k):
        mask = assignments == j
        members = [sample_labels[i] for i in range(num_samples) if mask[i]]
        w_str = np.round(weights[j], 5).tolist()
        text_lines_list.append(f"  Cụm {j+1} (đại diện bởi {weight_labels[j]} = {w_str}): {', '.join(members)}")
    
    # Final distance table
    iter_tables.append(list_to_table(
        final_dist_rows,
        [''] + [f'{neuron_labels[j]} ({weight_labels[j]})' for j in range(k)],
        title="Khoảng cách cuối cùng từ các điểm đến các nơron"
    ))
    
    # Cluster summary
    cluster_summary_rows = []
    final_centroids = weights.copy()
    for j in range(k):
        mask = assignments == j
        count = int(np.sum(mask))
        indices = np.where(mask)[0]
        if count > 0:
            target_counts = y_true.iloc[indices].value_counts()
            target_dist = "; ".join([f"{val}={cnt}" for val, cnt in target_counts.items()])
            sample_list = (indices + 1).tolist()
            w_str = np.round(final_centroids[j], 5).tolist()
            cluster_summary_rows.append([j+1, count, sample_list, target_dist, str(w_str)])
    
    text_lines = text_lines_list
    
    tables = iter_tables + [
        list_to_table(
            cluster_summary_rows,
            ['Cụm', 'Số mẫu', 'Danh sách mẫu', f'Phân phối {target_label}', 'Trọng số đại diện'],
            title="Kết quả phân cụm Kohonen SOM"
        )
    ]
    
    return make_cell_result(
        text_lines=text_lines,
        tables=tables,
        extra={
            'k': k,
            'epochs': epochs,
            'alpha0': alpha0,
            'n_iters': len(all_epochs_data),
            'n_features': num_features,
            'final_weights': [np.round(weights[j], 5).tolist() for j in range(k)],
            'assignments': assignments.tolist()
        }
    )


# ==================== CELL 8: So sánh ====================
def cell8_comparison(params=None):
    if params is None:
        params = {}
    k = int(params.get('k', 3))
    epochs = int(params.get('epochs', 10))
    alpha0 = float(params.get('alpha', 0.8))
    
    X = df.drop(target_label, axis=1).copy()
    y_true = df[target_label]
    
    X_encoded = pd.get_dummies(X, drop_first=False)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded).astype(np.float32)
    
    n_samples = X_scaled.shape[0]
    sample_labels = [f'Mẫu {i+1}' for i in range(n_samples)]
    
    def euclidean_distance(a, b):
        return float(np.sqrt(np.sum((a - b) ** 2)))
    
    # ====== 1) K-MEANS ======
    np.random.seed(42)
    init_assignments = np.random.randint(0, k, n_samples)
    for j in range(k):
        if np.sum(init_assignments == j) == 0:
            init_assignments[j % n_samples] = j
    assignments_km = init_assignments.copy()
    
    for iteration in range(100):
        centroids_km = np.zeros((k, X_scaled.shape[1]))
        for j in range(k):
            members = X_scaled[assignments_km == j]
            if len(members) > 0:
                centroids_km[j] = members.mean(axis=0)
        dist_matrix = np.zeros((n_samples, k))
        for i in range(n_samples):
            for j in range(k):
                dist_matrix[i, j] = euclidean_distance(X_scaled[i], centroids_km[j])
        new_assignments = np.argmin(dist_matrix, axis=1)
        if np.all(new_assignments == assignments_km):
            break
        assignments_km = new_assignments
    
    # Final K-Means centroids
    centroids_km = np.zeros((k, X_scaled.shape[1]))
    for j in range(k):
        members = X_scaled[assignments_km == j]
        if len(members) > 0:
            centroids_km[j] = members.mean(axis=0)
    
    # ====== 2) KOHONEN SOM ======
    weights_koh = X_scaled[:k].astype(float).copy()
    current_alpha = alpha0
    for epoch in range(epochs):
        for i in range(n_samples):
            x = X_scaled[i]
            dists = [euclidean_distance(x, weights_koh[j]) for j in range(k)]
            winner = int(np.argmin(dists))
            weights_koh[winner] = weights_koh[winner] + current_alpha * (x - weights_koh[winner])
        current_alpha = current_alpha / 2
    
    assignments_koh = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        dists = [euclidean_distance(X_scaled[i], weights_koh[j]) for j in range(k)]
        assignments_koh[i] = int(np.argmin(dists))
    
    # ====== 3) BUILD COMPARISON ======
    # Align clusters: need to match cluster labels between methods
    # Build mapping from K-Means assignment to Kohonen assignment
    # Find best matching by checking overlap
    def align_clusters(assignments_a, assignments_b, k):
        """Reorder clusters in b to best match a"""
        from itertools import permutations
        best_perm = None
        best_score = -1
        for perm in permutations(range(k)):
            mapped = np.array([perm[a] for a in assignments_b])
            score = np.sum(assignments_a == mapped)
            if score > best_score:
                best_score = score
                best_perm = perm
        # Remap b
        mapping = {old: new for new, old in enumerate(best_perm)}
        return np.array([mapping[a] for a in assignments_b])
    
    # Align Kohonen to K-Means
    assignments_koh_aligned = align_clusters(assignments_km, assignments_koh, k)
    
    # Build comparison table (like Lab5: k-Means Cụm vs Kohonen Cụm)
    comparison_rows = []
    kmeans_sets = []
    kohonen_sets = []
    for j in range(k):
        km_mask = assignments_km == j
        koh_mask = assignments_koh_aligned == j
        km_members = ", ".join([sample_labels[i] for i in range(n_samples) if km_mask[i]])
        koh_members = ", ".join([sample_labels[i] for i in range(n_samples) if koh_mask[i]])
        comparison_rows.append([str(j), f"Cụm {j+1}: {km_members}", f"Cụm {j+1}: {koh_members}"])
        kmeans_sets.append(frozenset([i for i in range(n_samples) if km_mask[i]]))
        kohonen_sets.append(frozenset([i for i in range(n_samples) if koh_mask[i]]))
    
    # Check if results are the same
    same = set(kmeans_sets) == set(kohonen_sets)
    same_text = "✅ Hai thuật toán cho kết quả phân cụm giống nhau (các thành viên trong từng cụm tương ứng nhau)." if same else "⚠️ Hai thuật toán cho kết quả phân cụm khác nhau."
    
    # ====== 4) METRICS ======
    def safe_metrics(labels):
        unique = np.unique(labels)
        if len(unique) < 2:
            return {'Silhouette': None, 'Davies-Bouldin': None, 'Calinski-Harabasz': None, 'n_clusters': len(unique)}
        try:
            return {
                'Silhouette': round(float(silhouette_score(X_scaled, labels)), 4),
                'Davies-Bouldin': round(float(davies_bouldin_score(X_scaled, labels)), 4),
                'Calinski-Harabasz': round(float(calinski_harabasz_score(X_scaled, labels)), 4),
                'n_clusters': len(unique),
            }
        except:
            return {'Silhouette': None, 'Davies-Bouldin': None, 'Calinski-Harabasz': None, 'n_clusters': len(unique)}
    
    kmeans_metrics = safe_metrics(assignments_km)
    kohonen_metrics = safe_metrics(assignments_koh_aligned)
    
    metrics_compare_rows = []
    for method, m in [('K-Means', kmeans_metrics), ('Kohonen (SOM)', kohonen_metrics)]:
        row = [method, m['n_clusters']]
        for metric_name in ['Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz']:
            val = m.get(metric_name)
            row.append(str(val) if val is not None else 'N/A')
        metrics_compare_rows.append(row)
    
    text_lines = [
        "=" * 60,
        "SO SÁNH KẾT QUẢ HAI THUẬT TOÁN:",
        "=" * 60,
    ]
    text_lines.append(f"{'k-Means (Cụm)':<40} {'Kohonen (Cụm)'}")
    for row in comparison_rows:
        text_lines.append(f"{row[1]:<40} {row[2]}")
    text_lines.append("")
    text_lines.append(same_text)
    
    tables = [
        list_to_table(
            comparison_rows,
            ['STT', 'k-Means (Cụm)', 'Kohonen (Cụm)'],
            title="So sánh kết quả phân cụm"
        ),
        list_to_table(
            metrics_compare_rows,
            ['Phương pháp', 'Số cụm', 'Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz'],
            title="Bảng so sánh metrics"
        ),
    ]
    
    return make_cell_result(
        text_lines=text_lines + ["", same_text],
        tables=tables,
        extra={
            'k': k,
            'epochs': epochs,
            'alpha0': alpha0,
            'kmeans_metrics': kmeans_metrics,
            'kohonen_metrics': kohonen_metrics,
            'kmeans_assignments': assignments_km.tolist(),
            'kohonen_assignments': assignments_koh_aligned.tolist(),
            'same_result': same,
            'comparison_rows': comparison_rows
        }
    )


# ==================== Flask Routes ====================
CELL_FUNCTIONS = {
    1: ('Tương quan Pearson', cell1_correlation),
    2: ('Apriori & Luật kết hợp', cell2_apriori),
    3: ('Tập thô (Rough Set)', cell3_roughset),
    4: ('Naive Bayes', cell4_naivebayes),
    5: ('Cây quyết định ID3', cell5_id3),
    6: ('K-Means', cell6_kmeans),
    7: ('Kohonen SOM', cell7_kohonen),
    8: ('So sánh K-Means vs SOM', cell8_comparison)
}

CELL_PARAMS = {
    1: {
        'col_chosen': {'type': 'select', 'label': 'Thuộc tính', 'options': feature_cols, 'default': 'Outlook'}
    },
    2: {
        'min_sup': {'type': 'slider', 'label': 'Min Support', 'min': 0.1, 'max': 0.5, 'step': 0.05, 'default': 0.2},
        'min_conf': {'type': 'slider', 'label': 'Min Confidence', 'min': 0.5, 'max': 1.0, 'step': 0.05, 'default': 0.7}
    },
    4: {
        'laplace': {'type': 'toggle', 'label': 'Làm trơn Laplace', 'default': True}
    },
    6: {
        'k': {'type': 'slider', 'label': 'Số cụm (k)', 'min': 2, 'max': 5, 'step': 1, 'default': 2}
    },
    7: {
        'k': {'type': 'slider', 'label': 'Số cụm (k)', 'min': 2, 'max': 5, 'step': 1, 'default': 3},
        'epochs': {'type': 'slider', 'label': 'Số lần lặp (epochs)', 'min': 1, 'max': 10, 'step': 1, 'default': 5},
        'alpha': {'type': 'slider', 'label': 'Tốc độ học α', 'min': 0.1, 'max': 1.0, 'step': 0.1, 'default': 0.8}
    },
    8: {
        'k': {'type': 'slider', 'label': 'Số cụm (k)', 'min': 2, 'max': 5, 'step': 1, 'default': 3},
        'epochs': {'type': 'slider', 'label': 'Số lần lặp (epochs)', 'min': 1, 'max': 10, 'step': 1, 'default': 5},
        'alpha': {'type': 'slider', 'label': 'Tốc độ học α', 'min': 0.1, 'max': 1.0, 'step': 0.1, 'default': 0.8}
    }
}


@app.route('/')
def index():
    data_preview = df.to_html(index=False, classes='table table-striped table-sm')
    stats = {
        'rows': len(df),
        'cols': len(df.columns),
        'columns': df.columns.tolist(),
        'target': target_label,
        'target_values': df[target_label].value_counts().to_dict()
    }
    return render_template('index.html', data_preview=data_preview, stats=stats, cell_params=CELL_PARAMS)


@app.route('/api/cell/<int:cell_id>', methods=['POST'])
def run_cell_api(cell_id):
    if cell_id not in CELL_FUNCTIONS:
        return jsonify({'error': 'Không tìm thấy cell!'}), 404
    
    params = request.get_json(silent=True) or {}
    name, func = CELL_FUNCTIONS[cell_id]
    
    try:
        result = func(params)
        result['cell_id'] = cell_id
        result['name'] = name
        result['status'] = 'success'
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'cell_id': cell_id,
            'name': name,
            'status': 'error',
            'error': str(e)
        })


@app.route('/api/cell/<int:cell_id>/params', methods=['GET'])
def get_cell_params(cell_id):
    params = CELL_PARAMS.get(cell_id, {})
    return jsonify(params)


@app.route('/api/run-all', methods=['POST'])
def run_all_cells_api():
    results = {}
    for cell_id, (name, func) in CELL_FUNCTIONS.items():
        try:
            result = func({})
            result['cell_id'] = cell_id
            result['name'] = name
            result['status'] = 'success'
            results[cell_id] = result
        except Exception as e:
            results[cell_id] = {
                'cell_id': cell_id,
                'name': name,
                'status': 'error',
                'error': str(e)
            }
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True, port=5000)