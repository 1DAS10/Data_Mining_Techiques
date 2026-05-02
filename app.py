import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import math
from itertools import combinations

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree

# ================= 1. THIẾT LẬP GIAO DIỆN =================
st.set_page_config(page_title="Hệ thống Khai thác Dữ liệu", page_icon="📊", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #fcfcfc; font-family: 'Segoe UI', sans-serif; }
    [data-testid="stMetricValue"] { font-size: 1.8rem; color: #1e293b; font-weight: 700; }
    div[data-testid="stMetric"] { background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #edf2f7; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    h1, h2, h3 { color: #334155 !important; }
    .stButton>button { width: 100%; border-radius: 8px; background-color: #3b82f6; color: white; border: none; transition: 0.3s; }
    .stButton>button:hover { background-color: #2563eb; transform: translateY(-2px); }
    </style>
    """, unsafe_allow_html=True)

# ================= 2. HÀM HỖ TRỢ & THUẬT TOÁN =================
@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file, engine='openpyxl')
        else:
            st.error("Định dạng file không được hỗ trợ!")
            return None
            
        df = df.loc[:, ~df.columns.duplicated()]
        # Bỏ các cột không cần thiết theo logic gốc
        cols_to_drop = ['student_id', 'timestamp']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc file: {str(e)}")
        return None

def preprocess_for_ml(df, target):
    df_ml = df.copy().dropna()
    le = LabelEncoder()
    for col in df_ml.columns:
        if not pd.api.types.is_numeric_dtype(df_ml[col]) or df_ml[col].dtype == 'object':
            df_ml[col] = le.fit_transform(df_ml[col].astype(str))
    return df_ml

def pearson_correlation(x, y):
    n = len(x)
    if n == 0: return 0
    mean_x, mean_y = sum(x)/n, sum(y)/n
    tu_so = sum((x_i - mean_x) * (y_i - mean_y) for x_i, y_i in zip(x, y))
    sum_sq_x = sum((x_i - mean_x)**2 for x_i in x)
    sum_sq_y = sum((y_i - mean_y)**2 for y_i in y)
    mau_so = math.sqrt(sum_sq_x * sum_sq_y)
    return tu_so / mau_so if mau_so != 0 else 0

# --- Class Naive Bayes tự code ---
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
            self.class_probs[c] = np.sum(y == c) / n_samples
            
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
                    if prob == 0: prob = 1e-10 
                    self.feature_probs[c][i][val] = prob

    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = {}
            for c in self.classes:
                prior = np.log(self.class_probs[c])
                conditional = 0
                for i, val in enumerate(x):
                    prob = self.feature_probs[c][i].get(val, 1e-10) 
                    conditional += np.log(prob)
                posteriors[c] = prior + conditional
            predictions.append(max(posteriors, key=posteriors.get))
        return np.array(predictions)

# ================= 3. THANH ĐIỀU KHIỂN =================
with st.sidebar:
    st.title("📊 Data Mining Tool")
    uploaded_file = st.file_uploader("Tải file dữ liệu", type=["csv", "xlsx", "xls"])
    
    if uploaded_file:
        st.success(f"Đã tải: {uploaded_file.name}")
        algo = st.selectbox("Chọn thuật toán:", 
            ["Trình diễn dữ liệu", "Tương quan Pearson", "Luật kết hợp (Apriori)", 
             "Phân loại Naive Bayes", "Cây quyết định ID3", "Lý thuyết Tập thô (Rough Set)"])
        
        st.divider()
        if algo in ["Phân loại Naive Bayes", "Cây quyết định ID3"]:
            test_size = st.slider("Tỉ lệ Test (%)", 10, 50, 30) / 100
        if algo == "Cây quyết định ID3":
            max_depth = st.slider("Độ sâu tối đa của cây (ID3)", 2, 20, 5)

# ================= 4. XỬ LÝ CHÍNH =================
if not uploaded_file:
    st.title("Hệ thống Phân tích Dữ liệu Thông minh")
    st.info("💡 Mời bạn tải file CSV hoặc Excel từ thanh bên trái để bắt đầu.")
else:
    df = load_data(uploaded_file)
    if df is not None:
        target_col = st.sidebar.selectbox("🎯 Biến mục tiêu (Target):", df.columns, index=len(df.columns)-1)

        # --- TRÌNH DIỄN DỮ LIỆU ---
        if algo == "Trình diễn dữ liệu":
            st.header("📋 Tổng quan dữ liệu")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Số dòng", df.shape[0])
            c2.metric("Số cột", df.shape[1])
            c3.metric("Ô trống", df.isna().sum().sum())
            c4.metric("Trùng lặp", df.duplicated().sum())
            
            st.subheader("Bản xem trước (10 dòng đầu)")
            st.dataframe(df.head(10), use_container_width=True)
            
            if target_col:
                fig = px.pie(df, names=target_col, title=f"Phân bổ lớp của {target_col}", hole=0.4)
                st.plotly_chart(fig, use_container_width=True)

        # --- TƯƠNG QUAN PEARSON ---
        elif algo == "Tương quan Pearson":
            st.header("🔗 Hệ số tương quan Pearson")
            feature = st.selectbox("Chọn thuộc tính so sánh với Target:", [c for c in df.select_dtypes(include=np.number).columns if c != target_col] if pd.api.types.is_numeric_dtype(df[target_col]) else [c for c in df.columns if c != target_col])
            
            if st.button("Tính toán tương quan"):
                df_p = preprocess_for_ml(df[[feature, target_col]], target_col)
                x = df_p[feature].values
                y = df_p[target_col].values
                r = pearson_correlation(x, y)
                
                abs_r = abs(r)
                if abs_r == 1.0: conclusion = "Quan hệ tuyến tính hoàn hảo."
                elif 0.7 <= abs_r < 1.0: conclusion = "Quan hệ tuyến tính chặt chẽ."
                elif 0.3 <= abs_r < 0.7: conclusion = "Quan hệ tuyến tính trung bình."
                else: conclusion = "Rất ít hoặc không có quan hệ tuyến tính."

                col_l, col_r = st.columns([1, 2])
                with col_l:
                    st.metric("Hệ số r", f"{r:.4f}")
                    st.info(f"**Kết luận:** {conclusion}")
                with col_r:
                    st.plotly_chart(px.scatter(df_p, x=feature, y=target_col, trendline="ols", title="Biểu đồ tương quan"), use_container_width=True)

        # --- APRIORI ---
        elif algo == "Luật kết hợp (Apriori)":
            st.header("🛒 Khai phá luật kết hợp (Apriori)")
            col1, col2 = st.columns(2)
            min_sup = col1.slider("Mức độ hỗ trợ tối thiểu (Min Support %)", 1, 50, 5) / 100
            min_conf = col2.slider("Độ tin cậy tối thiểu (Min Confidence %)", 10, 100, 50) / 100
            
            if st.button("Bắt đầu phân tích"):
                with st.spinner("Đang chuẩn bị dữ liệu và tính toán (Có thể mất thời gian với dữ liệu lớn)..."):
                    df_ap = df.copy().dropna()
                    # Chuyển số thành low/mid/high như logic gốc
                    for col in df_ap.select_dtypes(include=[np.number]).columns:
                        q1, q2 = df_ap[col].quantile([0.33, 0.67])
                        df_ap[col] = pd.cut(df_ap[col], bins=[-np.inf, q1, q2, np.inf], labels=['low', 'mid', 'high'])
                    
                    # Lấy mẫu 300 dòng để tránh treo trình duyệt nếu file quá lớn
                    df_ap = df_ap.sample(n=min(300, len(df_ap)), random_state=42)
                    trans = [set(f"{c}_{v}" for c, v in row.items()) for _, row in df_ap.iterrows()]
                    min_sup_count = max(1, int(np.ceil(min_sup * len(trans))))
                    
                    st.success(f"Đã tạo {len(trans)} giao tác. Min support count = {min_sup_count}")
                    
                    # Mô phỏng tìm tập phổ biến (Sử dụng code rút gọn để minh họa trên UI)
                    from mlxtend.preprocessing import TransactionEncoder
                    from mlxtend.frequent_patterns import apriori, association_rules
                    
                    te = TransactionEncoder()
                    te_ary = te.fit([list(t) for t in trans]).transform([list(t) for t in trans])
                    df_trans = pd.DataFrame(te_ary, columns=te.columns_)
                    
                    frequent_itemsets = apriori(df_trans, min_support=min_sup, use_colnames=True)
                    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
                    
                    if not frequent_itemsets.empty:
                        st.subheader("Các tập phổ biến (Top 10)")
                        fi_show = frequent_itemsets.sort_values(by=['length', 'support'], ascending=[False, False]).head(10)
                        fi_show['itemsets'] = fi_show['itemsets'].apply(lambda x: ', '.join(list(x)))
                        st.dataframe(fi_show)

                        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)
                        if not rules.empty:
                            st.subheader(f"Các luật kết hợp (Top 10 - Y là {target_col})")
                            rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                            rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
                            
                            # Lọc luật có vế phải chứa target_col
                            target_rules = rules[rules['consequents'].str.contains(target_col, na=False)]
                            if target_rules.empty: target_rules = rules # Nếu không có thì hiện tất cả
                            
                            show_rules = target_rules[['antecedents', 'consequents', 'support', 'confidence']].head(10)
                            st.dataframe(show_rules)
                        else:
                            st.warning("Không tìm thấy luật nào đạt độ tin cậy yêu cầu.")
                    else:
                        st.warning("Không tìm thấy tập phổ biến nào đạt Support yêu cầu.")

        # --- NAIVE BAYES ---
        elif algo == "Phân loại Naive Bayes":
            st.header("🕊️ Phân loại Naive Bayes")
            use_laplace = st.checkbox("Sử dụng Làm trơn Laplace (Laplace Smoothing)", value=True)
            
            if st.button("Huấn luyện mô hình"):
                df_ml = preprocess_for_ml(df, target_col)
                X = df_ml.drop(target_col, axis=1).values
                y = df_ml[target_col].values
                classes = np.unique(y)
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                
                model = NaiveBayesClassifier(smoothing=use_laplace)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                acc = accuracy_score(y_test, y_pred)
                st.success(f"Đã huấn luyện xong mô hình! Độ chính xác (Accuracy): {acc*100:.2f}%")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Ma trận nhầm lẫn (Confusion Matrix)")
                    cm = confusion_matrix(y_test, y_pred)
                    fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Dự đoán", y="Thực tế"), x=classes, y=classes)
                    st.plotly_chart(fig_cm, use_container_width=True)
                with col2:
                    st.subheader("Báo cáo phân loại")
                    st.code(classification_report(y_test, y_pred, target_names=[str(c) for c in classes]))

        # --- ID3 ---
        elif algo == "Cây quyết định ID3":
            st.header("🌳 Cây quyết định (ID3)")
            if st.button("Xây dựng cây"):
                df_ml = preprocess_for_ml(df, target_col)
                X = df_ml.drop(target_col, axis=1)
                y = df_ml[target_col]
                
                model = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=42).fit(X, y)
                st.success("Xây dựng cây hoàn tất dựa trên Information Gain (Entropy)!")
                
                fig, ax = plt.subplots(figsize=(20, 10))
                plot_tree(model, feature_names=X.columns, class_names=[str(c) for c in np.unique(y)], filled=True, rounded=True, ax=ax)
                st.pyplot(fig)

        # --- ROUGH SET ---
        elif algo == "Lý thuyết Tập thô (Rough Set)":
            st.header("🧮 Phân tích Tập thô")
            sample_size = st.number_input("Kích thước tập mẫu để lập ma trận phân biệt", min_value=10, max_value=200, value=50)
            
            if st.button("Lập ma trận phân biệt & Tìm Reducts"):
                with st.spinner("Đang tính toán trên tập mẫu..."):
                    df_rs = df.copy().dropna()
                    for col in df_rs.select_dtypes(include=[np.number]).columns:
                        df_rs[col] = pd.cut(df_rs[col], bins=3, labels=['low', 'mid', 'high']).astype(str)
                    
                    df_sample = df_rs.sample(n=min(sample_size, len(df_rs)), random_state=42).reset_index(drop=True)
                    X_sample = df_sample.drop(target_col, axis=1)
                    y_sample = df_sample[target_col]
                    attributes = list(X_sample.columns)
                    
                    n = len(df_sample)
                    conditions = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if y_sample.iloc[i] != y_sample.iloc[j]:
                                diff_attrs = {attr for attr in attributes if X_sample.iloc[i][attr] != X_sample.iloc[j][attr]}
                                if diff_attrs: conditions.append(diff_attrs)
                    
                    unique_conditions = [list(c) for c in set(tuple(sorted(c)) for c in conditions if c)]
                    
                    # Mô phỏng tìm reducts
                    valid_reducts = []
                    for k in range(1, len(attributes) + 1):
                        for subset in combinations(attributes, k):
                            subset_set = set(subset)
                            is_valid = all(set(cond).intersection(subset_set) for cond in unique_conditions)
                            if is_valid: valid_reducts.append(list(subset))
                        if valid_reducts: break 
                    
                    st.success("Phân tích hoàn tất!")
                    st.subheader(f"Tìm thấy {len(valid_reducts)} tập rút gọn (Reducts) tối thiểu:")
                    for i, red in enumerate(valid_reducts[:5]): # Hiện tối đa 5 reducts
                        st.markdown(f"**Reduct {i+1}:** `[{', '.join(red)}]`")
                    
                    if valid_reducts:
                        chosen_reduct = valid_reducts[0]
                        st.info(f"💡 **Ứng dụng:** Với Reduct `{chosen_reduct}`, hệ thống có thể phân loại chính xác `{target_col}` mà không cần dùng đến toàn bộ các thuộc tính gốc, giúp tối ưu hiệu năng mô hình.")