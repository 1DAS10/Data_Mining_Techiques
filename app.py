import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score
import io

# --- 1. TỰ CÀI ĐẶT THUẬT TOÁN (CUSTOM IMPLEMENTATIONS) ---

def pearson_correlation(x, y):
    """Tính hệ số tương quan Pearson thủ công"""
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x_sq = np.sum(x**2)
    sum_y_sq = np.sum(y**2)
    sum_xy = np.sum(x * y)
    
    numerator = n * sum_xy - sum_x * sum_y
    denominator = np.sqrt((n * sum_x_sq - sum_x**2) * (n * sum_y_sq - sum_y**2))
    
    return numerator / denominator if denominator != 0 else 0

class NaiveBayesClassifier:
    """Phân loại Naive Bayes với Laplace Smoothing"""
    def __init__(self):
        self.prior = {}
        self.likelihood = {}
        self.classes = []

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples = len(y)
        for c in self.classes:
            self.prior[c] = (np.sum(y == c) + 1) / (n_samples + len(self.classes))
            self.likelihood[c] = {}
            X_c = X[y == c]
            for col in X.columns:
                self.likelihood[c][col] = {}
                values = np.unique(X[col])
                for v in values:
                    self.likelihood[c][col][v] = (np.sum(X_c[col] == v) + 1) / (len(X_c) + len(values))

    def predict(self, X):
        preds = []
        for _, row in X.iterrows():
            probs = {}
            for c in self.classes:
                p = self.prior[c]
                for col in X.columns:
                    val = row[col]
                    p *= self.likelihood[c][col].get(val, 1 / (len(self.likelihood[c][col]) + 1))
                probs[c] = p
            preds.append(max(probs, key=probs.get))
        return np.array(preds)

def get_rough_set_reducts(df, target_col):
    """Tìm tập rút gọn (Reducts) bằng ma trận phân biệt"""
    features = [c for c in df.columns if c != target_col]
    n = len(df)
    reducts = set()
    
    # Lấy mẫu nhỏ để đảm bảo hiệu năng trên web nếu dữ liệu quá lớn
    sample_df = df.head(100) 
    for i in range(len(sample_df)):
        for j in range(i + 1, len(sample_df)):
            if sample_df.iloc[i][target_col] != sample_df.iloc[j][target_col]:
                diff = []
                for f in features:
                    if sample_df.iloc[i][f] != sample_df.iloc[j][f]:
                        diff.append(f)
                if len(diff) == 1:
                    reducts.add(diff[0])
    return list(reducts) if reducts else features[:2]

# --- 2. GIAO DIỆN STREAMLIT ---

st.set_page_config(page_title="Data Mining Project", layout="wide")
st.title("🚀 Hệ thống Khám phá Dữ liệu & Học máy")

uploaded_file = st.sidebar.file_uploader("Tải lên file CSV dữ liệu", type=["csv"])

if uploaded_file:
    @st.cache_data
    def load_data(file):
        df = pd.read_csv(file)
        # Tiền xử lý: Bỏ các cột ID/Timestamp nếu có[cite: 3]
        cols_to_drop = [c for c in df.columns if 'id' in c.lower() or 'time' in c.lower()]
        return df.drop(columns=cols_to_drop)

    df = load_data(uploaded_file)
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Tổng quan", "📈 Tương quan", "🕸️ Luật kết hợp", 
        "🧠 Naive Bayes", "🌳 Cây quyết định", "💎 Tập thô"
    ])

    # --- TAB 1: TỔNG QUAN ---
    with tab1:
        st.subheader("Dữ liệu thô")
        st.write(df.head(10))
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"Số dòng: {df.shape[0]}")
            st.info(f"Số cột: {df.shape[1]}")
        with col2:
            target_col = st.selectbox("Chọn biến mục tiêu (Target):", df.columns, index=len(df.columns)-1)
            fig_pie = px.pie(df, names=target_col, title=f"Phân bổ lớp {target_col}")
            st.plotly_chart(fig_pie)

    # --- TAB 2: TƯƠNG QUAN ---
    with tab2:
        st.subheader("Hệ số tương quan Pearson")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) >= 2:
            c1, c2 = st.columns(2)
            var1 = c1.selectbox("Biến X:", num_cols, index=0)
            var2 = c2.selectbox("Biến Y:", num_cols, index=1)
            
            r = pearson_correlation(df[var1], df[var2])
            st.metric(f"Hệ số r ({var1} vs {var2})", f"{r:.4f}")
            
            fig_scat = px.scatter(df, x=var1, y=var2, color=target_col, trendline="ols")
            st.plotly_chart(fig_scat)
        else:
            st.warning("Cần ít nhất 2 cột số để tính tương quan.")

    # --- TAB 3: LUẬT KẾT HỢP ---
    with tab3:
        st.subheader("Thuật toán Apriori")
        min_supp = st.slider("Minimum Support", 0.01, 0.5, 0.1)
        
        # Rời rạc hóa dữ liệu số thành categorical[cite: 3]
        df_disc = df.copy()
        for col in df_disc.select_dtypes(include=[np.number]).columns:
            df_disc[col] = pd.cut(df_disc[col], bins=3, labels=["Low", "Mid", "High"])
        
        df_dummy = pd.get_dummies(df_disc)
        frequent_itemsets = apriori(df_dummy, min_support=min_supp, use_colnames=True)
        
        if not frequent_itemsets.empty:
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
            st.write(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values('lift', ascending=False))
        else:
            st.error("Không tìm thấy luật kết hợp với Support này.")

    # --- TAB 4: NAIVE BAYES ---
    with tab4:
        st.subheader("Phân loại Naive Bayes (Manual Implementation)")
        # Rời rạc hóa để dùng NB
        X = df_disc.drop(columns=[target_col])
        y = df_disc[target_col]
        
        nb = NaiveBayesClassifier()
        nb.fit(X, y)
        y_pred = nb.predict(X)
        
        st.write(f"**Độ chính xác (Accuracy):** {accuracy_score(y, y_pred):.2%}")
        
        cm = confusion_matrix(y, y_pred)
        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        plt.xlabel("Dự đoán")
        plt.ylabel("Thực tế")
        st.pyplot(fig_cm)

    # --- TAB 5: CÂY QUYẾT ĐỊNH ---
    with tab5:
        st.subheader("Cây quyết định ID3 (Entropy)")
        X_tree = pd.get_dummies(df.drop(columns=[target_col]))
        y_tree = df[target_col]
        
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=3)
        clf.fit(X_tree, y_tree)
        
        fig_tree = plt.figure(figsize=(15,8))
        plot_tree(clf, feature_names=X_tree.columns, class_names=True, filled=True)
        st.pyplot(fig_tree)

    # --- TAB 6: TẬP THÔ ---
    with tab6:
        st.subheader("Lý thuyết Tập thô (Rough Set Theory)")
        if st.button("Tìm tập rút gọn tối thiểu (Reducts)"):
            reducts = get_rough_set_reducts(df_disc, target_col)
            st.success(f"Các thuộc tính rút gọn tìm được: {', '.join(reducts)}")
            st.info("Logic: Sử dụng ma trận phân biệt để loại bỏ các thuộc tính dư thừa nhưng vẫn giữ nguyên khả năng phân lớp.")

else:
    st.info("Vui lòng tải file CSV từ sidebar để bắt đầu.")