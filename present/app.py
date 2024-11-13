from process_data import *

# Tiêu đề ứng dụng
st.title("GROUP 11")

# Tải file CSV từ người dùng
uploaded_file = st.file_uploader("Choose data file", type="csv")

# Nếu có file được tải lên
if uploaded_file is not None:
    # Đọc file CSV và chọn các features thành DataFrame
    data = pd.read_csv(uploaded_file)
    data['Date'] = pd.to_datetime(data['Date'])
    scaler = MinMaxScaler()

    # Chọn features
    features = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    selected_features = st.sidebar.multiselect("Select other features", options=features)
    default_features = ['Store', 'Dept', 'Date', 'Weekly_Sales']
    all_selected_features = list(default_features + selected_features)
    df = data[all_selected_features]
    df['Weekly_Sales_Scaled'] = scaler.fit_transform(df[['Weekly_Sales']]) # Scale để dễ tính
    calcul_feature = list(['Weekly_Sales_Scaled'] + selected_features) # Feature để dùng cho anomaly detection
    # Chọn ngày
    unique_dates = df['Date'].dt.date.unique()
    selected_date = st.sidebar.selectbox("Select a date", options=unique_dates)
    # Chọn store
    unique_stores = df['Store'].unique().tolist()
    unique_stores.insert(0, "All")
    selected_stores = st.sidebar.multiselect("Select store(s)", options=unique_stores, default="All")
    # Chọn thuật toán
    algorithms = ["LOF", "DBSCAN", "KDE"]
    selected_algorithm = st.sidebar.selectbox("Select algorithm", options=algorithms)

    # Lọc dữ liệu theo ngày đã chọn
    filtered_df = df[df['Date'] == pd.to_datetime(selected_date)]
    # Lọc dữ liệu theo cửa hàng đã chọn
    if "All" not in selected_stores:
        filtered_df = filtered_df[filtered_df['Store'].isin(selected_stores)]
    plot(df=filtered_df, colx="Weekly_Sales", coly="Store")

    st.write("ENTER SOME HYPERPARAMETERS")
    # Xử lý với LOF
    if selected_algorithm == "LOF":
        # Nhập hyperparameter
        n_neighbors = st.number_input("Enter n_neighbors:", min_value=1, max_value=len(filtered_df), value=20, step=1)

        # Train và predict anomaly
        fit_lof(filtered_df, features=calcul_feature, n_neighbors=n_neighbors)
        point_anomalies = filtered_df[filtered_df['Point_Anomaly'] == -1]
        
        # Plot và hiển thị kết quả
        if len(calcul_feature) == 2:
            plot_with_anomalies(df_normal=filtered_df, df_anomaly=point_anomalies, colx='Weekly_Sales', coly=calcul_feature[1], tittle="LOF")
        elif len(calcul_feature) == 3:
            plot_with_anomalies_3D(df_normal=filtered_df, df_anomaly=point_anomalies, colx='Weekly_Sales', coly=calcul_feature[1], colz=calcul_feature[2], tittle="LOF")
        else:
            plot_with_anomalies(df_normal=filtered_df, df_anomaly=point_anomalies, colx='Weekly_Sales', coly='Store', tittle="LOF")
        
        st.write(f"Particular answers ({len(point_anomalies)} answers):")
        st.dataframe(point_anomalies.sort_values(by='Weekly_Sales'))

    # Xử lý với DBSCAN
    if selected_algorithm == "DBSCAN":
        # Nhập hyperparameter
        eps = st.number_input("Enter eps:", min_value=0.01, max_value=1.00, value=0.01, step=0.01)
        min_examples = st.number_input("Enter min_examples:", min_value=1, max_value=len(filtered_df), value=20, step=1)

        # Train và predict anomaly
        fit_dbscan(filtered_df, features=calcul_feature, eps=eps, min_examples=min_examples)
        point_anomalies = filtered_df[filtered_df['Point_Anomaly'] == -1]
        
        # Plot và hiển thị kết quả
        if len(calcul_feature) == 2:
            plot_with_anomalies(df_normal=filtered_df, df_anomaly=point_anomalies, colx='Weekly_Sales', coly=calcul_feature[1], tittle="DBSCAN")
        elif len(calcul_feature) == 3:
            plot_with_anomalies_3D(df_normal=filtered_df, df_anomaly=point_anomalies, colx='Weekly_Sales', coly=calcul_feature[1], colz=calcul_feature[2], tittle="DBSCAN")
        else:
            plot_with_anomalies(df_normal=filtered_df, df_anomaly=point_anomalies, colx='Weekly_Sales', coly='Store', tittle="DBSCAN")
        st.write(f"Particular answers ({len(point_anomalies)} answers):")
        st.dataframe(point_anomalies.sort_values(by='Weekly_Sales'))

    # Xử lý với KDE 
    if selected_algorithm == "KDE":
        # Nhập hyperparameter
        list_kernel = ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']
        kernel = st.selectbox("Select a kernel", options=list_kernel)
        bandwidth = st.number_input("Enter bandwidth:", min_value=0.1, max_value=1.0, value=0.3, step=0.1)
        threshold = st.number_input("Enter threshold (%):")

        # Train và predict anomaly
        fit_kde(filtered_df, features=calcul_feature, kernel=kernel, bandwidth=bandwidth, threshold=threshold)
        point_anomalies = filtered_df[filtered_df['Point_Anomaly'] == 1]
        
        # Plot và hiển thị kết quả
        if len(calcul_feature) == 2:
            plot_with_anomalies(df_normal=filtered_df, df_anomaly=point_anomalies, colx='Weekly_Sales', coly=calcul_feature[1], tittle="KDE")
        elif len(calcul_feature) == 3:
            plot_with_anomalies_3D(df_normal=filtered_df, df_anomaly=point_anomalies, colx='Weekly_Sales', coly=calcul_feature[1], colz=calcul_feature[2], tittle="KDE")
        else:
            plot_with_anomalies(df_normal=filtered_df, df_anomaly=point_anomalies, colx='Weekly_Sales', coly='Store', tittle="KDE")
        st.write(f"Particular answers ({len(point_anomalies)} answers):")
        st.dataframe(point_anomalies.sort_values(by='Weekly_Sales'))   
