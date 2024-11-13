import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore') 
from datetime import datetime
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KernelDensity

def plot(df, colx, coly):
    # Vẽ biểu đồ scatter
    fig = px.scatter(
        df, 
        x=colx, 
        y=coly, 
        title=f"{colx} by {coly}",
        template="plotly_white",
        opacity=0.7,
    )
    # Hiển thị biểu đồ trong Streamlit
    st.plotly_chart(fig)

def plot_with_anomalies(df_normal, df_anomaly, colx, coly, tittle=''):
    fig = go.Figure()
    # Thêm dữ liệu từ DataFrame normal
    fig.add_trace(go.Scatter(
        x=df_normal[colx], 
        y=df_normal[coly], 
        mode='markers',
        marker=dict(color='green'),
        name="Normal"
    ))

    # Thêm dữ liệu từ DataFrame 2
    fig.add_trace(go.Scatter(
        x=df_anomaly[colx], 
        y=df_anomaly[coly], 
        mode='markers',
        marker=dict(color='red'),
        name="Anomaly"
    ))

    # Cài đặt tiêu đề và nhãn
    fig.update_layout(
        title=f"{colx} by {coly} with {tittle}",
        xaxis_title=colx,
        yaxis_title=coly,
        template="plotly_white"
    )

    # Hiển thị biểu đồ trong Streamlit
    st.plotly_chart(fig)

def plot_with_anomalies_3D(df_normal, df_anomaly, colx, coly, colz, tittle=''):
    fig = go.Figure()

    # Thêm dữ liệu từ DataFrame normal với trục z
    fig.add_trace(go.Scatter3d(
        x=df_normal[colx], 
        y=df_normal[coly], 
        z=df_normal[colz],  # Thêm trục z
        mode='markers',
        marker=dict(color='green'),
        name="Normal"
    ))

    # Thêm dữ liệu từ DataFrame anomaly với trục z
    fig.add_trace(go.Scatter3d(
        x=df_anomaly[colx], 
        y=df_anomaly[coly], 
        z=df_anomaly[colz],  # Thêm trục z
        mode='markers',
        marker=dict(color='red'),
        name="Anomaly"
    ))

    # Cài đặt tiêu đề và nhãn
    fig.update_layout(
        title=f"{colx}, {coly}, and {colz} with {tittle}",
        scene=dict(
            xaxis_title=colx,
            yaxis_title=coly,
            zaxis_title=colz
        ),
        template="plotly_white"
    )

    # Hiển thị biểu đồ 3D trong Streamlit
    st.plotly_chart(fig)

def fit_lof(df, features, n_neighbors=50):
    lof = LocalOutlierFactor(n_neighbors=n_neighbors)
    df['Point_Anomaly'] = lof.fit_predict(df[features])

def fit_dbscan(df, features, eps=0.01, min_examples=3):
    dbscan = DBSCAN(eps=eps, min_samples=min_examples)
    df['Point_Anomaly'] = dbscan.fit_predict(df[features])

def fit_kde(df, features, kernel='gaussian', bandwidth=0.1, threshold=3):
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
    kde.fit(df[features])
    log_density = kde.score_samples(df[features])

    threshold = np.percentile(log_density, threshold)
    df['Point_Anomaly'] = (log_density < threshold).astype(int)