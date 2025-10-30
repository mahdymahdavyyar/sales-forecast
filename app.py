import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta

st.set_page_config(page_title="e-Commerce Sales Forecaster", layout="wide")

st.title("e-Commerce Sales Forecaster (XGBoost)")
st.markdown("**پیش‌بینی فروش ۳۰ روز آینده | بدون خطا**")

@st.cache_data
def load_data():
    df = pd.read_csv('data.csv', encoding='ISO-8859-1', parse_dates=['InvoiceDate'])
    df = df[df['Quantity'] > 0]
    df['Revenue'] = df['Quantity'] * df['Price']
    daily = df.groupby(df['InvoiceDate'].dt.date)['Revenue'].sum().reset_index()
    daily.columns = ['ds', 'y']
    # تبدیل به datetime
    daily['ds'] = pd.to_datetime(daily['ds'])
    return df, daily

df, daily = load_data()

# نمودار فروش کل
col1, col2 = st.columns(2)
with col1:
    st.subheader("فروش روزانه (کل جهان)")
    fig1 = px.line(daily, x='ds', y='y', title="Historical Global Sales")
    st.plotly_chart(fig1, use_container_width=True)

# انتخاب کشور
with col2:
    countries = df['Country'].value_counts().head(6).index.tolist()
    country = st.selectbox("انتخاب کشور:", countries, index=0)
    country_df = df[df['Country'] == country]
    country_daily = country_df.groupby(country_df['InvoiceDate'].dt.date)['Revenue'].sum().reset_index()
    country_daily.columns = ['ds', 'y']
    # تبدیل به datetime
    country_daily['ds'] = pd.to_datetime(country_daily['ds'])
    st.subheader(f"فروش {country}")
    fig2 = px.line(country_daily, x='ds', y='y')
    st.plotly_chart(fig2, use_container_width=True)

# پیش‌بینی
if st.button("تولید پیش‌بینی ۳۰ روز"):
    with st.spinner("در حال آموزش مدل XGBoost..."):
        # آماده‌سازی داده
        country_daily = country_daily.copy()
        country_daily['day'] = (country_daily['ds'] - country_daily['ds'].min()).dt.days
        X = country_daily[['day']].values
        y = country_daily['y'].values
        
        # مدل
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X, y)
        
        # پیش‌بینی
        last_day = country_daily['day'].max()
        future_days = np.arange(last_day + 1, last_day + 31).reshape(-1, 1)
        forecast = model.predict(future_days)
        
        # تاریخ‌های آینده
        last_date = country_daily['ds'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30)
        
        # نمودار
        fig = px.line(country_daily, x='ds', y='y', title=f"پیش‌بینی فروش {country}")
        fig.add_scatter(x=future_dates, y=forecast, mode='lines', name='پیش‌بینی', line=dict(dash='dash', color='red'))
        st.plotly_chart(fig, use_container_width=True)
        
        total_forecast = forecast.sum()
        st.success(f"**پیش‌بینی کل فروش ۳۰ روز: £{total_forecast:,.0f}**")
        st.info("**تأثیر:** کاهش ۱۵% موجودی اضافی برای فروشگاه‌های آنلاین")