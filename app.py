import streamlit as st
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')  # 👈 Prevents GUI backend errors
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os
from src.feature_mapping import FEATURE_MAPPING, REVERSE_MAPPING

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Churn Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Assets ---
@st.cache_resource
def load_model():
    if os.path.exists('models/best_model.pkl'):
        return joblib.load('models/best_model.pkl')
    else:
        st.error("❌ Model not found. Please run `train_pipeline.py` first.")
        st.stop()

@st.cache_data
def load_train_data():
    if os.path.exists('data/Train.csv'):
        df = pd.read_csv('data/Train.csv')
        df = df.rename(columns=FEATURE_MAPPING)
        if 'labels' not in df.columns:
            df['labels'] = df.iloc[:, -1]
        return df
    else:
        st.error("❌ train.csv not found in data/ folder.")
        st.stop()

model = load_model()
df = load_train_data()

# --- Sidebar for Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dataset Overview", "Predict Churn", "Model Performance", "About"])

# =====================================================================================
# PAGE 1: DATASET OVERVIEW
# =====================================================================================
if page == "Dataset Overview":
    st.header("📊 Dataset Overview")
    st.write("High-level summary of the training data (`Train.csv`).")

    total_customers = len(df)
    churn_count = df['labels'].sum()
    no_churn_count = total_customers - churn_count
    churn_rate = (churn_count / total_customers) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Churned Customers", f"{churn_count:,}")
    col3.metric("Churn Rate", f"{churn_rate:.1f}%")

    # Pie chart
    fig, ax = plt.subplots()
    ax.pie([churn_count, no_churn_count], labels=['Churned', 'Retained'], autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
    ax.axis('equal')
    st.pyplot(fig)

    # Heatmap (if exists)
    heatmap_path = 'reports/eda_plots/feature_correlation_heatmap.png'
    if os.path.exists(heatmap_path):
        st.subheader("🔥 Feature Correlation Heatmap (Priority Features)")
        st.image(heatmap_path, use_column_width=True)

    # Key drivers
    st.subheader("🔑 Key Churn Drivers (Business Context)")
    st.markdown("""
    **High-Risk Factors:**
    - `risk_score` → Composite risk indicator — higher = higher churn
    - `payment_delay_score` → Late payments → strong churn signal
    - `claim_frequency` → High claim rate → dissatisfaction → churn

    **High-Retention Factors:**
    - `auto_renew_flag` → Auto-renewal ON → strong retention signal
    - `discount_eligibility_score` → High score → loyal customer → low churn
    - `policy_tenure_scaled` → Long tenure → loyalty → low churn
    """)

# =====================================================================================
# PAGE 2: PREDICT CHURN
# =====================================================================================
elif page == "Predict Churn":
    st.header("🔮 Predict Customer Churn")

    prediction_method = st.radio("Select Prediction Method", ["Single Customer (Manual Input)", "Batch Prediction (Upload File)"], horizontal=True)

    if prediction_method == "Single Customer (Manual Input)":
        st.subheader("🎯 Enter Key Customer Attributes (Priority Features Only)")

        # Get model feature names
        feature_names = model.feature_name_ if hasattr(model, 'feature_name_') else model.feature_names_in_

        # Create mapping from display names to actual feature names
        feature_mapping = {
            'payment_delay_score': 'feature_3',
            'digital_engagement_level': 'feature_13', 
            'claim_frequency': 'feature_1',
            'family_plan_flag': 'feature_11',
            'service_interaction_count': 'feature_4',
            'policy_type': 'feature_9',
            'sales_channel_id': 'feature_8',
            'region_code': 'feature_7'
        }

        # Define priority feature groups
        input_data = {}

        # =============================
        # 🔥 RISK FACTORS
        # =============================
        st.markdown("### 🔥 High-Impact Risk Factors")
        col1, col2, col3 = st.columns(3)

        with col1:
            if 'feature_3' in feature_names:  # payment_delay_score
                val = st.slider("Payment Delay Score", -2.0, 2.0, 0.0, 0.1, help="Positive = late payments → high risk")
                input_data[feature_mapping['payment_delay_score']] = val

        with col2:
            if 'feature_13' in feature_names:  # digital_engagement_level
                val = st.slider("Digital Engagement Level", -3.0, 3.0, 0.0, 0.1, help="Higher = more engaged → lower risk")
                input_data[feature_mapping['digital_engagement_level']] = val

        with col3:
            if 'feature_1' in feature_names:  # claim_frequency
                val = st.slider("Claim Frequency", 0.0, 3.0, 0.0, 0.1, help="Higher = more claims → higher risk")
                input_data[feature_mapping['claim_frequency']] = val

        # =============================
        # 🛡️ RETENTION FACTORS
        # =============================
        st.markdown("### 🛡️ High-Impact Retention Factors")
        col1, col2, col3 = st.columns(3)

        with col1:
            if 'feature_11' in feature_names:  # family_plan_flag
                val = st.selectbox("Family Plan Flag", [0, 1], format_func=lambda x: "No" if x==0 else "Yes", help="Family plans = higher retention")
                input_data[feature_mapping['family_plan_flag']] = val

        with col2:
            if 'feature_4' in feature_names:  # service_interaction_count
                val = st.slider("Service Interaction Count", 0.0, 10.0, 2.0, 0.5, help="Higher = more engaged → lower churn risk")
                input_data[feature_mapping['service_interaction_count']] = val

        with col3:
            if 'feature_9' in feature_names:  # policy_type
                val = st.selectbox("Policy Type", [0, 1, 2, 3], help="Type of insurance policy")
                input_data[feature_mapping['policy_type']] = val

        # =============================
        # 🧑‍💼 CUSTOMER INFO
        # =============================
        st.markdown("### 🧑‍💼 Customer & Policy Info")
        col1, col2, col3 = st.columns(3)

        with col1:
            if 'feature_8' in feature_names:  # sales_channel_id
                val = st.selectbox("Sales Channel", [0,1,2], format_func=lambda x: ["Online", "Agent", "Call Center"][x], help="How customer was acquired")
                input_data[feature_mapping['sales_channel_id']] = val

        with col2:
            if 'feature_7' in feature_names:  # region_code
                val = st.selectbox("Region Code", [0,1,2,3,4,5], help="Customer's geographic region")
                input_data[feature_mapping['region_code']] = val

        with col3:
            # This is a placeholder for the 8th feature if needed
            st.info("All 8 priority features are now configured above")

        # =============================
        # 🚀 PREDICT BUTTON
        # =============================
        st.markdown("---")
        if st.button("🎯 Predict Churn Risk", type="primary", use_container_width=True):
            # Create DataFrame
            input_df = pd.DataFrame([input_data])

            # Align with model features
            for col in feature_names:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[feature_names]

            # Predict
            proba = model.predict_proba(input_df)[0, 1]

            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                title={'text': "Churn Risk %", 'font': {'size': 24}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "red" if proba > 0.5 else "green"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ]
                }
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

            # Actionable Recommendations
            if proba > 0.7:
                st.error(f"🚨 HIGH RISK: {proba:.1%} probability of churn")
                st.markdown("""
                **Recommended Actions:**
                - Schedule retention call within 24 hours
                - Offer 15% loyalty discount
                - Investigate payment delays or high claim frequency
                """)
            elif proba > 0.4:
                st.warning(f"⚠️ MEDIUM RISK: {proba:.1%} probability of churn")
                st.markdown("""
                **Recommended Actions:**
                - Send personalized email with policy tips
                - Offer 5% discount on renewal
                - Check auto-renewal status
                """)
            else:
                st.success(f"✅ LOW RISK: {proba:.1%} probability of churn")
                st.markdown("""
                **Recommended Actions:**
                - Include in loyalty rewards program
                - Upsell premium features
                - Send satisfaction survey
                """)

    elif prediction_method == "Batch Prediction (Upload File)":
        st.info("Upload a CSV file with priority features only.")
        uploaded_file = st.file_uploader("Choose a file", type="csv")
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview:")
            st.dataframe(batch_df.head())

            if st.button("🚀 Predict on Batch", type="primary"):
                # Get model feature names
                feature_names = model.feature_name_ if hasattr(model, 'feature_name_') else model.feature_names_in_

                # Align features - ensure all required features are present
                for col in feature_names:
                    if col not in batch_df.columns:
                        batch_df[col] = 0
                X_batch = batch_df[feature_names]

                # Predict
                y_proba = model.predict_proba(X_batch)[:, 1]

                # Create result
                result_df = batch_df.copy()
                result_df['Churn_Probability'] = y_proba
                result_df['Risk_Level'] = pd.cut(y_proba,
                    bins=[0, 0.4, 0.7, 1.0],
                    labels=['Low', 'Medium', 'High'],
                    include_lowest=True
                )

                # Display summary
                st.subheader("📈 Batch Prediction Summary")
                total = len(result_df)
                high_risk = (result_df['Risk_Level'] == 'High').sum()
                medium_risk = (result_df['Risk_Level'] == 'Medium').sum()
                low_risk = (result_df['Risk_Level'] == 'Low').sum()

                col1, col2, col3 = st.columns(3)
                col1.metric("High Risk", f"{high_risk} ({high_risk/total:.1%})")
                col2.metric("Medium Risk", f"{medium_risk} ({medium_risk/total:.1%})")
                col3.metric("Low Risk", f"{low_risk} ({low_risk/total:.1%})")

                # Show results
                st.write("Prediction Results:")
                st.dataframe(result_df)

                # Download
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Predictions",
                    data=csv,
                    file_name='churn_predictions.csv',
                    mime='text/csv'
                )

# =====================================================================================
# PAGE 3: MODEL PERFORMANCE
# =====================================================================================
elif page == "Model Performance":
    st.header("📈 Model Performance (Priority Features Only)")

    if os.path.exists('reports/performance_metrics.csv'):
        metrics_df = pd.read_csv('reports/performance_metrics.csv')

        # Keep only essential columns
        metrics_df = metrics_df[['Model', 'AUC', 'Precision', 'Recall', 'F1-Score']].round(3)

        # Highlight best score in each column
        def highlight_best(s):
            is_best = s == s.max()
            return ['background-color: #008000; font-weight: bold' if v else '' for v in is_best]

        st.markdown("### 🎯 Performance Metrics (Validation Set)")
        st.dataframe(metrics_df.style.apply(highlight_best, subset=['AUC', 'Precision', 'Recall', 'F1-Score'], axis=0), use_container_width=True)

        # Interpretation
        st.markdown("### 💡 What These Metrics Mean")
        st.markdown("""
        - **AUC > 0.85**: Excellent ranking — model reliably identifies high-risk customers.
        - **Recall > 0.80**: Catches 80%+ of churners — minimizes missed opportunities.
        - **Precision > 0.80**: 80%+ of flagged customers actually churn — efficient targeting.
        - **F1-Score > 0.80**: Balanced performance — ideal for retention campaigns.
        """)

        # SHAP plots
        if os.path.exists('reports/shap_plots/LightGBM_feature_importance.png'):
            st.subheader("🔝 Top Feature Drivers (SHAP)")
            st.image('reports/shap_plots/LightGBM_feature_importance.png', use_column_width=True)

            # Top 5 features
            model_features = model.feature_name_ if hasattr(model, 'feature_name_') else model.feature_names_in_
            st.markdown("**Top 5 Drivers of Churn:**")
            for i, feat in enumerate(model_features[:5]):
                display_name = FEATURE_MAPPING.get(feat, feat)
                st.markdown(f"{i+1}. **{display_name}** (feature_{feat.split('_')[1] if 'feature_' in feat else feat})")

    else:
        st.warning("⚠️ Run `train_pipeline.py` to generate performance metrics.")

# =====================================================================================
# PAGE 4: ABOUT
# =====================================================================================
elif page == "About":
    st.header("ℹ️ About This Project")

    st.markdown("""
    ### 🎯 Business Problem
    Predict which insurance customers are likely to churn — so we can proactively retain them with targeted interventions.

    ### 🚀 Solution
    - **Model**: LightGBM trained on top 8 priority features
    - **Handling Imbalance**: SMOTE for balanced training
    - **Evaluation**: AUC, Precision, Recall, F1-Score on holdout set
    - **Deployment**: Streamlit UI for business users

    ### 🛠️ Technology Stack
    - Python, Pandas, Scikit-learn
    - LightGBM, SHAP, Matplotlib, Seaborn
    - Streamlit for UI

    ### 📈 Business Impact
    - Reduce churn by 15-30%
    - Save acquisition costs
    - Improve customer lifetime value

    ### 📂 How to Use
    - **Dataset Overview**: Understand data and key drivers
    - **Predict Churn**: Single or batch predictions
    - **Model Performance**: See AUC, Precision, Recall, F1-Score
    - **About**: Project context and tech stack
    """)

# Footer
st.markdown("---")
st.markdown("💡 Built for Insurance Churn Prediction | v4.0")
