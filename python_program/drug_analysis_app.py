import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="Drug Analysis Q&A", layout="wide")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('realistic_drug_labels_side_effects.csv')
        
        # Basic preprocessing
        df['side_effects_list'] = df['side_effects'].apply(lambda x: [s.strip() for s in str(x).split(',') if s.strip()] if pd.notna(x) else [])
        df['num_side_effects'] = df['side_effects_list'].apply(len)
        df['market_success'] = ((df['approval_status'] == 'Approved') & (df['price_usd'] > df['price_usd'].median())).astype(int)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def get_analysis_response(question, df):
    question = question.lower()
    
    if any(word in question for word in ['approval', 'approved', 'reject']):
        approval_stats = df['approval_status'].value_counts()
        class_approval = df.groupby('drug_class')['approval_status'].apply(lambda x: (x=='Approved').mean()).sort_values(ascending=False)
        
        response = f"""**Drug Approval Analysis:**
        
📊 **Overall Approval Rates:**
- Approved: {approval_stats.get('Approved', 0)} drugs ({approval_stats.get('Approved', 0)/len(df)*100:.1f}%)
- Pending: {approval_stats.get('Pending', 0)} drugs ({approval_stats.get('Pending', 0)/len(df)*100:.1f}%)
- Rejected: {approval_stats.get('Rejected', 0)} drugs ({approval_stats.get('Rejected', 0)/len(df)*100:.1f}%)

🏆 **Best Drug Classes for Approval:**
{class_approval.head().to_string()}

💡 **Key Insights:**
- {class_approval.index[0]} has the highest approval rate at {class_approval.iloc[0]:.3f}
- Overall approval rate is {(df['approval_status']=='Approved').mean():.3f}
"""
        return response, class_approval
        
    elif any(word in question for word in ['price', 'cost', 'expensive']):
        price_stats = df['price_usd'].describe()
        class_price = df.groupby('drug_class')['price_usd'].mean().sort_values(ascending=False)
        
        response = f"""**Drug Pricing Analysis:**
        
💰 **Price Statistics:**
- Average Price: ${price_stats['mean']:.2f}
- Median Price: ${price_stats['50%']:.2f}
- Price Range: ${price_stats['min']:.2f} - ${price_stats['max']:.2f}

💸 **Most Expensive Drug Classes:**
{class_price.head().to_string()}

📈 **Price Insights:**
- {class_price.index[0]} is the most expensive class at ${class_price.iloc[0]:.2f}
- Price correlation with dosage: {df['dosage_mg'].corr(df['price_usd']):.3f}
"""
        return response, class_price
        
    elif any(word in question for word in ['side effect', 'adverse', 'severity']):
        all_effects = []
        for effects in df['side_effects_list']:
            all_effects.extend(effects)
        effect_counts = Counter(all_effects)
        severity_dist = df['side_effect_severity'].value_counts()
        
        response = f"""**Side Effect Analysis:**
        
⚠️ **Side Effect Severity Distribution:**
{severity_dist.to_string()}

🔍 **Most Common Side Effects:**
{dict(effect_counts.most_common(10))}

📊 **Key Findings:**
- Average side effects per drug: {df['num_side_effects'].mean():.2f}
- Most common side effect: {effect_counts.most_common(1)[0][0]} ({effect_counts.most_common(1)[0][1]} drugs)
- {severity_dist.index[0]} severity is most common ({severity_dist.iloc[0]} drugs)
"""
        return response, effect_counts
        
    elif any(word in question for word in ['manufacturer', 'company']):
        mfg_stats = df.groupby('manufacturer').agg({
            'approval_status': lambda x: (x=='Approved').mean(),
            'price_usd': 'mean',
            'drug_name': 'count'
        }).round(3)
        mfg_stats.columns = ['Approval_Rate', 'Avg_Price', 'Drug_Count']
        mfg_stats = mfg_stats.sort_values('Approval_Rate', ascending=False)
        
        # Get drugs by manufacturer
        mfg_drugs = df.groupby('manufacturer')['drug_name'].apply(list).to_dict()
        drugs_summary = "\n".join([f"- **{mfg}**: {', '.join(drugs[:3])}{'...' if len(drugs) > 3 else ''}" for mfg, drugs in list(mfg_drugs.items())[:5]])
        
        response = f"""**Manufacturer Analysis:**
        
🏭 **Manufacturer Performance:**
{mfg_stats.to_string()}

💊 **Drugs by Top Manufacturers:**
{drugs_summary}

🎯 **Top Insights:**
- Best approval rate: {mfg_stats.index[0]} ({mfg_stats.iloc[0]['Approval_Rate']:.3f})
- Most drugs produced: {mfg_stats.sort_values('Drug_Count', ascending=False).index[0]} ({mfg_stats.sort_values('Drug_Count', ascending=False).iloc[0]['Drug_Count']} drugs)
- Highest average price: {mfg_stats.sort_values('Avg_Price', ascending=False).index[0]} (${mfg_stats.sort_values('Avg_Price', ascending=False).iloc[0]['Avg_Price']:.2f})
"""
        return response, mfg_stats
        
    elif any(word in question for word in ['success', 'market']):
        success_rate = df['market_success'].mean()
        class_success = df.groupby('drug_class')['market_success'].mean().sort_values(ascending=False)
        
        response = f"""**Market Success Analysis:**
        
🚀 **Overall Market Success Rate:** {success_rate:.3f} ({success_rate*100:.1f}%)

🏆 **Most Successful Drug Classes:**
{class_success.head().to_string()}

💡 **Success Factors:**
- Market success = Approved + Above median price
- {class_success.index[0]} leads with {class_success.iloc[0]:.3f} success rate
- Success correlates with drug class and manufacturer quality
"""
        return response, class_success
        
    else:
        return """**Available Analysis Topics:**
        
🔍 **Ask me about:**
- **Approval rates**: "What are the drug approval rates?"
- **Pricing**: "Which drugs are most expensive?"
- **Side effects**: "What are common side effects?"
- **Manufacturers**: "Which companies perform best?"
- **Market success**: "What makes drugs successful?"
        
💡 **Example questions:**
- "Which drug class has the highest approval rate?"
- "What's the average price by manufacturer?"
- "What are the most severe side effects?"
""", None

# Main app
st.title("🏥 Drug Analysis Q&A System")
st.markdown("Ask questions about drug approval, pricing, side effects, and market success!")

# Load data
df = load_data()
if df.empty:
    st.stop()

# Sidebar with data overview
st.sidebar.header("📊 Dataset Overview")
st.sidebar.metric("Total Drugs", len(df))
st.sidebar.metric("Manufacturers", df['manufacturer'].nunique())
st.sidebar.metric("Drug Classes", df['drug_class'].nunique())
st.sidebar.metric("Approval Rate", f"{(df['approval_status']=='Approved').mean():.1%}")

# Main interface
question = st.text_input("💬 Ask your question about the drug data:", 
                        placeholder="e.g., Which drug class has the highest approval rate?")

if question:
    with st.spinner("Analyzing..."):
        response, data = get_analysis_response(question, df)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(response)
        
        with col2:
            if data is not None:
                try:
                    if isinstance(data, pd.Series) and len(data) <= 10:
                        fig = px.bar(x=data.index, y=data.values, 
                                   title="Analysis Results")
                        st.plotly_chart(fig, use_container_width=True)
                    elif isinstance(data, Counter):
                        top_10 = dict(data.most_common(10))
                        fig = px.bar(x=list(top_10.keys()), y=list(top_10.values()),
                                   title="Top 10 Results")
                        st.plotly_chart(fig, use_container_width=True)
                except:
                    st.info("Chart data available - see text analysis above")

# Quick insights section
st.header("🎯 Quick Insights")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Avg Price", f"${df['price_usd'].mean():.0f}")
    
with col2:
    st.metric("Avg Side Effects", f"{df['num_side_effects'].mean():.1f}")
    
with col3:
    try:
        best_class = df.groupby('drug_class')['approval_status'].apply(lambda x: (x=='Approved').mean()).idxmax()
        st.metric("Best Class", best_class)
    except:
        st.metric("Best Class", "N/A")
    
with col4:
    st.metric("Market Success", f"{df['market_success'].mean():.1%}")

# Interactive charts
st.header("📈 Interactive Analysis")

tab1, tab2, tab3 = st.tabs(["Approval Analysis", "Price Analysis", "Side Effects"])

with tab1:
    try:
        fig = px.sunburst(df, path=['drug_class', 'approval_status'], 
                         title="Approval Status by Drug Class")
        st.plotly_chart(fig, use_container_width=True)
    except:
        approval_by_class = df.groupby(['drug_class', 'approval_status']).size().reset_index(name='count')
        fig = px.bar(approval_by_class, x='drug_class', y='count', color='approval_status',
                    title="Approval Status by Drug Class")
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    try:
        fig = px.box(df, x='drug_class', y='price_usd', 
                    title="Price Distribution by Drug Class")
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    except:
        price_by_class = df.groupby('drug_class')['price_usd'].mean().reset_index()
        fig = px.bar(price_by_class, x='drug_class', y='price_usd',
                    title="Average Price by Drug Class")
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    try:
        severity_counts = df['side_effect_severity'].value_counts()
        fig = px.pie(values=severity_counts.values, names=severity_counts.index,
                    title="Side Effect Severity Distribution")
        st.plotly_chart(fig, use_container_width=True)
    except:
        severity_counts = df['side_effect_severity'].value_counts()
        fig = px.bar(x=severity_counts.index, y=severity_counts.values,
                    title="Side Effect Severity Distribution")
        st.plotly_chart(fig, use_container_width=True)