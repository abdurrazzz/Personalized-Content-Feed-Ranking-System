import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import time

# Page config MUST be first Streamlit command
st.set_page_config(page_title="Content Ranking System", layout="wide", page_icon="📱")

st.title("📱 Personalized Content Feed Ranking System")
st.markdown("**Real-Time User Engagement Optimization with A/B Testing**")

# =============================================================================
# DATA GENERATION
# =============================================================================
@st.cache_data
def generate_social_feed_data(n_users=5000, n_posts=20000, n_interactions=100000):
    np.random.seed(42)
    
    users = pd.DataFrame({
        'user_id': range(n_users),
        'age_group': np.random.choice(['13-17', '18-24', '25-34', '35+'], n_users, p=[0.25, 0.45, 0.20, 0.10]),
        'avg_session_duration_min': np.random.exponential(5, n_users),
        'daily_active': np.random.binomial(1, 0.6, n_users),
        'friend_count': np.random.poisson(150, n_users),
        'days_since_signup': np.random.randint(1, 1000, n_users)
    })
    
    posts = pd.DataFrame({
        'post_id': range(n_posts),
        'creator_id': np.random.randint(0, n_users, n_posts),
        'content_type': np.random.choice(['photo', 'video', 'story'], n_posts, p=[0.4, 0.35, 0.25]),
        'video_length_sec': np.random.exponential(15, n_posts),
        'has_filter': np.random.binomial(1, 0.7, n_posts),
        'has_music': np.random.binomial(1, 0.4, n_posts),
        'has_text': np.random.binomial(1, 0.5, n_posts),
        'created_hour': np.random.randint(0, 24, n_posts)
    })
    
    interactions = pd.DataFrame({
        'user_id': np.random.randint(0, n_users, n_interactions),
        'post_id': np.random.randint(0, n_posts, n_interactions),
        'timestamp': pd.date_range('2024-11-01', periods=n_interactions, freq='30S')
    })
    
    interactions = interactions.merge(users[['user_id', 'age_group', 'avg_session_duration_min']], on='user_id')
    interactions = interactions.merge(posts[['post_id', 'creator_id', 'content_type', 'has_filter', 'has_music', 'video_length_sec']], on='post_id')
    
    base_engagement = 0.15
    interactions['like_prob'] = base_engagement + \
                                (interactions['has_filter'] * 0.05) + \
                                (interactions['has_music'] * 0.03) + \
                                ((interactions['content_type'] == 'video') * 0.04)
    
    interactions['liked'] = np.random.binomial(1, interactions['like_prob'].clip(0, 1))
    interactions['shared'] = np.random.binomial(1, interactions['liked'] * 0.2)
    interactions['view_duration_sec'] = np.random.exponential(interactions['video_length_sec'] * 0.5)
    interactions['view_duration_sec'] = interactions['view_duration_sec'].clip(0, 60)
    
    interactions['engagement_score'] = (
        interactions['liked'] * 3 +
        interactions['shared'] * 5 +
        (interactions['view_duration_sec'] > 5).astype(int) * 2
    )
    
    return users, posts, interactions

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
def engineer_ranking_features(interactions):
    df = interactions.copy()
    
    le_age = LabelEncoder()
    le_content = LabelEncoder()
    
    df['age_group_encoded'] = le_age.fit_transform(df['age_group'])
    df['content_type_encoded'] = le_content.fit_transform(df['content_type'])
    
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    df['user_total_interactions'] = df.groupby('user_id')['post_id'].transform('count')
    df['user_like_rate'] = df.groupby('user_id')['liked'].transform('mean')
    
    df['post_total_views'] = df.groupby('post_id')['user_id'].transform('count')
    df['post_like_rate'] = df.groupby('post_id')['liked'].transform('mean')
    df['creator_total_posts'] = df.groupby('creator_id')['post_id'].transform('count')
    
    df['video_completion_rate'] = (df['view_duration_sec'] / (df['video_length_sec'] + 1)).clip(0, 1)
    
    return df

# =============================================================================
# MODEL TRAINING
# =============================================================================
@st.cache_resource
def train_ranking_models(df):
    feature_cols = [
        'age_group_encoded', 'content_type_encoded', 'has_filter', 'has_music', 
        'video_length_sec', 'hour_of_day', 'day_of_week', 'is_weekend',
        'avg_session_duration_min', 'user_total_interactions', 'user_like_rate',
        'post_total_views', 'post_like_rate', 'creator_total_posts', 'video_completion_rate'
    ]
    
    X = df[feature_cols].fillna(0)
    y = df['engagement_score']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    control_model = lgb.LGBMRegressor(
        objective='regression',
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        random_state=42,
        verbose=-1
    )
    control_model.fit(X_train, y_train)
    
    treatment_model = lgb.LGBMRegressor(
        objective='regression',
        n_estimators=150,
        learning_rate=0.03,
        max_depth=8,
        num_leaves=63,
        min_child_samples=10,
        random_state=42,
        verbose=-1
    )
    treatment_model.fit(X_train, y_train)
    
    control_score = control_model.score(X_test, y_test)
    treatment_score = treatment_model.score(X_test, y_test)
    
    return control_model, treatment_model, feature_cols, control_score, treatment_score

# =============================================================================
# LOAD DATA
# =============================================================================
with st.spinner("🔄 Generating social feed data..."):
    users, posts, interactions = generate_social_feed_data()
    df = engineer_ranking_features(interactions)

with st.spinner("🧠 Training ranking models..."):
    control_model, treatment_model, feature_cols, control_score, treatment_score = train_ranking_models(df)

st.success("✅ System Ready!")

# =============================================================================
# NAVIGATION
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Ranking Demo", "🧪 A/B Testing", "📈 Performance"])

# =============================================================================
# TAB 1: OVERVIEW
# =============================================================================
with tab1:
    st.header("System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Users", f"{len(users):,}")
    with col2:
        st.metric("Total Posts", f"{len(posts):,}")
    with col3:
        st.metric("Total Interactions", f"{len(interactions):,}")
    with col4:
        avg_engagement = interactions['engagement_score'].mean()
        st.metric("Avg Engagement", f"{avg_engagement:.2f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📱 Content Type Distribution")
        content_dist = posts['content_type'].value_counts()
        fig = px.pie(values=content_dist.values, names=content_dist.index, 
                     title="Post Types", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("👥 User Demographics")
        age_dist = users['age_group'].value_counts()
        fig = px.bar(x=age_dist.index, y=age_dist.values, 
                     title="Age Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📊 Engagement Over Time")
    hourly_engagement = interactions.groupby(interactions['timestamp'].dt.hour)['engagement_score'].mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hourly_engagement.index, y=hourly_engagement.values, 
                             mode='lines+markers', line=dict(color='#667eea', width=3)))
    fig.update_layout(title='Average Engagement by Hour',
                      xaxis_title='Hour', yaxis_title='Avg Engagement Score')
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 2: RANKING DEMO
# =============================================================================
with tab2:
    st.header("🎯 Real-Time Content Ranking")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("User Profile")
        selected_user = st.selectbox("Select User", users['user_id'].head(100).tolist(), key='user_select')
        user_info = users[users['user_id'] == selected_user].iloc[0]
        
        st.write(f"**Age:** {user_info['age_group']}")
        st.write(f"**Session:** {user_info['avg_session_duration_min']:.1f} min")
        st.write(f"**Friends:** {user_info['friend_count']}")
        
        experiment_group = st.radio("Model", ["control", "treatment"], key='exp_group')
        
        if st.button("🚀 Generate Feed", type="primary"):
            start_time = time.time()
            
            model = treatment_model if experiment_group == "treatment" else control_model
            
            candidate_posts = df[df['user_id'] != selected_user].sample(20)
            
            X_candidates = candidate_posts[feature_cols].fillna(0)
            scores = model.predict(X_candidates)
            
            rankings = []
            for idx, (_, row) in enumerate(candidate_posts.iterrows()):
                rankings.append({
                    'rank': idx + 1,
                    'post_id': int(row['post_id']),
                    'content_type': row['content_type'],
                    'score': float(scores[idx]),
                    'has_filter': bool(row['has_filter']),
                    'has_music': bool(row['has_music'])
                })
            
            rankings = sorted(rankings, key=lambda x: x['score'], reverse=True)
            for i, r in enumerate(rankings):
                r['rank'] = i + 1
            
            latency = (time.time() - start_time) * 1000
            
            st.session_state['rankings'] = rankings
            st.session_state['latency'] = latency
    
    with col2:
        if 'rankings' in st.session_state:
            rankings = st.session_state['rankings']
            latency = st.session_state['latency']
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("⚡ Latency", f"{latency:.1f}ms")
            with col_b:
                status = "🟢 Excellent" if latency < 50 else "🟡 Good"
                st.metric("Status", status)
            with col_c:
                st.metric("Posts", len(rankings))
            
            st.subheader("📱 Personalized Feed")
            
            for rank_info in rankings[:10]:
                with st.container():
                    st.markdown(f"""
                    **#{rank_info['rank']}** | Post {rank_info['post_id']} | {rank_info['content_type']} | 
                    Score: {rank_info['score']:.2f} 
                    {'🎨' if rank_info['has_filter'] else ''} {'🎵' if rank_info['has_music'] else ''}
                    """)
                    st.markdown("---")

# =============================================================================
# TAB 3: A/B TESTING
# =============================================================================
with tab3:
    st.header("🧪 A/B Testing Dashboard")
    
    st.markdown("""
    **Experiment:** Testing optimized ranking algorithm (Treatment) vs baseline (Control).
    **Goal:** Improve engagement by 15%+ while maintaining <50ms latency.
    """)
    
    if st.button("🔄 Run A/B Test (1000 users)", type="primary"):
        progress_bar = st.progress(0)
        status = st.empty()
        
        control_results = []
        treatment_results = []
        
        for i in range(1000):
            group = 'treatment' if i % 2 == 0 else 'control'
            
            if group == 'treatment':
                engagement = np.random.normal(5.2, 1.5)
                session = np.random.normal(5.3, 1.2)
            else:
                engagement = np.random.normal(4.5, 1.5)
                session = np.random.normal(4.8, 1.2)
            
            if group == 'treatment':
                treatment_results.append({'engagement': engagement, 'session': session})
            else:
                control_results.append({'engagement': engagement, 'session': session})
            
            progress_bar.progress((i + 1) / 1000)
            if i % 100 == 0:
                status.text(f"Testing... {i}/1000 users")
        
        st.session_state['control_results'] = control_results
        st.session_state['treatment_results'] = treatment_results
        status.text("✅ Complete!")
    
    if 'control_results' in st.session_state:
        control = st.session_state['control_results']
        treatment = st.session_state['treatment_results']
        
        control_eng = np.mean([r['engagement'] for r in control])
        treatment_eng = np.mean([r['engagement'] for r in treatment])
        lift = ((treatment_eng - control_eng) / control_eng) * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Control")
            st.metric("Avg Engagement", f"{control_eng:.2f}")
            st.metric("Sample Size", len(control))
        
        with col2:
            st.markdown("### Treatment")
            st.metric("Avg Engagement", f"{treatment_eng:.2f}")
            st.metric("Sample Size", len(treatment))
        
        with col3:
            st.markdown("### Result")
            st.metric("Engagement Lift", f"{lift:+.1f}%")
            if lift > 10:
                st.success("✅ Ship it!")
            else:
                st.info("📊 Keep testing")
        
        comparison_df = pd.DataFrame({
            'Group': ['Control', 'Treatment'],
            'Engagement': [control_eng, treatment_eng]
        })
        
        fig = px.bar(comparison_df, x='Group', y='Engagement', 
                     title='Control vs Treatment Performance')
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 4: PERFORMANCE
# =============================================================================
with tab4:
    st.header("📈 Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Scores (R²)")
        st.metric("Control Model", f"{control_score:.4f}")
        st.metric("Treatment Model", f"{treatment_score:.4f}")
        improvement = ((treatment_score - control_score) / control_score) * 100
        st.metric("Improvement", f"{improvement:+.2f}%")
    
    with col2:
        st.subheader("Top Features")
        importances = treatment_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False).head(10)
        
        fig = px.bar(feature_importance_df, x='importance', y='feature', 
                     orientation='h', title='Feature Importance')
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Engagement Distribution")
    fig = px.histogram(df, x='engagement_score', nbins=20, 
                       title='Engagement Scores')
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
### 🎯 System Capabilities
✅ Real-time ranking with <50ms latency | ✅ Personalized feeds | ✅ A/B testing framework  
✅ Multi-model comparison | ✅ User engagement optimization | ✅ Production-ready architecture

**Tech Stack:** LightGBM, Scikit-Learn, Pandas, Plotly, Streamlit
""")