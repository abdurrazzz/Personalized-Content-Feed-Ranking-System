# Personalized Content Feed Ranking System

A production-ready machine learning system for real-time social media content ranking with integrated A/B testing capabilities. This system demonstrates how modern social platforms optimize user engagement through intelligent feed curation.

## 🎯 Overview

This application simulates a social media content ranking system that personalizes user feeds based on engagement predictions. It showcases the complete ML lifecycle from data generation to model deployment and experimentation.

## ✨ Key Features

### 🤖 Machine Learning Models
- **Dual Model Architecture**: Control (baseline) and Treatment (optimized) models for comparison
- **LightGBM Regressors**: Fast, efficient gradient boosting for engagement prediction
- **Real-time Scoring**: Sub-50ms latency for production-grade performance
- **Feature Engineering**: 15+ engineered features including temporal, behavioral, and content signals

### 📊 Interactive Dashboards
Four comprehensive tabs providing complete system visibility:

1. **Overview Dashboard**
   - System-wide metrics and KPIs
   - Content type distribution analysis
   - User demographic breakdowns
   - Temporal engagement patterns

2. **Ranking Demo**
   - Live feed generation for individual users
   - Model selection (Control vs Treatment)
   - Real-time latency monitoring
   - Personalized content scoring with visual indicators

3. **A/B Testing Framework**
   - Simulated experiments with 1,000 users
   - Statistical comparison of model variants
   - Engagement lift calculations
   - Visual performance comparisons

4. **Performance Analytics**
   - Model R² scores and improvement metrics
   - Feature importance rankings
   - Engagement distribution analysis
   - Production readiness indicators

## 🔧 Technical Architecture

### Data Generation
- **5,000 users** with realistic demographic and behavioral profiles
- **20,000 posts** with varied content types (photos, videos, stories)
- **100,000 interactions** simulating user engagement patterns
- Probabilistic engagement modeling based on content features

### Feature Engineering
The system generates rich feature sets including:

**User Features:**
- Age group encoding
- Average session duration
- Historical interaction counts
- Personal engagement rates

**Content Features:**
- Content type (photo/video/story)
- Video length and completion rates
- Creative elements (filters, music, text)
- Creator popularity metrics

**Temporal Features:**
- Hour of day and day of week
- Weekend vs weekday patterns
- Time-based engagement signals

**Interaction Features:**
- Post-level engagement statistics
- View duration analysis
- User-content affinity scores

### Model Training

**Control Model (Baseline):**
- 100 estimators, learning rate 0.05
- Max depth 6, 31 leaves
- Simpler architecture for baseline performance

**Treatment Model (Optimized):**
- 150 estimators, learning rate 0.03
- Max depth 8, 63 leaves
- Enhanced complexity for improved predictions

Both models use regression objectives to predict engagement scores based on:
- Likes (weight: 3)
- Shares (weight: 5)
- View duration >5s (weight: 2)

## 🚀 Getting Started

### Prerequisites
```bash
pip install streamlit pandas numpy plotly scikit-learn lightgbm
```

### Installation
```bash
# Clone or download the application
# Navigate to the directory containing the script

# Run the application
streamlit run app.py
```

### Usage

1. **Launch the app** - The system automatically generates synthetic data and trains models
2. **Explore the Overview** - Understand the dataset characteristics
3. **Test Ranking** - Select a user and generate personalized feeds
4. **Run A/B Tests** - Compare model performance with simulated experiments
5. **Analyze Performance** - Review model metrics and feature importance

## 📈 Performance Metrics

### Latency
- Target: <50ms for real-time ranking
- Typical: 10-30ms per feed generation
- Status indicators: 🟢 Excellent (<50ms) | 🟡 Good (50-100ms)

### Engagement
- Control baseline: ~4.5 engagement score
- Treatment target: 15%+ lift over control
- Success criteria: Statistically significant improvement

### Model Accuracy
- R² scores tracked for both models
- Feature importance analysis for interpretability
- Continuous monitoring for production readiness

## 🧪 A/B Testing Methodology

The system implements a rigorous experimentation framework:

1. **Random Assignment**: Users split 50/50 between control and treatment
2. **Sample Size**: 1,000 users per experiment for statistical power
3. **Metrics Tracked**: 
   - Average engagement score
   - Session duration
   - Engagement lift percentage
4. **Decision Framework**: 
   - >10% lift → Ship to production
   - <10% lift → Continue iterating

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **ML Framework** | LightGBM | Fast gradient boosting for ranking |
| **Data Processing** | Pandas, NumPy | Data manipulation and feature engineering |
| **Visualization** | Plotly | Interactive charts and dashboards |
| **UI Framework** | Streamlit | Rapid dashboard development |
| **Model Validation** | Scikit-learn | Train/test splits and scoring |

## 📊 Data Schema

### Users Table
- `user_id`: Unique identifier
- `age_group`: Demographic segment
- `avg_session_duration_min`: Engagement behavior
- `daily_active`: Activity status
- `friend_count`: Social graph size
- `days_since_signup`: User tenure

### Posts Table
- `post_id`: Unique identifier
- `creator_id`: Content author
- `content_type`: Media format
- `video_length_sec`: Duration for videos
- `has_filter`, `has_music`, `has_text`: Content features
- `created_hour`: Publication time

### Interactions Table
- `user_id`, `post_id`: Foreign keys
- `timestamp`: Interaction time
- `liked`, `shared`: Engagement actions
- `view_duration_sec`: Time spent
- `engagement_score`: Composite metric

## 🎓 Use Cases

This system demonstrates concepts relevant to:
- Social media feed optimization
- E-commerce product ranking
- Content recommendation systems
- News feed personalization
- Video streaming recommendations
- Advertisement targeting

## 🔍 Key Insights

The system reveals important patterns in content ranking:
- **Temporal Effects**: Engagement varies significantly by hour and day
- **Content Preferences**: Different demographics prefer different content types
- **Feature Impact**: Filters and music boost engagement by 5-8%
- **Video Performance**: Video content shows 4% higher engagement than static posts
- **Completion Rates**: View duration is a strong signal for content quality

## 🚦 Production Considerations

For real-world deployment, consider:
- **Scalability**: Batch prediction for millions of users
- **Model Updates**: Regular retraining with fresh data
- **Feature Store**: Centralized feature computation and caching
- **Monitoring**: Real-time performance tracking and alerting
- **Fairness**: Bias detection and mitigation strategies
- **Privacy**: User data protection and compliance

## 📝 Future Enhancements

Potential improvements include:
- Deep learning models (neural networks)
- Multi-objective optimization (engagement + diversity)
- Contextual bandits for online learning
- Graph-based features (social connections)
- Real-time feature updates
- Advanced A/B testing (multi-armed bandits)
- Explainable AI for transparency

## 📄 License

This is an educational demonstration system. Adapt for your specific use case.

## 🤝 Contributing

This is a standalone demonstration. Feel free to fork and customize for your needs.

---

*For questions or feedback, consult the integrated documentation and explore the interactive dashboards.*
