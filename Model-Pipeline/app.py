"""
EchoAI Streamlit Interface
Interactive UI for review sentiment analysis and response generation
"""
import streamlit as st
import pandas as pd
from datetime import datetime
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import subprocess
import requests
import time
import os

# Import your inference pipeline
from inference_pipeline import EchoAIInference

# Page configuration
st.set_page_config(
    page_title="EchoAI - Review Response Generator",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #2c3e50;
        padding: 1rem 0;
        border-bottom: 2px solid #3498db;
        margin-bottom: 2rem;
    }
    .sentiment-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .amazing { background-color: #27ae60; color: white; }
    .positive { background-color: #3498db; color: white; }
    .neutral { background-color: #95a5a6; color: white; }
    .negative { background-color: #e67e22; color: white; }
    .terrible { background-color: #e74c3c; color: white; }
    .response-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
    }
    .stats-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
    st.session_state.model_loaded = False
    st.session_state.history = []
    st.session_state.stats = {
        'total': 0,
        'amazing': 0,
        'positive': 0,
        'neutral': 0,
        'negative': 0,
        'terrible': 0
    }

def start_mlflow_ui(port=5000):
    """Start MLflow UI server if not already running"""
    try:
        # Check if MLflow is already running
        response = requests.get(f"http://localhost:{port}", timeout=1)
        if response.status_code == 200:
            return f"http://localhost:{port}"
    except:
        pass
    
    # Start MLflow UI
    try:
        # Get MLflow tracking URI from environment or use default
        mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', './mlruns')
        
        # Start MLflow UI in background
        process = subprocess.Popen(
            ['mlflow', 'ui', '--port', str(port), '--backend-store-uri', mlflow_tracking_uri],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to start
        time.sleep(3)
        
        # Verify it's running
        try:
            response = requests.get(f"http://localhost:{port}", timeout=2)
            if response.status_code == 200:
                st.session_state.mlflow_process = process
                return f"http://localhost:{port}"
        except:
            pass
            
    except Exception as e:
        st.error(f"Failed to start MLflow UI: {str(e)}")
    
    return None

def stop_mlflow_ui():
    """Stop MLflow UI server if running"""
    if st.session_state.mlflow_process:
        try:
            st.session_state.mlflow_process.terminate()
            st.session_state.mlflow_process = None
            st.session_state.mlflow_url = None
        except:
            pass

def load_model(llm_model='google/flan-t5-base', load_llm=True):
    """Load the inference pipeline"""
    with st.spinner('🔄 Loading models... This may take a moment...'):
        try:
            pipeline = EchoAIInference(llm_model=llm_model)
            pipeline.load_models(load_llm=load_llm)
            st.session_state.pipeline = pipeline
            st.session_state.model_loaded = True
            st.success('✅ Models loaded successfully!')
            return True
        except Exception as e:
            st.error(f'❌ Error loading models: {str(e)}')
            return False

def process_review(review_data):
    """Process a single review through the pipeline"""
    try:
        result = st.session_state.pipeline.process_review(
            review_data, 
            generate_response=True
        )
        
        # Update statistics
        if result['status'] == 'success':
            sentiment = result['sentiment_analysis']['sentiment']
            st.session_state.stats['total'] += 1
            st.session_state.stats[sentiment] += 1
            
            # Add to history
            history_entry = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'review': review_data.get('reviewText', ''),
                'sentiment': sentiment,
                'confidence': result['sentiment_analysis'].get('confidence', 0),
                'response': result.get('generated_response', ''),
                'metadata': review_data
            }
            st.session_state.history.insert(0, history_entry)
            
            # Limit history to last 50 entries
            if len(st.session_state.history) > 50:
                st.session_state.history = st.session_state.history[:50]
        
        return result
    except Exception as e:
        st.error(f'Error processing review: {str(e)}')
        return None

def display_sentiment_result(result):
    """Display sentiment analysis results"""
    if result and result['status'] == 'success':
        sentiment_data = result['sentiment_analysis']
        sentiment = sentiment_data['sentiment']
        confidence = sentiment_data.get('confidence', 0)
        
        # Sentiment display with color coding
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"""
                <div class="sentiment-box {sentiment}">
                    <h2 style="margin: 0;">Sentiment: {sentiment.upper()}</h2>
                    <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem;">
                        Confidence: {confidence:.1%}
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Confidence gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = confidence * 100,
                title = {'text': "Confidence"},
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=200, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # Sentiment score
            score = sentiment_data.get('sentiment_score', 3)
            st.metric(
                label="Sentiment Score",
                value=f"{score}/5",
                delta=f"{score - 3:+d} from neutral"
            )
        
        # Probability distribution
        if sentiment_data.get('probabilities'):
            st.subheader("📊 Probability Distribution")
            probs_df = pd.DataFrame([
                {'Sentiment': k.capitalize(), 'Probability': v}
                for k, v in sentiment_data['probabilities'].items()
            ])
            
            fig = px.bar(
                probs_df, 
                x='Sentiment', 
                y='Probability',
                color='Sentiment',
                color_discrete_map={
                    'Amazing': '#27ae60',
                    'Positive': '#3498db',
                    'Neutral': '#95a5a6',
                    'Negative': '#e67e22',
                    'Terrible': '#e74c3c'
                },
                title="Sentiment Class Probabilities"
            )
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)

def display_response(result):
    """Display generated response"""
    if result and 'generated_response' in result:
        st.markdown("""
            <div class="response-box">
                <h3 style="margin-top: 0;">📝 Generated Response</h3>
                <p style="font-size: 1.1rem; line-height: 1.6;">
                    {}
                </p>
            </div>
        """.format(result['generated_response']), unsafe_allow_html=True)

def display_mlflow_reports():
    """Display MLflow reports tab"""
    st.header("📊 MLflow Experiment Reports")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.info("View your model training runs, hyperparameter tuning results, and metrics")
    
    with col2:
        port = st.number_input("MLflow Port", value=5000, min_value=1000, max_value=9999)
    
    with col3:
        if st.button("🔄 Refresh MLflow UI"):
            stop_mlflow_ui()
            st.session_state.mlflow_url = None
    
    # MLflow UI display options
    display_option = st.radio(
        "Display Option",
        ["Embedded View", "External Link", "Direct Integration"],
        horizontal=True,
        help="Choose how to display MLflow UI"
    )
    
    if display_option == "Embedded View":
        # Try to start MLflow UI if not running
        if not st.session_state.mlflow_url:
            with st.spinner("Starting MLflow UI server..."):
                url = start_mlflow_ui(port)
                if url:
                    st.session_state.mlflow_url = url
                    st.success(f"✅ MLflow UI running at {url}")
                else:
                    st.error("❌ Could not start MLflow UI. Make sure MLflow is installed and mlruns directory exists.")
        
        if st.session_state.mlflow_url:
            # Embed MLflow UI in iframe
            st.markdown(f"""
                <iframe src="{st.session_state.mlflow_url}" 
                        class="mlflow-iframe"
                        title="MLflow UI">
                </iframe>
            """, unsafe_allow_html=True)
            
            st.info(f"🔗 MLflow UI is also accessible directly at: [{st.session_state.mlflow_url}]({st.session_state.mlflow_url})")
    
    elif display_option == "External Link":
        mlflow_url = f"http://localhost:{port}"
        st.markdown(f"""
            ### 🔗 Access MLflow UI
            
            Click the link below to open MLflow UI in a new tab:
            
            **[Open MLflow UI]({mlflow_url})**
            
            If the link doesn't work, make sure MLflow UI is running:
            ```bash
            mlflow ui --port {port}
            ```
        """)
        
        if st.button("Start MLflow UI"):
            with st.spinner("Starting MLflow UI..."):
                url = start_mlflow_ui(port)
                if url:
                    st.success(f"✅ MLflow UI started at {url}")
                    st.markdown(f"**[Click here to open]({url})**")
                else:
                    st.error("Failed to start MLflow UI")
    
    elif display_option == "Direct Integration":
        # Direct integration with MLflow API
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            
            # Set tracking URI
            mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', './mlruns'))
            client = MlflowClient()
            
            # Get all experiments
            experiments = client.search_experiments()
            
            if experiments:
                st.subheader("📂 Experiments")
                
                # Experiment selector
                exp_names = [exp.name for exp in experiments]
                selected_exp = st.selectbox("Select Experiment", exp_names)
                
                # Get experiment details
                exp = next(e for e in experiments if e.name == selected_exp)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Experiment ID", exp.experiment_id)
                with col2:
                    st.metric("Lifecycle Stage", exp.lifecycle_stage)
                with col3:
                    st.metric("Artifact Location", exp.artifact_location.split('/')[-1])
                
                # Get runs for selected experiment
                runs = client.search_runs(experiment_ids=[exp.experiment_id])
                
                if runs:
                    st.subheader("🏃 Runs")
                    
                    # Create DataFrame with run information
                    run_data = []
                    for run in runs:
                        run_info = {
                            'Run ID': run.info.run_id[:8],
                            'Status': run.info.status,
                            'Start Time': datetime.fromtimestamp(run.info.start_time/1000).strftime('%Y-%m-%d %H:%M:%S'),
                        }
                        
                        # Add metrics
                        for key, value in run.data.metrics.items():
                            run_info[key] = round(value, 4)
                        
                        # Add selected params
                        for key, value in list(run.data.params.items())[:5]:  # Limit to 5 params for display
                            run_info[f"param_{key}"] = value
                        
                        run_data.append(run_info)
                    
                    runs_df = pd.DataFrame(run_data)
                    
                    # Display runs table
                    st.dataframe(runs_df, use_container_width=True)
                    
                    # Metrics visualization
                    if len(runs_df) > 0:
                        st.subheader("📈 Metrics Comparison")
                        
                        # Get numeric columns (metrics)
                        metric_cols = [col for col in runs_df.columns 
                                     if col not in ['Run ID', 'Status', 'Start Time'] 
                                     and not col.startswith('param_')]
                        
                        if metric_cols:
                            selected_metric = st.selectbox("Select Metric to Visualize", metric_cols)
                            
                            # Create bar chart
                            fig = px.bar(
                                runs_df, 
                                x='Run ID', 
                                y=selected_metric,
                                title=f'{selected_metric} Across Runs',
                                color=selected_metric,
                                color_continuous_scale='viridis'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Best run for selected metric
                            best_run = runs_df.loc[runs_df[selected_metric].idxmax() if 'accuracy' in selected_metric.lower() 
                                                  else runs_df[selected_metric].idxmin()]
                            st.success(f"🏆 Best run for {selected_metric}: Run {best_run['Run ID']} with value {best_run[selected_metric]}")
                else:
                    st.info("No runs found for this experiment")
            else:
                st.warning("No experiments found. Train some models first!")
                
        except ImportError:
            st.error("MLflow is not installed. Install it with: pip install mlflow")
        except Exception as e:
            st.error(f"Error connecting to MLflow: {str(e)}")
            st.info("Make sure MLflow tracking server is running and mlruns directory exists")

# Main App
def main():
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🤖 EchoAI - Intelligent Review Response System</h1>
            <p style="font-size: 1.2rem; margin-top: 0.5rem;">
                Analyze customer sentiment and generate personalized responses
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model settings
        st.subheader("Model Settings")
        llm_model = st.selectbox(
            "LLM Model",
            ["google/flan-t5-base", "google/flan-t5-large", "google/flan-t5-small"],
            help="Select the language model for response generation"
        )
        
        load_llm = st.checkbox(
            "Enable Response Generation",
            value=True,
            help="Uncheck for sentiment analysis only"
        )
        
        if st.button("🚀 Load/Reload Models", type="primary"):
            load_model(llm_model, load_llm)
        
        # Statistics
        if st.session_state.stats['total'] > 0:
            st.divider()
            st.subheader("📈 Session Statistics")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Processed", st.session_state.stats['total'])
            with col2:
                avg_conf = sum(h['confidence'] for h in st.session_state.history) / len(st.session_state.history)
                st.metric("Avg Confidence", f"{avg_conf:.1%}")
            
            # Sentiment distribution pie chart
            sentiment_counts = {
                k: v for k, v in st.session_state.stats.items() 
                if k != 'total' and v > 0
            }
            
            if sentiment_counts:
                fig = px.pie(
                    values=list(sentiment_counts.values()),
                    names=[k.capitalize() for k in sentiment_counts.keys()],
                    color_discrete_map={
                        'Amazing': '#27ae60',
                        'Positive': '#3498db',
                        'Neutral': '#95a5a6',
                        'Negative': '#e67e22',
                        'Terrible': '#e74c3c'
                    },
                    title="Sentiment Distribution"
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True)
        
        # Export options
        st.divider()
        st.subheader("📥 Export")
        if st.button("Export History to JSON"):
            if st.session_state.history:
                json_str = json.dumps(st.session_state.history, indent=2)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_str,
                    file_name=f"echoai_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.info("No history to export")
    
    # Main content area
    if not st.session_state.model_loaded:
        st.info("Please load the models from the sidebar to begin")
    else:
        # Create tabs
        tab1, tab2 = st.tabs(["✍️ Single Review", "📜 History"])
        
        with tab1:
            st.header("Enter Review Details")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                review_text = st.text_area(
                    "Review Text *",
                    placeholder="Enter the customer review here...",
                    height=150,
                    help="The main review content to analyze"
                )
            
            with col2:
                st.subheader("Optional Metadata")
                
                rating = st.slider(
                    "Rating",
                    min_value=1.0,
                    max_value=5.0,
                    value=3.0,
                    step=0.5,
                    help="Customer's star rating"
                )
                
                place_name = st.text_input(
                    "Place Name",
                    placeholder="e.g., The Grand Restaurant"
                )
                
                author_name = st.text_input(
                    "Author Name",
                    placeholder="e.g., John Smith"
                )
            
            col3, col4 = st.columns([1, 1])
            
            with col3:
                place_address = st.text_input(
                    "Place Address",
                    placeholder="e.g., 123 Main St, Boston, MA"
                )
                
                provider = st.selectbox(
                    "Review Platform",
                    ["", "Google", "Yelp", "TripAdvisor", "Other"],
                    help="Source platform of the review"
                )
            
            with col4:
                review_date = st.date_input(
                    "Review Date",
                    value=datetime.now(),
                    help="Date when the review was posted"
                )
            
            # Process button
            if st.button("🔍 Analyze Review", type="primary", disabled=not review_text):
                with st.spinner("Processing review..."):
                    # Prepare review data
                    review_data = {
                        'reviewText': review_text,
                        'reviewRating': rating,
                        'placeName': place_name if place_name else None,
                        'placeAddress': place_address if place_address else None,
                        'provider': provider if provider else None,
                        'authorName': author_name if author_name else None,
                        'reviewDate': review_date.strftime('%Y-%m-%d')
                    }
                    
                    # Process review
                    result = process_review(review_data)
                    
                    if result:
                        st.success("✅ Analysis complete!")
                        
                        # Display results
                        display_sentiment_result(result)
                        display_response(result)
                        
                        # Show raw JSON in expander
                        with st.expander("🔧 View Raw Results"):
                            st.json(result)

            if st.button("Open MLflow UI"):
                mlflow_url = "http://127.0.0.1:5000"
                st.markdown(f"[Click here to open MLflow UI]({mlflow_url})")

        
        with tab2:
            st.header("📋 Batch Processing")
            
            uploaded_file = st.file_uploader(
                "Upload CSV file with reviews",
                type=['csv'],
                help="CSV must contain 'reviewText' column. Optional columns: placeName, placeAddress, provider, reviewRating, authorName, reviewDate"
            )
            
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                
                st.subheader("Preview of uploaded data")
                st.dataframe(df.head(), use_container_width=True)
                
                if 'reviewText' not in df.columns:
                    st.error("❌ CSV must contain 'reviewText' column")
                else:
                    st.success(f"✅ Found {len(df)} reviews to process")
                    
                    if st.button("🚀 Process All Reviews", type="primary"):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        results = []
                        for idx, row in df.iterrows():
                            status_text.text(f"Processing review {idx+1}/{len(df)}")
                            progress_bar.progress((idx + 1) / len(df))
                            
                            review_data = row.to_dict()
                            result = st.session_state.pipeline.process_review(
                                review_data, 
                                generate_response=True
                            )
                            results.append(result)
                        
                        # Add results to dataframe
                        df['sentiment'] = [r['sentiment_analysis']['sentiment'] if r['status'] == 'success' else 'error' for r in results]
                        df['confidence'] = [r['sentiment_analysis'].get('confidence', 0) if r['status'] == 'success' else 0 for r in results]
                        df['generated_response'] = [r.get('generated_response', '') if r['status'] == 'success' else '' for r in results]
                        
                        status_text.text("✅ Processing complete!")
                        
                        # Display results
                        st.subheader("Processed Results")
                        st.dataframe(df, use_container_width=True)
                        
                        # Download button
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results CSV",
                            data=csv,
                            file_name=f"processed_reviews_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
        

if __name__ == "__main__":
    main()
