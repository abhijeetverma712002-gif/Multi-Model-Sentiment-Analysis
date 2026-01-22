import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.graph_objects as go
import plotly.express as px
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Page configuration
st.set_page_config(
    page_title="Multi-Model Sentiment Analysis",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Model configurations - 4 trained models
MODEL_CONFIGS = {
    "3-Class Sentiment": {
        "description": "Basic sentiment analysis (Negative, Neutral, Positive)",
        "icon": "😊",
        "model_file": "xgboost_sentiment_model.json",
        "vectorizer_file": "tfidf_vectorizer.pkl",
        "mappings_file": "label_mappings.pkl",
        "classes": ["negative", "neutral", "positive"],
        "colors": ["#ff4b4b", "#ffa500", "#00cc66"]
    },
    "Binary Sentiment": {
        "description": "Simple positive/negative classification",
        "icon": "👍👎",
        "model_file": "binary_sentiment_model.json",
        "vectorizer_file": "binary_vectorizer.pkl",
        "mappings_file": "binary_mappings.pkl",
        "classes": ["negative", "positive"],
        "colors": ["#ff4b4b", "#00cc66"]
    },
    "5-Star Rating": {
        "description": "5-star rating prediction (1-5 stars)",
        "icon": "⭐",
        "model_file": "fiveStar_model.json",
        "vectorizer_file": "fiveStar_vectorizer.pkl",
        "mappings_file": "fiveStar_mappings.pkl",
        "classes": ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"],
        "colors": ["#ff0000", "#ff6600", "#ffcc00", "#99cc00", "#00cc00"]
    },
    "Emotion Detection": {
        "description": "Detect 6 basic emotions using BERT",
        "icon": "😄😢😠",
        "model_file": "emotion_bert_model",  # BERT model directory
        "vectorizer_file": None,  # BERT uses its own tokenizer
        "mappings_file": "emotion_mappings.pkl",
        "classes": ["anger", "disgust", "fear", "joy", "sadness", "surprise"],
        "colors": ["#cc0000", "#006600", "#9900cc", "#ffcc00", "#0066cc", "#ff6600"],
        "model_type": "bert"  # Mark as BERT model
    }
}

# Load model and vectorizer
@st.cache_resource
def load_model(model_type):
    config = MODEL_CONFIGS[model_type]
    
    # Check if files exist
    if not os.path.exists(config["model_file"]):
        return None, None, None, "model_missing"
    
    try:
        # Check if this is a BERT model
        if config.get("model_type") == "bert":
            # Ensure pytorch_model.bin exists for transformers compatibility
            safetensors_path = os.path.join(config["model_file"], "model.safetensors")
            bin_path = os.path.join(config["model_file"], "pytorch_model.bin")
            if os.path.exists(safetensors_path) and not os.path.exists(bin_path):
                import shutil
                shutil.copy(safetensors_path, bin_path)
            # Load BERT model and tokenizer
            model = BertForSequenceClassification.from_pretrained(config["model_file"])
            tokenizer = BertTokenizer.from_pretrained(config["model_file"])
            
            # Load label mappings
            with open(config["mappings_file"], 'rb') as f:
                mappings = pickle.load(f)
            
            return model, tokenizer, mappings['id2label'], "success"
        else:
            # Load XGBoost model
            model = xgb.XGBClassifier()
            model.load_model(config["model_file"])
            
            # Load vectorizer
            with open(config["vectorizer_file"], 'rb') as f:
                vectorizer = pickle.load(f)
            
            # Load label mappings
            with open(config["mappings_file"], 'rb') as f:
                mappings = pickle.load(f)
            
            return model, vectorizer, mappings['id2label'], "success"
    except FileNotFoundError:
        return None, None, None, "file_not_found"
    except Exception as e:
        return None, None, None, str(e)

# Predict sentiment
def predict_sentiment(text, model, vectorizer, id2label, model_type):
    config = MODEL_CONFIGS[model_type]
    
    # Check if this is a BERT model
    if config.get("model_type") == "bert":
        # BERT prediction
        tokenizer = vectorizer  # For BERT, vectorizer is actually the tokenizer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Tokenize input
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get prediction
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_id = torch.argmax(probs, dim=-1).item()
            pred_proba = probs[0].cpu().numpy()
    else:
        # XGBoost prediction
        text_tfidf = vectorizer.transform([text])
        pred_id = model.predict(text_tfidf)[0]
        pred_proba = model.predict_proba(text_tfidf)[0]
    
    # Build probabilities dict dynamically based on classes
    probabilities = {}
    for idx, class_name in enumerate(config["classes"]):
        probabilities[class_name] = float(pred_proba[idx]) if idx < len(pred_proba) else 0.0
    
    return {
        'label': id2label[pred_id],
        'confidence': float(pred_proba[pred_id]),
        'probabilities': probabilities
    }

# Create probability chart
def create_probability_chart(probabilities, model_type):
    config = MODEL_CONFIGS[model_type]
    
    labels = [label.capitalize() for label in config["classes"]]
    values = [probabilities[cls] for cls in config["classes"]]
    colors = config["colors"]
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f'{v:.1%}' for v in values],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=f"{model_type} Probability Distribution",
        xaxis_title="Category",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        height=400,
        showlegend=False
    )
    
    return fig

# Main app
def main():
    # Header
    st.title("🤖 Multi-Model Sentiment Analysis")
    st.markdown("Advanced NLP sentiment analysis with 7 different model types")
    
    # Sidebar - Model Selection
    with st.sidebar:
        st.header("🎯 Select Analysis Type")
        
        model_type = st.selectbox(
            "Choose Model:",
            list(MODEL_CONFIGS.keys()),
            format_func=lambda x: f"{MODEL_CONFIGS[x]['icon']} {x}"
        )
        
        # Display model info
        config = MODEL_CONFIGS[model_type]
        st.info(f"**{config['icon']} {model_type}**\n\n{config['description']}")
        
        # Load selected model
        model, vectorizer, id2label, status = load_model(model_type)
        
        if status == "model_missing":
            st.warning(f"⚠️ Model not trained yet!\n\nTo train this model:\n1. Create training data\n2. Run training notebook\n3. Save model files")
            
            # Show which model is available
            st.markdown("---")
            st.success("✅ **3-Class Sentiment** model is ready!")
            if st.button("Switch to 3-Class"):
                st.session_state['model_type'] = "3-Class Sentiment"
                st.rerun()
        elif status != "success":
            st.error(f"❌ Error loading model: {status}")
        else:
            st.success("✅ Model loaded successfully!")
        
        st.markdown("---")
        st.header("📚 All Models")
        for name, cfg in MODEL_CONFIGS.items():
            exists = os.path.exists(cfg["model_file"])
            status_icon = "✅" if exists else "⚪"
            st.markdown(f"{status_icon} {cfg['icon']} **{name}**")
        
        st.markdown("---")
        st.header("🎯 Example Texts")
        
        # Different examples per model
        examples_dict = {
            "3-Class Sentiment": [
                "I love this product! It's amazing!",
                "The product is okay, nothing special.",
                "Worst purchase ever. Total waste of money."
            ],
            "Binary Sentiment": [
                "Great service and quality!",
                "Terrible experience, disappointed."
            ],
            "5-Star Rating": [
                "Absolutely perfect! Best ever!",
                "Pretty good, would recommend.",
                "It's okay, nothing special.",
                "Not great, several issues.",
                "Completely awful! Total disaster!"
            ],
            "Emotion Detection": [
                "I'm so happy and excited!",
                "This makes me really sad.",
                "I'm furious about this!",
                "That's terrifying!",
                "Wow, I didn't expect that!",
                "This is disgusting."
            ]
        }
        
        examples = examples_dict.get(model_type, examples_dict["3-Class Sentiment"])
        
        if st.button("Load Random Example"):
            import random
            st.session_state['example_text'] = random.choice(examples)
        
        # Show example list
        with st.expander("View all examples"):
            for ex in examples:
                st.caption(f"• {ex}")
    
    # Check if model is loaded
    if status != "success":
        st.error("⚠️ Please select a trained model from the sidebar or train a new model.")
        st.stop()
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Single Prediction", "📊 Batch Analysis", "🔄 Model Comparison", "📥 Export Results", "📈 Model Info"])
    
    with tab1:
        st.header(f"Single Text Analysis - {model_type}")
        
        # Text input
        text_input = st.text_area(
            "Enter text to analyze:",
            value=st.session_state.get('example_text', ''),
            height=150,
            placeholder="Type or paste your text here..."
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
        with col2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_button:
            st.session_state['example_text'] = ''
            st.rerun()
        
        if analyze_button and text_input.strip():
            with st.spinner("Analyzing..."):
                result = predict_sentiment(text_input, model, vectorizer, id2label, model_type)
                
                # Display results
                st.markdown("---")
                st.subheader("Results")
                
                # Sentiment badge with dynamic styling
                sentiment = result['label']
                confidence = result['confidence']
                
                # Dynamic emoji and color based on model type
                if model_type == "Emotion Detection":
                    emoji_map = {"joy": "😊", "sadness": "😢", "anger": "😠", "fear": "😨", "surprise": "😮", "disgust": "🤢"}
                    emoji = emoji_map.get(sentiment, "😐")
                elif model_type == "Sarcasm Detection":
                    emoji = "🙃" if sentiment == "sarcastic" else "😊"
                elif model_type == "Fine-Grained (5-Star)":
                    emoji = "⭐" * int(sentiment.split()[0])
                else:
                    emoji_map = {"positive": "😊", "negative": "😞", "neutral": "😐"}
                    emoji = emoji_map.get(sentiment, "😐")
                
                st.markdown(f"### {emoji} **{sentiment.upper()}** ({confidence:.1%} confidence)")
                
                # Metrics in columns
                cols = st.columns(len(config["classes"]))
                for idx, (cls, col) in enumerate(zip(config["classes"], cols)):
                    with col:
                        st.metric(cls.capitalize(), f"{result['probabilities'][cls]:.1%}")
                
                # Chart
                st.plotly_chart(create_probability_chart(result['probabilities'], model_type), use_container_width=True)
        
        elif analyze_button:
            st.warning("⚠️ Please enter some text to analyze.")
    
    with tab2:
        st.header("Batch Text Analysis")
        st.markdown("Analyze multiple texts at once by uploading a CSV file or entering texts manually.")
        
        # File upload
        uploaded_file = st.file_uploader("Upload CSV file (must have a 'text' column)", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            if 'text' not in df.columns:
                st.error("⚠️ CSV must contain a 'text' column!")
            else:
                st.success(f"✅ Loaded {len(df)} rows")
                
                if st.button("🔍 Analyze All", type="primary"):
                    with st.spinner(f"Analyzing {len(df)} texts..."):
                        results = []
                        progress_bar = st.progress(0)
                        
                        for idx, text in enumerate(df['text']):
                            if pd.notna(text) and str(text).strip():
                                result = predict_sentiment(str(text), model, vectorizer, id2label, model_type)
                                row = {
                                    'text': text,
                                    'sentiment': result['label'],
                                    'confidence': result['confidence']
                                }
                                # Add all probability columns dynamically
                                for cls in config["classes"]:
                                    row[f'{cls}_prob'] = result['probabilities'][cls]
                                results.append(row)
                            progress_bar.progress((idx + 1) / len(df))
                        
                        results_df = pd.DataFrame(results)
                        
                        # Display results
                        st.subheader("Results Summary")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Analyzed", len(results_df))
                        with col2:
                            st.metric("Positive", len(results_df[results_df['sentiment'] == 'positive']))
                        with col3:
                            st.metric("Neutral", len(results_df[results_df['sentiment'] == 'neutral']))
                        with col4:
                            st.metric("Negative", len(results_df[results_df['sentiment'] == 'negative']))
                        
                        # Sentiment distribution chart
                        sentiment_counts = results_df['sentiment'].value_counts()
                        
                        # Create color map dynamically
                        color_map = {cls: color for cls, color in zip(config["classes"], config["colors"])}
                        
                        fig = px.pie(
                            values=sentiment_counts.values,
                            names=sentiment_counts.index,
                            title=f"{model_type} Distribution",
                            color=sentiment_counts.index,
                            color_discrete_map=color_map
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show results table
                        st.subheader("Detailed Results")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name="sentiment_analysis_results.csv",
                            mime="text/csv"
                        )
        
        else:
            st.info("👆 Upload a CSV file to get started with batch analysis.")
            
            # Manual input option
            st.markdown("---")
            st.subheader("Or Enter Multiple Texts Manually")
            manual_texts = st.text_area(
                "Enter one text per line:",
                height=200,
                placeholder="Text 1\nText 2\nText 3"
            )
            
            if st.button("🔍 Analyze Manual Entries", type="primary"):
                if manual_texts.strip():
                    texts = [t.strip() for t in manual_texts.split('\n') if t.strip()]
                    
                    with st.spinner(f"Analyzing {len(texts)} texts..."):
                        results = []
                        for text in texts:
                            result = predict_sentiment(text, model, vectorizer, id2label, model_type)
                            results.append({
                                'text': text,
                                'sentiment': result['label'],
                                'confidence': f"{result['confidence']:.1%}"
                            })
                        
                        results_df = pd.DataFrame(results)
                        st.dataframe(results_df, use_container_width=True)
                else:
                    st.warning("⚠️ Please enter at least one text.")
    
    with tab3:
        st.header("🔄 Model Comparison")
        st.markdown("Run the same text through all 4 models and compare results")
        
        comparison_text = st.text_area(
            "Enter text to compare across models:",
            height=120,
            placeholder="Enter text here to see how different models analyze it..."
        )
        
        if st.button("🔍 Compare All Models", type="primary"):
            if comparison_text.strip():
                with st.spinner("Running all 4 models..."):
                    comparison_results = []
                    
                    for model_name in MODEL_CONFIGS.keys():
                        try:
                            m, v, labels, s = load_model(model_name)
                            if s == "success":
                                result = predict_sentiment(comparison_text, m, v, labels, model_name)
                                comparison_results.append({
                                    'Model': model_name,
                                    'Prediction': result['label'].upper(),
                                    'Confidence': f"{result['confidence']:.1%}",
                                    'Top 3 Probabilities': ', '.join([f"{k}: {v:.1%}" for k, v in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)[:3]])
                                })
                        except:
                            pass
                    
                    if comparison_results:
                        st.subheader("📊 Comparison Results")
                        
                        # Display as table
                        comp_df = pd.DataFrame(comparison_results)
                        st.dataframe(comp_df, use_container_width=True, height=250)
                        
                        # Visual comparison
                        st.subheader("📈 Confidence Comparison")
                        conf_df = pd.DataFrame(comparison_results)
                        conf_df['Confidence_num'] = conf_df['Confidence'].str.rstrip('%').astype(float) / 100
                        
                        fig = px.bar(
                            conf_df,
                            x='Model',
                            y='Confidence_num',
                            text='Prediction',
                            title='Model Confidence Levels',
                            labels={'Confidence_num': 'Confidence'},
                            color='Confidence_num',
                            color_continuous_scale='RdYlGn'
                        )
                        fig.update_traces(textposition='outside')
                        fig.update_layout(height=400, yaxis_tickformat='.0%')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Best recommendation
                        best_model = max(comparison_results, key=lambda x: float(x['Confidence'].rstrip('%')))
                        st.success(f"🏆 **Most Confident Model:** {best_model['Model']} - {best_model['Prediction']} ({best_model['Confidence']})")
            else:
                st.warning("⚠️ Please enter text to compare.")
    
    with tab4:
        st.header("📥 Export & Download")
        st.markdown("Export your predictions in various formats")
        
        # Store results in session state for export
        if 'export_results' not in st.session_state:
            st.session_state.export_results = None
        
        export_text = st.text_area(
            "Enter texts to analyze and export (one per line):",
            height=200,
            placeholder="Review 1\nReview 2\nReview 3..."
        )
        
        if st.button("📊 Generate Export Data", type="primary"):
            if export_text.strip():
                texts = [t.strip() for t in export_text.split('\n') if t.strip()]
                
                with st.spinner(f"Analyzing {len(texts)} texts for export..."):
                    export_results = []
                    for text in texts:
                        result = predict_sentiment(text, model, vectorizer, id2label, model_type)
                        row = {'Text': text, 'Prediction': result['label'], 'Confidence': f"{result['confidence']:.1%}"}
                        # Add probabilities
                        for cls, prob in result['probabilities'].items():
                            row[f'{cls.capitalize()}_Prob'] = f"{prob:.1%}"
                        export_results.append(row)
                    
                    st.session_state.export_results = pd.DataFrame(export_results)
                    st.success(f"✅ Generated {len(export_results)} predictions ready for export!")
        
        if st.session_state.export_results is not None:
            df_export = st.session_state.export_results
            
            st.subheader("Preview")
            st.dataframe(df_export, use_container_width=True)
            
            st.subheader("Download Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # CSV Download
                csv = df_export.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📄 Download CSV",
                    data=csv,
                    file_name=f"{model_type.replace(' ', '_')}_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Excel Download
                from io import BytesIO
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    df_export.to_excel(writer, index=False, sheet_name='Predictions')
                    
                    # Get the xlsxwriter workbook and worksheet objects
                    workbook = writer.book
                    worksheet = writer.sheets['Predictions']
                    
                    # Add a header format
                    header_format = workbook.add_format({
                        'bold': True,
                        'bg_color': '#4CAF50',
                        'font_color': 'white',
                        'border': 1
                    })
                    
                    # Write headers with format
                    for col_num, value in enumerate(df_export.columns.values):
                        worksheet.write(0, col_num, value, header_format)
                
                excel_data = buffer.getvalue()
                st.download_button(
                    label="📊 Download Excel",
                    data=excel_data,
                    file_name=f"{model_type.replace(' ', '_')}_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            with col3:
                # JSON Download
                json_data = df_export.to_json(orient='records', indent=2)
                st.download_button(
                    label="🔗 Download JSON",
                    data=json_data,
                    file_name=f"{model_type.replace(' ', '_')}_results.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            # Statistics
            st.markdown("---")
            st.subheader("📈 Export Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(df_export))
            with col2:
                avg_conf = df_export['Confidence'].str.rstrip('%').astype(float).mean()
                st.metric("Avg Confidence", f"{avg_conf:.1f}%")
            with col3:
                most_common = df_export['Prediction'].mode()[0] if len(df_export) > 0 else "N/A"
                st.metric("Most Common", most_common.capitalize())
    
    with tab5:
        st.header(f"Model Information - {model_type}")
        st.markdown(f"**{config['icon']} {config['description']}**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Model Classes")
            classes_df = pd.DataFrame({
                'Class': [cls.capitalize() for cls in config["classes"]],
                'Color': config["colors"]
            })
            
            # Create visual representation
            for idx, row in classes_df.iterrows():
                st.markdown(f"<span style='color:{row['Color']}; font-size:20px;'>●</span> **{row['Class']}**", unsafe_allow_html=True)
            
            st.markdown("---")
            
            st.subheader("🔧 Model Configuration")
            model_config = {
                'Algorithm': 'XGBoost Classifier',
                'Features': 'TF-IDF Vectorization',
                'Max Features': '5,000',
                'N-grams': 'Unigrams + Bigrams',
                'Max Depth': '6',
                'Learning Rate': '0.1',
                'N Estimators': '200'
            }
            
            for param, value in model_config.items():
                st.text(f"{param}: {value}")
        
        with col2:
            st.subheader("📚 Model Descriptions")
            
            st.markdown("""
            **Available Models:**
            
            1. **3-Class Sentiment** (✅ Ready)
               - Basic sentiment analysis
               - Classes: Negative, Neutral, Positive
            
            2. **Binary Sentiment**
               - Simple pos/neg classification
               - Classes: Negative, Positive
            
            3. **Fine-Grained (5-Star)**
               - 5-level rating system
               - Classes: 1-5 stars
            
            4. **Emotion Detection**
               - 6 basic emotions
               - Classes: Joy, Sadness, Anger, Fear, Surprise, Disgust
            
            5. **Sarcasm Detection**
               - Identify sarcastic text
               - Classes: Non-sarcastic, Sarcastic
            
            6. **Aspect-Based**
               - Sentiment per feature/aspect
               - Classes: Negative, Neutral, Positive
            
            7. **Target-Specific**
               - Sentiment toward entities
               - Classes: Negative, Neutral, Positive
            """)
        
        st.markdown("---")
        st.info(f"💡 **Current Model:** {model_type}\n\nTo train other models, create labeled datasets and run training notebooks for each model type.")

if __name__ == "__main__":
    main()

