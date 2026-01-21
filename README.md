# 🚀 Advanced Sentiment Analysis Platform

Production-ready sentiment analysis with 4 trained models, REST API, and enterprise features.

## ✨ Features

### 🎯 **4 Trained Models**
1. **3-Class Sentiment** - Negative/Neutral/Positive (73.3% accuracy)
2. **Binary Sentiment** - Positive/Negative (86.5% accuracy)
3. **5-Star Rating** - 1-5 star predictions (52% accuracy)
4. **Emotion Detection** - 6 emotions using BERT (72.7% accuracy)

### 🔥 **New Advanced Features**

#### 📊 **Batch Processing**
- Upload CSV files with thousands of reviews
- Analyze multiple texts at once
- Progress tracking and statistics
- Visual distribution charts

#### 🔄 **Model Comparison**
- Run same text through all 4 models
- Side-by-side comparison view
- Confidence level visualization
- Best model recommendation

#### 📥 **Export Options**
- **CSV** - Comma-separated values
- **Excel** - Professional formatted reports with charts
- **JSON** - API-friendly format
- Includes predictions, confidence scores, and all probabilities

#### 🌐 **REST API**
- FastAPI-based production API
- Swagger documentation at `/docs`
- Rate limiting ready
- Batch processing support

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Streamlit App
```bash
streamlit run app.py
```
Access at: http://localhost:8501

### 3. Run REST API
```bash
python api.py
```
Access at: http://localhost:8000
API Docs: http://localhost:8000/docs

## 📖 API Usage

### Single Prediction
```python
import requests

response = requests.post('http://localhost:8000/predict', json={
    "text": "This product is amazing!",
    "model": "3-class"
})
print(response.json())
```

### Batch Processing
```python
response = requests.post('http://localhost:8000/batch', json={
    "texts": [
        "Great product!",
        "Not satisfied",
        "It's okay"
    ],
    "model": "3-class"
})
```

### Model Comparison
```python
response = requests.post('http://localhost:8000/compare', 
    params={"text": "I love this!"})
```

### Test All Endpoints
```bash
python test_api.py
```

## 📊 Model Performance

| Model | Accuracy | Speed | Use Case |
|-------|----------|-------|----------|
| Binary Sentiment | 86.5% | Fast | Simple pos/neg |
| 3-Class Sentiment | 73.3% | Fast | Detailed sentiment |
| Emotion Detection | 72.7% | Slow | Emotion analysis |
| 5-Star Rating | 52% | Fast | Rating prediction |

## 🎯 Use Cases

### E-commerce
- Analyze product reviews at scale
- Track customer sentiment over time
- Identify negative feedback quickly

### Social Media
- Monitor brand sentiment
- Detect customer emotions
- Track campaign effectiveness

### Customer Support
- Prioritize urgent complaints
- Identify frustrated customers
- Route tickets by sentiment

### Market Research
- Analyze survey responses
- Track brand perception
- Competitor analysis

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/models` | GET | List available models |
| `/predict` | POST | Single prediction |
| `/batch` | POST | Batch predictions |
| `/compare` | POST | Compare all models |

## 📈 Production Features

### ✅ Implemented
- [x] 4 trained ML models
- [x] REST API with FastAPI
- [x] Batch CSV processing
- [x] Model comparison
- [x] Multiple export formats
- [x] Interactive visualizations
- [x] Confidence scoring

### 🔜 Coming Soon
- [ ] API authentication
- [ ] Rate limiting
- [ ] Usage analytics dashboard
- [ ] Model versioning
- [ ] A/B testing framework
- [ ] Multi-language support
- [ ] Feedback loop for retraining

## 📁 Project Structure

```
mmm/
├── app.py                          # Streamlit web app
├── api.py                          # FastAPI REST API
├── test_api.py                     # API testing script
├── requirements.txt                # Dependencies
│
├── Models (4 trained):
│   ├── xgboost_sentiment_model.json    # 3-class
│   ├── binary_sentiment_model.json     # Binary
│   ├── fiveStar_model.json             # 5-star
│   └── emotion_bert_model/             # Emotions
│
├── Training Notebooks:
│   ├── xgboost_sentiment.ipynb         # 3-class training
│   ├── train_binary_sentiment.ipynb    # Binary training
│   ├── train_fiveStar.ipynb            # 5-star training
│   └── train_emotion.ipynb             # Emotion training
│
└── Data:
    ├── Reviews.csv                     # Amazon reviews
    └── train.tsv                       # GoEmotions dataset
```

## 💡 Tips

### For Best Results:
- Use **Binary Sentiment** for simple positive/negative classification
- Use **3-Class Sentiment** when neutral sentiment matters
- Use **Emotion Detection** for customer support and social media
- Use **5-Star Rating** for e-commerce review predictions

### Performance:
- XGBoost models: ~10ms per prediction
- BERT emotion model: ~100ms per prediction
- Batch processing: ~1000 reviews per minute

## 🛠️ Development

### Add New Model
1. Create training notebook
2. Train and save model files
3. Add configuration to `MODEL_CONFIGS`
4. Update API endpoints

### Retrain Existing Model
1. Open respective training notebook
2. Run all cells
3. Models auto-saved and ready to use

## 📝 License

MIT License - Free to use and modify

## 🤝 Contributing

Contributions welcome! Areas to improve:
- More model types (sarcasm, aspect-based)
- Better accuracy
- Faster inference
- More export formats
- Additional languages

## 📞 Support

For issues or questions:
- Check API docs at `/docs`
- Review training notebooks
- Test with `test_api.py`

---

**Built with:** Python, Streamlit, FastAPI, XGBoost, BERT, scikit-learn
