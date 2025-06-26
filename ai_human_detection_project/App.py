# STREAMLIT TREE CLASSIFICATION APP - DUAL MODEL SUPPORT
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page Configuration
st.set_page_config(
    page_title="ML Text Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
       .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .success-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            margin: 1rem 0;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #007bff;
        }
    </style>
""", unsafe_allow_html=True)

    # ============================================================================
    # MODEL LOADING SECTION
    # ============================================================================
@st.cache_resource
def load_models():
    models = {}

    try:
        # Load main pipeline (SVM pipeline)
        try:
            models['pipeline'] = joblib.load('models/AI_vs_Human_Analyzer_pipeline.pkl')
            models['pipeline_available'] = True
        except FileNotFoundError:
            models['pipeline_available'] = False

        # Load TF-IDF vectorizer
        try:
            models['vectorizer'] = joblib.load('models/tfidf_vectorizer.pkl')
            models['vectorizer_available'] = True
        except FileNotFoundError:
            models['vectorizer_available'] = False

        # Load individual models
        #load svm model
        try:
            models['svm'] = joblib.load('models/svm_model.pkl')
            models['svm_available'] = True
        except FileNotFoundError:
            models['svm_available'] = False
            
        #load AdaBoost model
        try:
            models['adaboost'] = joblib.load('models/ada_model.pkl')
            models['ada_available'] = True
        except FileNotFoundError:
            models['ada_available'] = False

        #load decision tree
        try:
            models['decision_tree'] = joblib.load('models/tree_model.pkl')
            models['tree_available'] = True
        except FileNotFoundError:
            models['tree_available'] = False

        # Check if at least one complete setup is available
        pipeline_ready = models['pipeline_available']
        individual_ready = models['vectorizer_available'] and (
            models['svm_available'] or models['tree_available'] or models['ada_available']
        )

        if not (pipeline_ready or individual_ready):
            st.error("No complete model setup found!")
            return None

        return models

    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

    # ============================================================================\n",
    # PREDICTION FUNCTION\n",
    # ============================================================================\n",
def make_prediction(text, model_choice, models):
    """Make prediction using the selected model"""
    if models is None:
        return None, None

    try:
        prediction = None
        probabilities = None

        if model_choice == "pipeline" and models.get('pipeline_available'):
            prediction = models['pipeline'].predict([text])[0]
            probabilities = models['pipeline'].predict_proba([text])[0]

        elif model_choice == "svm":
            if models.get('pipeline_available'):
                prediction = models['pipeline'].predict([text])[0]
                probabilities = models['pipeline'].predict_proba([text])[0]
            elif models.get('vectorizer_available') and models.get('svm_available'):
                X = models['vectorizer'].transform([text])
                prediction = models['svm'].predict(X)[0]
                probabilities = models['svm'].predict_proba(X)[0]

        elif model_choice == "adaboost":
            if models.get('vectorizer_available') and models.get('ada_available'):
                X = models['vectorizer'].transform([text])
                prediction = models['adaboost'].predict(X)[0]
                probabilities = models['adaboost'].predict_proba(X)[0]

        elif model_choice == "decision_tree":
            if models.get('vectorizer_available') and models.get('tree_available'):
                X = models['vectorizer'].transform([text])
                prediction = models['decision_tree'].predict(X)[0]
                probabilities = models['decision_tree'].predict_proba(X)[0]

        if prediction is not None and probabilities is not None:
            class_names = ['Human', 'AI']
            prediction_label = class_names[prediction]
            return prediction_label, probabilities
        else:
            return None, None

    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.error(f"Model choice: {model_choice}")
        st.error(f"Available models: {[k for k, v in models.items() if isinstance(v, bool) and v]}")
        return None, None

def get_available_models(models):
    """Get list of available models for selection"""
    available = []

    if models is None:
        return available

    if models.get('pipeline_available'):
        available.append(("pipeline", "📦 SVM (Pipeline)"))
    elif models.get('vectorizer_available') and models.get('svm_available'):
        available.append(("svm", "📈 SVM (Individual)"))

    if models.get('vectorizer_available') and models.get('tree_available'):
        available.append(("decision_tree", "🌳 Decision Tree"))

    if models.get('vectorizer_available') and models.get('ada_available'):
        available.append(("adaboost", "⚡ AdaBoost"))

    return available

    # ============================================================================\n",
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.title("🧭 Navigation")
st.sidebar.markdown("Choose what you want to do:")

page = st.sidebar.selectbox(
    "Select Page:",
    ["🏠 Home", "🔮 Single Prediction", "📁 Batch Processing", "⚖️ Model Comparison", "📊 Model Info", "❓ Help"]
)

# Load models
models = load_models()

# ============================================================================\n",
# HOME PAGE
# ============================================================================

if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🤖 AI vs Human ML Text Classification App</h1>', unsafe_allow_html=True)

    st.markdown("""
    Welcome to your machine learning web application! This app demonstrates sentiment analysis  
    using multiple trained models: **SVM**, **AdaBoost**, and **Decision Tree**.
    """)

    # App overview
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 🔮 Single Prediction
        - Enter text manually  
        - Choose between models  
        - Get instant predictions  
        - See confidence scores  
        """)

    with col2:
        st.markdown("""
        ### 📁 Batch Processing
        - Upload text files  
        - Process multiple texts  
        - Compare model performance  
        - Download results  
        """)

    with col3:
        st.markdown("""
        ### ⚖️ Model Comparison
        - Compare different models  
        - Side-by-side results  
        - Agreement analysis  
        - Performance metrics  
        """)

    # Model status
    st.subheader("📋 Model Status")
    if models:
        st.success("✅ Models loaded successfully!")

        col1, col2, col3 = st.columns(3)

        with col1:
            if models.get('pipeline_available'):
                st.info("**📈 SVM**\n✅ Pipeline Available")
            elif models.get('svm_available') and models.get('vectorizer_available'):
                st.info("**📈 SVM**\n✅ Individual Components")
            else:
                st.warning("**📈 SVM**\n❌ Not Available")

        with col2:
            if models.get('ada_available') and models.get('vectorizer_available'):
                st.info("**⚡ AdaBoost**\n✅ Available")
            else:
                st.warning("**⚡ AdaBoost**\n❌ Not Available")

            if models.get('tree_available') and models.get('vectorizer_available'):
                st.info("**🌳 Decision Tree**\n✅ Available")
            else:
                st.warning("**🌳 Decision Tree**\n❌ Not Available")

        with col3:
            if models.get('vectorizer_available'):
                st.info("**🔤 TF-IDF Vectorizer**\n✅ Available")
            else:
                st.warning("**🔤 TF-IDF Vectorizer**\n❌ Not Available")
    else:
        st.error("❌ Models not loaded. Please check model files.")

# ============================================================================\n",
# SINGLE PREDICTION PAGE
# ============================================================================
elif page == "🔮 Single Prediction":
    st.header("🔮 Make a Single Prediction")
    st.markdown("Enter text below and select a model to get classification results (AI vs Human).")

    if models:
        available_models = get_available_models(models)

        if available_models:
            # Model selection
            model_choice = st.selectbox(
                "Choose a model:",
                options=[model[0] for model in available_models],
                format_func=lambda x: next(model[1] for model in available_models if model[0] == x)
            )

            # Handle session state for examples
            if 'user_input' not in st.session_state:
                st.session_state.user_input = ""

            # Text input
            user_input = st.text_area(
                "Enter your text here:",
                value=st.session_state.user_input,
                placeholder="Type or paste your text here (e.g., article, essay, or review)...",
                height=150
            )

            # Character/word count
            if user_input:
                st.caption(f"Character count: {len(user_input)} | Word count: {len(user_input.split())}")
                st.session_state.user_input = user_input

            # Example texts
            with st.expander("📝 Try these example texts"):
                examples = [
                    "The experiment was conducted in a controlled environment, ensuring minimal bias.",
                    "I generated this content using an AI model trained on diverse text sources.",
                    "Humans have the unique ability to contextualize and reason abstractly.",
                    "AI-generated text often lacks emotional nuance or subtlety.",
                    "This essay explores the ethical implications of artificial intelligence."
                ]
                col1, col2 = st.columns(2)
                for i, example in enumerate(examples):
                    with col1 if i % 2 == 0 else col2:
                        if st.button(f"Example {i+1}", key=f"example_{i}"):
                            st.session_state.user_input = example
                            st.rerun()

            # Prediction
            if st.button("🚀 Predict", type="primary"):
                if user_input.strip():
                    with st.spinner("Analyzing..."):
                        prediction, probabilities = make_prediction(user_input, model_choice, models)

                        if prediction is not None and probabilities is not None:
                            class_names = ['Human', 'AI']
                            pred_index = class_names.index(prediction)

                            col1, col2 = st.columns([3, 1])
                            with col1:
                                if prediction == "Human":
                                    st.success(f"🎯 Prediction: **{prediction}**")
                                else:
                                    st.warning(f"🎯 Prediction: **{prediction}**")

                            with col2:
                                confidence = max(probabilities)
                                st.metric("Confidence", f"{confidence:.1%}")

                            # Probabilities section
                            st.subheader("📊 Prediction Probabilities")
                            for idx, cls in enumerate(class_names):
                                st.metric(cls, f"{probabilities[idx]:.1%}")

                            # Bar chart
                            prob_df = pd.DataFrame({
                                'Class': class_names,
                                'Probability': probabilities
                            })
                            st.bar_chart(prob_df.set_index('Class'), height=300)
                        else:
                            st.error("Failed to make prediction.")
                else:
                    st.warning("Please enter some text to classify!")
        else:
            st.error("No models available for prediction.")
    else:
        st.warning("Models not loaded. Please check the model files.")

# ============================================================================\n",
 # BATCH PROCESSING PAGE
# ============================================================================
elif page == "📁 Batch Processing":
    st.header("📁 Upload File for Batch Processing")
    st.markdown("Upload a `.txt` or `.csv` file to classify multiple texts at once using a selected model.")

    if models:
        available_models = get_available_models(models)

        if available_models:
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['txt', 'csv'],
                help="Upload a .txt file (one text per line) or .csv file (text in first column)"
            )

            if uploaded_file:
                model_choice = st.selectbox(
                    "Choose model for batch processing:",
                    options=[model[0] for model in available_models],
                    format_func=lambda x: next(model[1] for model in available_models if model[0] == x)
                )

                if st.button("📊 Process File"):
                    try:
                        # Read input texts
                        if uploaded_file.type == "text/plain":
                            content = str(uploaded_file.read(), "utf-8")
                            texts = [line.strip() for line in content.split('\n') if line.strip()]
                        else:  # CSV
                            df = pd.read_csv(uploaded_file)
                            texts = df.iloc[:, 0].astype(str).tolist()

                        if not texts:
                            st.error("No text found in file.")
                        else:
                            st.info(f"Processing {len(texts)} texts...")
                            results = []
                            progress_bar = st.progress(0)

                            for i, text in enumerate(texts):
                                prediction, probabilities = make_prediction(text, model_choice, models)

                                if prediction is not None and probabilities is not None:
                                    results.append({
                                        'Text': text[:100] + "..." if len(text) > 100 else text,
                                        'Full_Text': text,
                                        'Prediction': prediction,
                                        'Confidence': f"{max(probabilities):.1%}",
                                        'Human_Prob': f"{probabilities[0]:.1%}",
                                        'AI_Prob': f"{probabilities[1]:.1%}"
                                    })

                                progress_bar.progress((i + 1) / len(texts))

                            if results:
                                st.success(f"✅ Processed {len(results)} texts successfully!")

                                results_df = pd.DataFrame(results)

                                # Summary statistics
                                st.subheader("📊 Summary Statistics")
                                col1, col2, col3, col4 = st.columns(4)

                                human_count = sum(1 for r in results if r['Prediction'] == 'Human')
                                ai_count = len(results) - human_count
                                avg_conf = np.mean([float(r['Confidence'].strip('%')) for r in results])

                                with col1:
                                    st.metric("Total Processed", len(results))
                                with col2:
                                    st.metric("🧠 Human", human_count)
                                with col3:
                                    st.metric("🤖 AI", ai_count)
                                with col4:
                                    st.metric("Avg Confidence", f"{avg_conf:.1f}%")

                                # Preview
                                st.subheader("📋 Results Preview")
                                st.dataframe(results_df[['Text', 'Prediction', 'Confidence']], use_container_width=True)

                                # Download full CSV
                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Full Results",
                                    data=csv,
                                    file_name=f"predictions_{model_choice}_{uploaded_file.name}",
                                    mime="text/csv"
                                )
                            else:
                                st.error("No valid predictions could be made.")
                    except Exception as e:
                        st.error(f"❌ Error processing file: {e}")
            else:
                st.info("Please upload a file to begin.")

                with st.expander("📄 Example File Formats"):
                    st.markdown("""
                    **Text File (.txt):**
                    ```
                    The experiment was carefully planned.
                    This content was generated by an AI.
                    The results show significant improvement.
                    ```

                    **CSV File (.csv):**
                    ```
                    text,source
                    "AI-generated text goes here.",ai
                    "Human-written essay content.",human
                    ```
                    """)
        else:
            st.error("❌ No models available for batch processing.")
    else:
        st.warning("⚠️ Models not loaded. Please check model files.")

# ============================================================================\n",
# MODEL COMPARISON PAGE
# ============================================================================
elif page == "⚖️ Model Comparison":
    st.header("⚖️ Compare Models")
    st.markdown("Compare predictions from different models on the same text input.")

    if models:
        available_models = get_available_models(models)

        if len(available_models) >= 2:
            comparison_text = st.text_area(
                "Enter text to compare models:",
                placeholder="Type or paste a text to analyze...",
                height=100
            )

            if st.button("📊 Compare All Models") and comparison_text.strip():
                st.subheader("🔍 Model Comparison Results")

                comparison_results = []

                for model_key, model_name in available_models:
                    prediction, probabilities = make_prediction(comparison_text, model_key, models)

                    if prediction is not None and probabilities is not None:
                        comparison_results.append({
                            'Model': model_name,
                            'Prediction': prediction,
                            'Confidence': f"{max(probabilities):.1%}",
                            'Human %': f"{probabilities[0]:.1%}",
                            'AI %': f"{probabilities[1]:.1%}",
                            'Raw_Probs': probabilities
                        })

                if comparison_results:
                    comparison_df = pd.DataFrame(comparison_results)

                    st.table(comparison_df[['Model', 'Prediction', 'Confidence', 'Human %', 'AI %']])

                    # Agreement analysis
                    predictions = [r['Prediction'] for r in comparison_results]
                    if len(set(predictions)) == 1:
                        st.success(f"✅ All models agree: **{predictions[0]}**")
                    else:
                        st.warning("⚠️ Models disagree on the prediction:")
                        for result in comparison_results:
                            st.write(f"- {result['Model']}: **{result['Prediction']}**")

                    # Side-by-side probability charts
                    st.subheader("📊 Probability Comparison by Model")
                    cols = st.columns(len(comparison_results))

                    for i, result in enumerate(comparison_results):
                        with cols[i]:
                            st.markdown(f"**{result['Model']}**")
                            chart_df = pd.DataFrame({
                                'Label': ['Human', 'AI'],
                                'Probability': result['Raw_Probs']
                            })
                            st.bar_chart(chart_df.set_index('Label'))
                else:
                    st.error("❌ Failed to get predictions from models.")
        elif len(available_models) == 1:
            st.info("Only one model available. Use the 'Single Prediction' page for analysis.")
        else:
            st.error("❌ No models available for comparison.")
    else:
        st.warning("⚠️ Models not loaded. Please check the model files.")

# ============================================================================\n",
# MODEL INFO PAGE
# ============================================================================
elif page == "📊 Model Info":
    st.header("📊 Model Information")

    if models:
        st.success("✅ Models are loaded and ready!")

        # Model details
        st.subheader("🔧 Available Models")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### 📈 Support Vector Machine (SVM)
            **Type:** Linear classifier (with probability estimates)

            **Features:** TF-IDF (unigrams + bigrams)

            **Strengths:**
            - High accuracy on complex boundaries
            - Robust to overfitting in high-dimensional space
            - Performs well on small to medium datasets
            - Good for text classification with clear margins
            """)

            st.markdown("""
            ### 🎯 Decision Tree
            **Type:** Non-linear tree-based classifier

            **Features:** TF-IDF (unigrams + bigrams)

            **Strengths:**
            - Interpretable logic
            - Handles both linear and non-linear data
            - Works without feature scaling
            - Fast training and prediction
            """)

        with col2:
            st.markdown("""
            ### ⚡ AdaBoost (Adaptive Boosting)
            **Type:** Ensemble meta-classifier

            **Base Estimator:** Decision Tree (stumps)

            **Strengths:**
            - Combines weak learners into strong classifier
            - Great performance on imbalanced data
            - Focuses on hard-to-classify examples
            - Works well with clean data
            """)

        # Feature engineering info
        st.subheader("🔤 Feature Engineering")
        st.markdown("""
        **Vectorization Method:** TF-IDF (Term Frequency–Inverse Document Frequency)

        - **N-grams:** Unigrams and Bigrams
        - **Max Features:** 5000
        - **Min Document Frequency:** 2
        - **Stop Words Removed:** English
        """)

        # File status
        st.subheader("📁 Model Files Status")

        file_status = []
        files_to_check = [
            ("AI_vs_Human_analyzer_pipeline.pkl", "Complete SVM Pipeline", models.get('pipeline_available', False)),
            ("tfidf_vectorizer.pkl", "TF-IDF Vectorizer", models.get('vectorizer_available', False)),
            ("svm_model.pkl", "SVM Classifier", models.get('svm_available', False)),
            ("tree_model.pkl", "Decision Tree Classifier", models.get('tree_available', False)),
            ("ada_model.pkl", "AdaBoost Classifier", models.get('ada_available', False)),
        ]

        for filename, description, status in files_to_check:
            file_status.append({
                "File": filename,
                "Description": description,
                "Status": "✅ Loaded" if status else "❌ Not Found"
            })

        st.table(pd.DataFrame(file_status))

        # Training information
        st.subheader("📚 Training Information")
        st.markdown("""
        **Dataset:** Essay classification — Human vs AI-written

        - **Classes:** Human, AI
        - **Preprocessing:** Lowercasing, punctuation removal, stopword filtering, stemming/lemmatization
        - **Vectorization:** TF-IDF with n-grams
        - **Evaluation:** Accuracy, Precision, Recall, F1-score, ROC AUC
        """)
    else:
        st.warning("Models not loaded. Please check model files in the 'models/' directory.")
# ============================================================================\n",
# HELP PAGE
# ============================================================================
elif page == "❓ Help":
    st.header("❓ How to Use This App")

    with st.expander("🔮 Single Prediction"):
        st.write("""
        1. **Select a model** from the dropdown (SVM, AdaBoost, or Decision Tree)
        2. **Enter or paste your text** in the input box (e.g., product reviews, comments)
        3. **Click 'Predict'** to analyze the sentiment
        4. **View results:** You’ll see a prediction label, confidence score, and probability breakdown
        5. **Try examples:** Use built-in examples to quickly test the model
        """)

    with st.expander("📁 Batch Processing"):
        st.write("""
        1. **Prepare your file:**
           - **.txt file:** One line per text
           - **.csv file:** Texts should be in the first column
        2. **Upload your file**
        3. **Choose a model** for sentiment classification
        4. **Click 'Process File'** to analyze all texts
        5. **Download results** as a CSV including predictions and confidence scores
        """)

    with st.expander("⚖️ Model Comparison"):
        st.write("""
        1. **Enter your text** to compare models' performance
        2. **Click 'Compare All Models'**
        3. **View the comparison table** showing predictions, confidence levels, and sentiment probabilities
        4. **Analyze agreement:** See whether the models agree or disagree
        5. **Visualize:** Compare predictions side-by-side using bar charts
        """)

    with st.expander("🔧 Troubleshooting"):
        st.write("""
        **Common Issues and How to Fix Them:**

        **Models not loading:**
        - Ensure required model files exist in the `models/` directory:
          - `AI_vs_Human_analyzer_pipeline.pkl` (if using full pipeline)
          - `svm_model.pkl`, `tree_model.pkl`, `ada_model.pkl`
          - `tfidf_vectorizer.pkl` (essential for predictions)

        **Prediction not working:**
        - Input must not be empty
        - Try shorter or simpler text if getting unusual errors
        - Ensure text is in UTF-8 and contains readable characters

        **Upload errors:**
        - Only `.txt` or `.csv` files are supported
        - CSVs must have text in the **first column**
        - Files should be encoded in UTF-8
        """)

    # System info
    st.subheader("💻 Project Structure")
    st.code("""
    streamlit_ml_app/
    ├── app.py                         # Main app file
    ├── requirements.txt              # Python dependencies
    ├── models/                       # Trained model files
    │   ├── AI_vs_Human_analyzer_pipeline.pkl
    │   ├── svm_model.pkl
    │   ├── tree_model.pkl
    │   ├── ada_model.pkl
    │   └── tfidf_vectorizer.pkl
    └── sample_data/                 # Sample text/CSV files
        ├── sample_texts.txt
        └── sample_data_
""")
# ============================================================================\n",
# FOOTER
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 App Information")
st.sidebar.info("""
**ML Text Classification App**  
Built with Streamlit

**Models:**  
- 📈 Support Vector Machine (SVM)  
- 🎯 AdaBoost  
- 🌳 Decision Tree

**Framework:** scikit-learn  
**Deployment:** Streamlit Cloud Ready
""")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666666;'>
    Built with ❤️ using Streamlit | Machine Learning Text Classification Demo | By Michael Djeumo<br>
    <small>As a part of the courses series <b>Introduction to Large Language Models / Intro to AI Agents</b></small><br>
    <small>This app demonstrates sentiment analysis using trained ML models</small>
</div>
""", unsafe_allow_html=True)

#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "id": "8954210f-16a3-461c-8ead-f00368733a49",
#    "metadata": {},
#    "outputs": [],
#    "source": []
#   }
#  ],
#  "metadata": {
#   "kernelspec": {
#    "display_name": "Python [conda env:base] *",
#    "language": "python",
#    "name": "conda-base-py"
#   },
#   "language_info": {
#    "codemirror_mode": {
#     "name": "ipython",
#     "version": 3
#    },
#    "file_extension": ".py",
#    "mimetype": "text/x-python",
#    "name": "python",
#    "nbconvert_exporter": "python",
#    "pygments_lexer": "ipython3",
#    "version": "3.12.7"
#   }
#  },
#  "nbformat": 4,
#  "nbformat_minor": 5
# }
