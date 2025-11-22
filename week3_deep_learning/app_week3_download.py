"""
Week 3: Deep Learning Foundations - Downloadable Package

This Streamlit app provides:
1. Overview of all Week 3 content
2. Individual day downloads
3. Complete Week 3 ZIP download

Run with: streamlit run app_week3_download.py
"""

import streamlit as st
import os
import zipfile
import io
from pathlib import Path
import base64

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Week 3: Deep Learning Foundations",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# CONTENT DESCRIPTIONS
# =============================================================================

WEEK3_CONTENT = {
    "day15_neural_networks": {
        "title": "Day 15: Neural Networks Theory",
        "description": """
        **Topics Covered:**
        - Perceptron and activation functions (sigmoid, tanh, ReLU)
        - Forward propagation mathematics
        - Backpropagation and gradient descent
        - Loss functions: MSE, cross-entropy

        **Assignment:** Implement single-layer perceptron from scratch for AND/OR gates

        **Files:**
        - `perceptron_from_scratch.py`: Complete implementation with visualization
        """,
        "files": ["perceptron_from_scratch.py"],
        "concepts": ["Perceptron", "Activation Functions", "Forward Propagation", "Backpropagation", "Gradient Descent", "Loss Functions"]
    },
    "day16_tensorflow_keras": {
        "title": "Day 16: Introduction to TensorFlow/Keras",
        "description": """
        **Topics Covered:**
        - Install TensorFlow and understand the API
        - Sequential model vs Functional API
        - Layers: Dense, Dropout, BatchNormalization
        - Model compilation: optimizers, loss, metrics

        **Assignment:** Build 3-layer neural network for MNIST digit classification

        **Files:**
        - `mnist_neural_network.py`: Complete MNIST classifier with both APIs
        """,
        "files": ["mnist_neural_network.py"],
        "concepts": ["TensorFlow", "Keras", "Sequential API", "Functional API", "Dense Layer", "Dropout", "BatchNormalization"]
    },
    "day17_deep_networks": {
        "title": "Day 17: Deep Neural Networks",
        "description": """
        **Topics Covered:**
        - Architecture design: input → hidden layers → output
        - Regularization: L1, L2, dropout
        - Learning rate schedules
        - Early stopping and model checkpointing

        **Assignment:** Build deep network (5+ layers) for Fashion-MNIST, achieve >90% accuracy

        **Files:**
        - `fashion_mnist_deep.py`: 5+ layer networks with regularization
        """,
        "files": ["fashion_mnist_deep.py"],
        "concepts": ["Deep Networks", "L1/L2 Regularization", "Learning Rate Schedules", "Early Stopping", "Model Checkpointing"]
    },
    "day18_cnn_part1": {
        "title": "Day 18: Convolutional Neural Networks (CNN) - Part 1",
        "description": """
        **Topics Covered:**
        - Convolution operation: filters, kernels, feature maps
        - Pooling layers: max pooling, average pooling
        - CNN architecture patterns: Conv → Pool → Conv → Pool → Dense
        - Understanding why CNNs work for images

        **Assignment:** Build basic CNN for CIFAR-10 classification

        **Files:**
        - `cifar10_cnn.py`: Complete CNN implementation with visualization
        """,
        "files": ["cifar10_cnn.py"],
        "concepts": ["Convolution", "Filters", "Feature Maps", "Max Pooling", "Average Pooling", "CNN Architecture"]
    },
    "day19_transfer_learning": {
        "title": "Day 19: CNNs - Part 2 & Transfer Learning",
        "description": """
        **Topics Covered:**
        - Famous architectures: LeNet, AlexNet, VGG, ResNet
        - Transfer learning concept
        - Fine-tuning pre-trained models
        - Data augmentation techniques

        **Assignment:** Use pre-trained ResNet50 for custom image classification (cats vs dogs)

        **Files:**
        - `cats_vs_dogs_resnet.py`: Transfer learning with ResNet50/MobileNetV2
        """,
        "files": ["cats_vs_dogs_resnet.py"],
        "concepts": ["LeNet", "AlexNet", "VGG", "ResNet", "Transfer Learning", "Fine-tuning", "Data Augmentation"]
    },
    "day20_rnn": {
        "title": "Day 20: Recurrent Neural Networks (RNN)",
        "description": """
        **Topics Covered:**
        - Sequential data and time series
        - RNN architecture and vanishing gradient problem
        - LSTM and GRU units
        - Bidirectional RNNs

        **Assignment:** Build sentiment analysis model for movie reviews (IMDB dataset)

        **Files:**
        - `imdb_sentiment_rnn.py`: LSTM/GRU/BiLSTM sentiment analysis
        """,
        "files": ["imdb_sentiment_rnn.py"],
        "concepts": ["RNN", "Vanishing Gradient", "LSTM", "GRU", "Bidirectional RNN", "Sentiment Analysis"]
    },
    "day21_project": {
        "title": "Day 21: Week 3 Project",
        "description": """
        **Topics Covered:**
        - Computer vision project: build image classifier for 10 classes
        - Implement from scratch and with transfer learning
        - Compare performance and training time
        - Deploy model using Streamlit

        **Assignment:** Complete image classifier web app with UI

        **Files:**
        - `image_classifier_app.py`: Full Streamlit app for image classification
        """,
        "files": ["image_classifier_app.py"],
        "concepts": ["Full Project", "Model Comparison", "Web Deployment", "Streamlit"]
    }
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_file_content(filepath):
    """Read file content."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"


def create_day_zip(day_folder, day_name):
    """Create a ZIP file for a specific day."""
    zip_buffer = io.BytesIO()

    base_path = Path(__file__).parent / day_folder

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        if base_path.exists():
            for file_path in base_path.rglob('*'):
                if file_path.is_file() and not file_path.name.startswith('.'):
                    arc_name = f"{day_name}/{file_path.relative_to(base_path)}"
                    zip_file.write(file_path, arc_name)

    zip_buffer.seek(0)
    return zip_buffer


def create_complete_zip():
    """Create a ZIP file with all Week 3 content."""
    zip_buffer = io.BytesIO()

    base_path = Path(__file__).parent

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add all day folders
        for day_folder in WEEK3_CONTENT.keys():
            day_path = base_path / day_folder
            if day_path.exists():
                for file_path in day_path.rglob('*'):
                    if file_path.is_file() and not file_path.name.startswith('.'):
                        arc_name = f"week3_deep_learning/{day_folder}/{file_path.relative_to(day_path)}"
                        zip_file.write(file_path, arc_name)

        # Add requirements file
        req_path = base_path / 'requirements_week3.txt'
        if req_path.exists():
            zip_file.write(req_path, 'week3_deep_learning/requirements.txt')

        # Add README
        readme_path = base_path / 'README_week3.md'
        if readme_path.exists():
            zip_file.write(readme_path, 'week3_deep_learning/README.md')

        # Add this download app
        app_path = base_path / 'app_week3_download.py'
        if app_path.exists():
            zip_file.write(app_path, 'week3_deep_learning/app_week3_download.py')

    zip_buffer.seek(0)
    return zip_buffer


def get_download_link(file_bytes, filename, text):
    """Generate a download link for binary data."""
    b64 = base64.b64encode(file_bytes.read()).decode()
    return f'<a href="data:application/zip;base64,{b64}" download="{filename}">{text}</a>'


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Title
    st.title(" Week 3: Deep Learning Foundations")

    st.markdown("""
    Welcome to the **Week 3 Deep Learning Foundations** curriculum!
    This comprehensive package covers neural networks, CNNs, transfer learning, and RNNs.
    """)

    # Sidebar navigation
    st.sidebar.header(" Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Overview", "Day 15", "Day 16", "Day 17", "Day 18", "Day 19", "Day 20", "Day 21", "Download All"]
    )

    if page == "Overview":
        overview_page()
    elif page == "Download All":
        download_page()
    else:
        day_num = page.split()[-1]
        day_key = f"day{day_num}_" + list(WEEK3_CONTENT.keys())[int(day_num) - 15].split('_', 1)[1]
        day_page(day_key)


def overview_page():
    """Show overview of all days."""
    st.header(" Course Overview")

    st.markdown("""
    ### Week 3 Learning Path

    This week covers the foundations of deep learning, from basic neural networks to
    advanced architectures like CNNs and RNNs.
    """)

    # Week summary
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### Days 15-17: Neural Network Fundamentals
        - Day 15: Perceptrons and activation functions
        - Day 16: TensorFlow/Keras introduction
        - Day 17: Deep networks and regularization
        """)

    with col2:
        st.markdown("""
        #### Days 18-21: Advanced Architectures
        - Day 18: CNN basics for image classification
        - Day 19: Transfer learning with pre-trained models
        - Day 20: RNNs for sequential data
        - Day 21: Complete project deployment
        """)

    st.divider()

    # Day cards
    st.subheader(" Daily Content")

    for i, (day_key, day_info) in enumerate(WEEK3_CONTENT.items()):
        with st.expander(f" {day_info['title']}", expanded=False):
            st.markdown(day_info['description'])

            # Show concepts as tags
            st.write("**Key Concepts:**")
            cols = st.columns(4)
            for j, concept in enumerate(day_info['concepts']):
                cols[j % 4].code(concept)


def day_page(day_key):
    """Show detailed page for a specific day."""
    day_info = WEEK3_CONTENT[day_key]

    st.header(day_info['title'])
    st.markdown(day_info['description'])

    # Key concepts
    st.subheader(" Key Concepts")
    cols = st.columns(4)
    for i, concept in enumerate(day_info['concepts']):
        cols[i % 4].info(concept)

    st.divider()

    # Code files
    st.subheader(" Source Code")

    base_path = Path(__file__).parent / day_key

    for filename in day_info['files']:
        filepath = base_path / filename

        with st.expander(f" {filename}", expanded=True):
            if filepath.exists():
                code = get_file_content(filepath)
                st.code(code, language='python')

                # Download individual file
                st.download_button(
                    label=f" Download {filename}",
                    data=code,
                    file_name=filename,
                    mime="text/plain"
                )
            else:
                st.warning(f"File not found: {filepath}")

    st.divider()

    # Day ZIP download
    st.subheader(f" Download {day_info['title']}")

    zip_buffer = create_day_zip(day_key, day_key)

    st.download_button(
        label=f" Download {day_key}.zip",
        data=zip_buffer,
        file_name=f"{day_key}.zip",
        mime="application/zip"
    )


def download_page():
    """Page for downloading complete package."""
    st.header(" Download Complete Package")

    st.markdown("""
    ### Download all Week 3 content as a single ZIP file

    The complete package includes:
    - All 7 days of content (Day 15-21)
    - Python source files with detailed comments
    - Requirements file
    - README documentation
    """)

    # Package contents
    st.subheader(" Package Contents")

    contents_md = """
    ```
    week3_deep_learning/
    ├── day15_neural_networks/
    │   └── perceptron_from_scratch.py
    ├── day16_tensorflow_keras/
    │   └── mnist_neural_network.py
    ├── day17_deep_networks/
    │   └── fashion_mnist_deep.py
    ├── day18_cnn_part1/
    │   └── cifar10_cnn.py
    ├── day19_transfer_learning/
    │   └── cats_vs_dogs_resnet.py
    ├── day20_rnn/
    │   └── imdb_sentiment_rnn.py
    ├── day21_project/
    │   └── image_classifier_app.py
    ├── app_week3_download.py
    ├── requirements.txt
    └── README.md
    ```
    """
    st.markdown(contents_md)

    st.divider()

    # Download button
    st.subheader(" Download")

    zip_buffer = create_complete_zip()

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.download_button(
            label=" Download week3_deep_learning.zip",
            data=zip_buffer,
            file_name="week3_deep_learning.zip",
            mime="application/zip",
            use_container_width=True
        )

    st.divider()

    # Installation instructions
    st.subheader(" Installation Instructions")

    st.markdown("""
    1. **Download and extract** the ZIP file

    2. **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\\Scripts\\activate
    ```

    3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

    4. **Run individual scripts**:
    ```bash
    python day15_neural_networks/perceptron_from_scratch.py
    python day16_tensorflow_keras/mnist_neural_network.py
    # ... etc
    ```

    5. **Run the Streamlit apps**:
    ```bash
    streamlit run day21_project/image_classifier_app.py
    streamlit run app_week3_download.py
    ```
    """)

    # System requirements
    st.subheader(" System Requirements")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Minimum Requirements:**
        - Python 3.8+
        - 8GB RAM
        - 5GB disk space

        **Recommended:**
        - Python 3.10+
        - 16GB RAM
        - NVIDIA GPU with CUDA support
        """)

    with col2:
        st.markdown("""
        **Key Dependencies:**
        - TensorFlow 2.13+
        - NumPy
        - Matplotlib
        - Scikit-learn
        - Streamlit
        - Seaborn
        """)


if __name__ == "__main__":
    main()
