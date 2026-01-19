<div align="center">
  <h1><b>Personalized Workout Recommender</b></h1>
</div>
<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest-orange.svg)
![Deployment](https://img.shields.io/badge/Deployment-Vercel-black.svg)

**An AI-powered fitness recommendation system that provides personalized workout type suggestions based on user characteristics and fitness goals.**

[Features](#-features) • [Quick Start](#-quick-start) • [API Documentation](#-api-documentation) • [Model Details](#-model-details) • [Deployment](#-deployment)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Technology Stack](#-technology-stack)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [API Documentation](#-api-documentation)
- [Model Details](#-model-details)
- [Deployment](#-deployment)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

The Personalized Workout Recommender is an intelligent system that analyzes user demographics, fitness goals, and experience levels to recommend the optimal workout type. Built with machine learning and deployed as a serverless application, it provides real-time, personalized fitness recommendations through a natural language interface.

### Workout Types

The system recommends one of three workout categories:

- **🏃‍♂️ Endurance Training** - Focus on stamina, heart health, and fat burn
- **💪 Muscle Building** - Focus on strength training, toning, and muscle growth
- **🔄 Balanced Fitness** - Combination approach for overall fitness and health

---

## ✨ Features

- 🤖 **AI-Powered Classification** - Multi-class Random Forest model with 100% accuracy
- 💬 **Natural Language Interface** - Users can describe themselves in plain English
- ⚡ **Real-Time Predictions** - Instant recommendations via serverless API
- 📱 **Responsive Design** - Works seamlessly on desktop, tablet, and mobile devices
- 🎯 **Personalized Recommendations** - Considers age, gender, BMI, goals, and experience
- 📊 **Confidence Scores** - Provides probability distributions for all workout types
- 🔧 **Feature Engineering** - 22 engineered features from user characteristics
- 🚀 **Production Ready** - Deployed on Vercel with serverless architecture

---

## 🛠️ Technology Stack

### Machine Learning
- **Algorithm**: Random Forest Classifier (100 decision trees)
- **Framework**: scikit-learn
- **Accuracy**: 100% on test dataset
- **Training Data**: 1,800 workout plans

### Backend
- **Language**: Python 3.8+
- **API Framework**: Flask (local), Serverless Functions (Vercel)
- **Data Processing**: pandas, numpy
- **Feature Engineering**: Custom preprocessing pipeline

### Frontend
- **HTML5/CSS3** - Modern, responsive interface
- **JavaScript** - Interactive UI and API integration
- **Design**: Gradient-based modern UI with smooth animations

### Deployment
- **Platform**: Vercel Serverless
- **Infrastructure**: Serverless functions for scalable API

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/Personalized-Workout-Recommender.git
   cd Personalized-Workout-Recommender
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (one-time setup)
   ```bash
   cd src
   python model.py
   cd ..
   ```

4. **Start the local server**
   ```bash
   python app.py
   ```

5. **Open your browser**
   ```
   http://localhost:5000
   ```

### Example Usage

Enter a natural language description of yourself and your fitness goals:

```
I'm a 25-year-old male, 180 cm tall and weigh 75 kg. 
I want to build muscle and get stronger. 
I've been going to the gym for a few months.
```

The system will return:
- Recommended workout type
- Confidence score
- Probability distribution for all three types
- Personalized fitness advice

---

## 📁 Project Structure

```
Personalized-Workout-Recommender/
│
├── api/                          # Serverless API functions
│   └── predict.py               # Vercel serverless function
│
├── src/                          # Source code
│   ├── model.py                 # Model training script
│   └── data_preprocessing.py    # Feature engineering pipeline
│
├── data/                         # Dataset
│   ├── train.csv                # Training data
│   └── train_sample.csv         # Sample data
│
├── models/                       # Trained models
│   ├── workout_model.pkl        # Serialized Random Forest model
│   └── feature_names.pkl        # Feature names for inference
│
├── templates/                    # Flask templates
│   └── index.html               # Frontend interface
│
├── app.py                        # Flask application (local development)
├── index.html                    # Frontend (Vercel deployment)
├── requirements.txt              # Python dependencies
├── package.json                  # Project metadata
└── README.md                     # Project documentation
```

---

## 📚 API Documentation

### Endpoint: `/predict`

**Method**: `POST`

**Request Body**:
```json
{
  "prompt": "I'm a 25-year-old male, 180 cm tall, weigh 75 kg. I want to build muscle and I'm a beginner."
}
```

**Response**:
```json
{
  "status": "success",
  "prediction": {
    "workout_type": "Muscle Building",
    "technical_type": "strength",
    "confidence": 85.67,
    "probabilities": {
      "Endurance Training": 10.23,
      "Balanced Fitness": 4.10,
      "Muscle Building": 85.67
    }
  },
  "user_input": "I'm a 25-year-old male..."
}
```

### Endpoint: `/health`

**Method**: `GET`

**Response**:
```json
{
  "status": "healthy",
  "message": "Workout Recommender API is running",
  "model_loaded": true
}
```

---

## 🧠 Model Details

### Architecture

- **Algorithm**: Random Forest Classifier
- **Trees**: 100 decision trees
- **Features**: 22 engineered features
- **Classes**: 3 (Endurance Training, Muscle Building, Balanced Fitness)
- **Optimization**: GridSearchCV with cross-validation

### Feature Engineering

The model uses 22 features extracted from user input:

**Primary Features:**
- Age, Height, Weight, Gender
- Fitness Goal, Gym Experience Level
- Calculated BMI

**Derived Features:**
- Exercise variety metrics
- Workout frequency and intensity
- Volume and repetition patterns
- Rest day distribution

### Performance Metrics

- **Training Accuracy**: 100%
- **Test Accuracy**: 100%
- **Dataset Size**: 1,800 workout plans
- **Class Balance**: Balanced across all three workout types

### Model Training

```bash
cd src
python model.py
```

This script will:
1. Load and preprocess the training data
2. Engineer features from raw input
3. Split data into training and testing sets
4. Tune hyperparameters using GridSearchCV
5. Train the Random Forest model
6. Evaluate performance and save the model

---

## 🌐 Deployment

### Local Development

```bash
python app.py
```

The Flask server will start on `http://localhost:5000`

### Vercel Deployment

1. **Install Vercel CLI** (optional)
   ```bash
   npm i -g vercel
   ```

2. **Deploy to Vercel**
   ```bash
   vercel
   ```

3. **Files for Deployment**:
   - `index.html` - Frontend interface
   - `api/predict.py` - Serverless function
   - `requirements.txt` - Python dependencies
   - `package.json` - Project metadata

The serverless function will be automatically available at `/api/predict`

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**tharani-2006**

- GitHub: [@tharani-2006](https://github.com/tharani-2006)

---

## 🙏 Acknowledgments

- scikit-learn community for excellent ML tools
- Vercel for seamless serverless deployment
- Fitness community for inspiration and feedback

---

<div align="center">

**Made with ❤️ for the fitness community**

⭐ Star this repo if you find it helpful!

</div>
