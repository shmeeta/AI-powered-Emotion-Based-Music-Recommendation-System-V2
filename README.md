# 🎵 AI-Powered Emotion-Based Music Recommendation System

> **Honours Research Project** | *A full-stack hybrid recommendation engine integrating emotion classification with content-based and collaborative filtering.*

[![Python](https://img.shields.io/badge/Python-3.12.6-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-5.0.6-092E20?style=flat&logo=django&logoColor=white)](https://www.djangoproject.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15.4-4169E1?style=flat&logo=postgresql&logoColor=white)](https://www.postgresql.org/)

---

## 📌 Research & System Overview



This research project investigates the effect of incorporating a user's current emotional state into music recommendation systems to evaluate its impact on overall user satisfaction. 

Built as a full-stack web application using Django and PostgreSQL, the system blends emotion classification with a dual-stage recommendation approach:
1. **Emotion Context Layer:** Maps user emotional states to specific musical audio features. Fine-tuned distilBERT for emotion classification.
2. **Content-Based Filtering:** Matches track feature vectors (such as tempo, valence, and energy) to the target emotional state.
3. **Collaborative Filtering:** Refines predictions using user similarity matrices.

---

## 🛠️ Tech Stack & Environment

* **Backend:** Python 3.12.6, Django 5.0.6
* **Database:** PostgreSQL 15.4 (pgAdmin 7.5 / pgAdmin 4)
* **Frontend:** HTML5, CSS3 (Bootstrap), JavaScript
* **Data Processing & ML:** Google Colab (Dataset preprocessing, model training, and feature visualization)
* **UI/UX Design:** Canva
* **External API Integration:** Spotify Web API (Track metadata and album artwork)

---

## 🏗️ Post-Implementation Engineering Audit (V1 vs. V2 Roadmap)

The V1 release was designed as a functional prototype to validate the core research premise and evaluate recommendation accuracy. Following a post-implementation code audit, several key optimization targets were identified for better V2) architecture:

### 1. Monolithic View Logic
* **V1 State:** Core user authentication, session management, liked songs state, and matrix-based recommendation generation are handled within unified view functions.
* **Impact:** Tight coupling and unnecessary execution overhead during simple routing or authentication steps.
* **V2 Architecture:** Separate views for each portion of the website.



### 3. External API Network Overhead
* **Current V1 State:** Album cover fetching via the Spotify API occurs synchronously during page rendering.
* **V2 Architecture:** Implement client-side asynchronous fetching and persist image URLs directly in PostgreSQL to reduce third-party HTTP calls.

---

---

## 📄 Documentation & Research Paper

The complete theoretical background, dataset preparation, algorithm evaluations, and user satisfaction results are detailed in the full research paper:

---

## ⚙️ Quick Start & Local Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/shmeeta/AI-powered-Emotion-Based-Music-Recommendation-System-V2.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
   cd AI-powered-Emotion-Based-Music-Recommendation-System-V2
