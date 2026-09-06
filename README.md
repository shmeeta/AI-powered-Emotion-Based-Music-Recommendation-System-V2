# AI-Powered Emotion-Based Music Recommendation System

> **Honours Research Project** | *A full-stack hybrid recommendation engine integrating emotion classification with content-based and collaborative filtering.*

[![Python](https://img.shields.io/badge/Python-3.12.6-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-5.0.6-092E20?style=flat&logo=django&logoColor=white)](https://www.djangoproject.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15.4-4169E1?style=flat&logo=postgresql&logoColor=white)](https://www.postgresql.org/)

---

##  Research & System Overview



This research project investigates the effect of incorporating a user's current emotional state into music recommendation systems to evaluate its impact on overall user satisfaction. 

Built as a full-stack web application using Django and PostgreSQL, the system blends emotion classification with a dual-stage recommendation approach:
1. **Emotion Context Layer:** Maps user emotional states to specific musical audio features. Fine-tuned distilBERT for emotion classification.
2. **Content-Based Filtering:** Matches track feature vectors (such as tempo, valence, and energy) to the target emotional state.
3. **Collaborative Filtering:** Refines predictions using user similarity matrices.

---

## Tech Stack & Environment

* **Backend:** Python 3.12.6, Django 5.0.6
* **Database:** PostgreSQL 15.4 (pgAdmin 7.5 / pgAdmin 4)
* **Frontend:** HTML5, CSS3 (Bootstrap), JavaScript
* **Data Processing & ML:** Google Colab (Dataset preprocessing, model training, and feature visualization)
* **UI/UX Design:** Canva
* **External API Integration:** Spotify Web API (Track metadata and album artwork)

---

## Post-Implementation Engineering Audit (V1 vs. V2 Roadmap)

The initial release (V1) served as a functional prototype to validate core machine learning performance, dataset feature engineering and recommendation accuracy. Since then, key architectural bottlenecks and execution overhead have been identified. The system is currently undergoing a v2 refactor focused on scalability, system decoupling and asynchronous execution.

### 1. Monolithic View Logic
* **V1 State:** User authentication, session validation and recommendation matrix calculations were implemented in one Django app and view. This led to high code complexity and unnecessary execution overhead during basic request routing.

* **V2 Architecture:** Separate views for each portion of the website. Restructure the application layer into smaller, dedicated helper functions so the code is cleaner, easier to test and simpler to manage.

### 2. Synchronous Request-Response Thread Blocking.
* **V1 State:** Core Machine Learning operations, such as NLP transformer operations, vector math and dynamic clustering are executed synchronously inside the web application request cycle, leading to High HTTP latency and susceptibility to server timeouts during heavy inference tasks.

* **V2 Architecture:** Decouple computation-heavy ML routines from the HTTP server loop by offloading execution to a background worker process and implementing an asynchronous task-tracking pattern.

### 3. External API Network Overhead
* **Current V1 State:** Album cover fetching via the Spotify API occurs synchronously during page rendering. Web server response times are therefore directly tied to third-party API rate limits and network latency.
* **V2 Architecture:** Implement client-side asynchronous fetching and persist image URLs directly in PostgreSQL to reduce third-party HTTP calls.

---

---



