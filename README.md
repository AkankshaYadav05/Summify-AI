# 🚀 Summify AI

> AI-powered multi-mode summarization platform for **Text, Dialogue, and Documents** built using **FastAPI** and **Hugging Face Transformers**.

Summify AI is an intelligent summarization platform that generates concise, meaningful summaries from plain text, conversations, and uploaded documents. It combines a **FLAN-T5-base** model for general summarization with a **T5-small model fine-tuned on the SAMSum dataset** for dialogue summarization, providing accurate summaries in multiple output formats.

---

## ✨ Features

### 📝 Text Summarization
- Summarize articles, paragraphs, reports, and long-form text.
- Powered by **Google FLAN-T5-base**.
- Automatic text preprocessing and cleaning.
- Handles lengthy content through chunk-based summarization.

---

### 💬 Dialogue Summarization
- Fine-tuned **T5-small** model on **14,000+ dialogue-summary pairs** from the **SAMSum dataset**.
- Optimized for conversations such as:
  - Meetings
  - Interviews
  - Chats
  - Discussions

---

### 📄 Document Summarization
Supports document upload in multiple formats:

- PDF
- DOCX
- DOC
- TXT
- Markdown (.md)

Features:
- Text extraction using **PyMuPDF**
- Automatic scanned page detection
- Document cleaning and preprocessing
- Summary generation using the same FLAN-T5-base pipeline

---

## 📋 Summary Formats

Each summarization mode (**Text, Dialogue, and Documents**) supports three output styles:

### ✅ Short Summary
Generates a concise paragraph capturing the key information.

### ✅ Bullet Points
Converts the content into easy-to-read bullet points.

Example:

```
• Main topic
• Important information
• Key findings
• Final outcome
```

### ✅ Meeting Notes
Produces structured notes including:

```
Overview

Key Discussion

Action Items

Conclusion
```

---

## 🧠 AI Models

### FLAN-T5-base
Used for:

- Text Summarization
- Document Summarization

Capabilities:

- Instruction-tuned transformer
- High-quality abstractive summaries
- Long text chunking support

---

### T5-small (Fine-tuned)

Used for:

- Dialogue Summarization

Training Details:

- Model: **T5-small**
- Dataset: **SAMSum**
- Dataset Size: **14,000+ dialogue-summary pairs**
- Framework: **Hugging Face Transformers**
- Fine-tuned using **PyTorch**

Designed specifically for conversational summarization.

---

## ⚙️ Tech Stack

### Backend
- FastAPI
- Uvicorn
- Jinja2

### AI / NLP
- Hugging Face Transformers
- FLAN-T5-base
- T5-small
- PyTorch

### Document Processing
- PyMuPDF
- python-docx

### Frontend
- HTML
- CSS
- JavaScript

---

## 📂 Project Structure

```
Summify-AI/
│
├── app.py
├── requirements.txt
│
├── routers/
├── services/
├── templates/
├── static/
├── text_summary_model/
└── README.md
```

---

## 🚀 Installation

### Clone the repository

```bash
git clone https://github.com/AkankshaYadav05/Summify-AI.git
```

### Navigate to the project

```bash
cd Summify-AI
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
uvicorn app:app --reload
```

Open your browser:

```
http://127.0.0.1:8000
```

---

## 📦 Dependencies

- FastAPI
- Uvicorn
- Transformers
- PyTorch
- PyMuPDF
- python-docx
- Jinja2

---

## 📈 Future Enhancements

- YouTube video summarization
- Website & article summarization
- OCR support for scanned documents
- Multi-language summarization
- Export summaries as PDF or DOCX
- User authentication
- Docker deployment

---

## 📸 Screenshots

Include screenshots of:

- Home Page
- Text Summarizer
- Dialogue Summarizer
- Document Summarizer
- Generated Summary

---

## 👩‍💻 Author

**Akanksha Yadav**

B.Tech Computer Science & Engineering

- AI/ML Enthusiast
- Full Stack Developer
- Java Developer

### GitHub

**Repository:**  
https://github.com/AkankshaYadav05/Summify-AI

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!
