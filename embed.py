# embed.py
import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

# ------------- Customize: your KB -------------
knowledge_base = [
    {
        "id": "profile",
        "title": "Profile summary",
        "text": (
            "Master of Science in Engineering in AI and Machine Learning at BTH (4th year). "
            "Strong programming skills in Python, C and Java. Experienced in machine learning, "
            "security in AI systems, genetic algorithms and object tracking (YOLO). Familiar with "
            "cloud technologies (Docker, GCP) and GitHub. Practical experience maintaining ERP systems "
            "and automotive/mechanical work. Interested in ML systems, software engineering and secure AI."
        ),
    },
    {
        "id": "contact",
        "title": "Contact info",
        "text": (
            "Contact information: Email: omar.dalal.2001@gmail.com. Phone: +46 76-9481560. "
            "Address: Kungsmarksvägen 7, 37144 Karlskrona, Sweden."
        ),
    },
    # Education
    {
        "id": "education_bth",
        "title": "Education — Blekinge Institute of Technology (BTH)",
        "text": (
            "M.Sc. in Engineering — AI & Machine Learning at Blekinge Institute of Technology (BTH). "
            "Study period: 2021 – expected 2026."
        ),
    },
    {
        "id": "education_highschool",
        "title": "Education — Technology Programme (High School)",
        "text": "Completed the Technology Programme in upper secondary school (Technology programme).",
    },

    # Work experience — separate items for each role
    {
        "id": "humly",
        "title": "Humly Solutions AB — QA & Technical Support (Workplace Solutions)",
        "text": (
            "Humly Solutions AB — December 2024 – June 2026. Responsible for quality assurance (QA) "
            "of new systems and firmware, performing thorough tests of software and hardware before production. "
            "Acted as a technical key resource in customer support handling troubleshooting, system analysis "
            "and guidance on system functionality and API integrations. Contributed across the stack from "
            "cloud-based backend services to physical devices."
        ),
    },
    {
        "id": "outliar",
        "title": "Outliar — Python Developer (AI company, remote)",
        "text": (
            "Outliar — February 2025 – June 2025. Developed advanced Python programs to improve AI models' "
            "abilities to write, understand and analyze code. Projects included code generation, semantic "
            "understanding, refactoring, and code analysis pipelines to improve model precision and usability."
        ),
    },
    {
        "id": "meliox",
        "title": "Meliox AB — AI Project Collaboration / Internship",
        "text": (
            "Meliox AB — January 2025 – June 2025. Led an AI project for sensor classification within smart "
            "building systems. Tasks included data collection, preprocessing, model selection, implementation "
            "and testing. The solution helped automate sensor identification and integration into company "
            "systems for energy management and automation."
        ),
    },
    {
        "id": "self_grocery",
        "title": "Self-employed & System Developer — Grocery store (Östra Göinge)",
        "text": (
            "Self-employed & System Developer — July 2021 – June 2025. Responsible for accounting, auditing, "
            "financial maintenance and declarations. Built and maintained a custom system for the grocery store "
            "to track inventory, incomes, expenses and other shop-specific functionality, and handled ongoing "
            "system maintenance."
        ),
    },
    {
        "id": "automotive_mechanic",
        "title": "Automotive Mechanic — Karlskrona",
        "text": (
            "Automotive Mechanic — June 2023 – June 2024. Diagnosed vehicles, identified and fixed problems, "
            "ordered and replaced parts, and ensured vehicles operated correctly through repairs and maintenance."
        ),
    },
    {
        "id": "assistant_nurse",
        "title": "Assistant Nurse (Night Patrol) — Karlskrona Municipality",
        "text": (
            "Assistant Nurse, Night Patrol — November 2022 – May 2023. Provided personal care, support and "
            "assistance to clients in their homes as part of home care services."
        ),
    },
    {
        "id": "patient_guard",
        "title": "Patient Guard — Hässleholm Hospital",
        "text": "Patient Guard — October 2020 – August 2021. Temporary role supporting patients with special needs in daily activities.",
    },
    {
        "id": "service_staff",
        "title": "Service Staff — Hässleholm Hospital",
        "text": "Service Staff — June 2020 – August 2020. Duties included transport responsibility in June and cleaning/facility maintenance in July and August.",
    },
    {
        "id": "support_hemfixare",
        "title": "Support Technician & Furniture Assembler — Hemfixare AB",
        "text": "Support Technician & Furniture Assembler — December 2019 – August 2020. Assembled furniture and provided technical support for hardware and software.",
    },
    {
        "id": "assembler_glentons",
        "title": "Assembler — Glentons AB",
        "text": "Assembler — November 2019 – January 2020. Assembled various sports trophies and awards.",
    },
    {
        "id": "seasonal_agneberg",
        "title": "Seasonal Worker — Agneberg Preschool (Hanaskog)",
        "text": "Seasonal Work — June 2018 – August 2018. Assisted in kitchen duties: cleaning, dishwashing and food preparation.",
    },
    {
        "id": "intern_pizzeria",
        "title": "Intern — Vingården Pizzeria (Osby)",
        "text": "Intern — November 2016 – December 2016. Internship as cook and dishwasher.",
    },

    # Skills sets as searchable items
    {
        "id": "skills_frontend",
        "title": "Frontend Skills",
        "text": "Frontend technologies: React, JavaScript / TypeScript, Tailwind CSS, HTML, CSS, UI/UX. Experience building modern, responsive user interfaces and portfolio components.",
    },
    {
        "id": "skills_backend",
        "title": "Backend & General Programming Skills",
        "text": "Backend / general: Node.js, Python, PHP, MySQL, pgAdmin, SQL, RESTful APIs, Flask, API Gateway, C, C++, Java, MongoDB. Experience implementing backend APIs and database-driven systems.",
    },
    {
        "id": "skills_ml_data",
        "title": "Machine Learning & Data Skills",
        "text": "Machine learning & data: Machine Learning, TensorFlow, Keras, CNN, YOLO, Object Tracking, Dataset Management, Data Augmentation, Model Evaluation, Random Forest, CRISP-DM, Data Mining, Feature Engineering, Adversarial Defense, API Security, LLMs.",
    },
    {
        "id": "skills_devops_tools",
        "title": "DevOps & Tools",
        "text": "DevOps & tools: Docker, Git/GitHub, GCP, MCP Server, phpMyAdmin, Jupyter Notebook, SolidWorks (CSWA). Familiar with containerization and cloud workflows.",
    },
    {
        "id": "skills_other",
        "title": "Other Technical & Business Skills",
        "text": "Other skills: System engineering, Cloud management, ERP system maintenance, Auditing and financial management, Mechanics and vehicle engineering.",
    },

    # Projects split to individual entries + summary
    {
        "id": "project_object_tracking",
        "title": "Project — Object Tracking of Football Players",
        "text": "Object Tracking of Football Players: Developed a tool for small football clubs to track player positions and performance during matches using the YOLO algorithm, dataset management, movement analysis and coach insights. GitHub: https://github.com/Omardll001/project_ObjectTracking",
    },
    {
        "id": "project_rasts",
        "title": "Project — Rasts (Swimrunner Service)",
        "text": "Rasts – Service for swimrunners: Website and app for event registration, timing and performance tracking. Backend in Python & PHP, MySQL database, frontend in HTML/CSS. GitHub: https://github.com/MiranIsmail/EXPproject.github.io",
    },
    {
        "id": "project_rag_ikea",
        "title": "Project — RAG System for IKEA",
        "text": "RAG System for IKEA: Retrieval-Augmented Generation tool with engineering standards knowledge base to help IKEA developers adopt standard workflows and reusable components; contextual recommendations, security compliance and scalable architecture.",
    },
    {
        "id": "project_chatbot",
        "title": "Project — AI Chatbot for Recruiters / Portfolio Chatbot",
        "text": "AI Chatbot for Recruiters: Integrated an interactive chatbot into the portfolio that answers questions about skills, projects and background to provide a conversational and engaging experience.",
    },
    {
        "id": "project_image_recognizer",
        "title": "Project — Cloud-based AI Image Recognizer",
        "text": "Cloud-based AI Image Recognizer: Flask APIs and a convolutional neural network that classifies fashion images; includes defenses against model extraction and adversarial attacks.",
    },
    {
        "id": "project_brain_tumor",
        "title": "Project — Brain Tumor Detector",
        "text": "Brain Tumor Detector: Deep learning pipeline for brain tumor detection using CNNs with MRI preprocessing, training and evaluation and model interpretability.",
    },
    {
        "id": "project_sensor_meliox",
        "title": "Project — Sensor Classification Algorithm (Meliox AB)",
        "text": "Sensor Classification Algorithm for Meliox AB: CRISP-DM applied to sensor data classification (temperature vs humidity) for smart buildings; feature engineering and Random Forest model used in deployment.",
    },
    {
        "id": "projects_summary",
        "title": "Projects — Summary",
        "text": "Summary of projects: Object Tracking (YOLO), Rasts (swimrun service), RAG System for IKEA, AI Chatbot, Cloud-based AI Image Recognizer (Flask/CNN), Brain Tumor Detector, Sensor Classification for Meliox AB.",
    },

    # Qualifications and languages
    {
        "id": "qualifications",
        "title": "Qualifications & Certifications",
        "text": "Driving licence: Class B. Forklift card: Indoor truck. Certified SOLIDWORKS Associate (CSWA). CPR training. Medical & insulin delegation.",
    },
    {
        "id": "languages",
        "title": "Languages",
        "text": "Swedish (Fluent), English (Fluent), Arabic (Fluent), Turkish (Basic).",
    },

    # Administrative / contact notes
    {
        "id": "misc_availability",
        "title": "Availability & Contact Note",
        "text": "For opportunities or project inquiries contact via email omar.dalal.2001@gmail.com. Open to internships, collaborations and junior/mid software and ML roles depending on fit.",
    },
]

# ------------- Config -------------
MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"  # small, fast and good
OUT_DIR = Path("kb_store")
OUT_DIR.mkdir(exist_ok=True)

# per-id default priority overrides
DEFAULT_PRIORITIES = {
    "humly": 1.0,
    "outliar": 0.9,
    "meliox": 0.3,
    "education_bth": 0.35
}

# ------------- Helpers -------------
def make_summary(text: str, max_chars: int = 180) -> str:
    if not text:
        return ""
    # try to extract first sentence
    ss = re.split(r'(?<=[.!?])\s+', text.strip())
    if ss:
        s0 = ss[0].strip()
        if len(s0) <= max_chars:
            return s0
        # otherwise truncate
        return (s0[: max_chars - 3].rstrip() + "...")
    return text[:max_chars]

def make_tags(title: str):
    if not title:
        return []
    # simple tokens: words longer than 2 chars, lowercase, remove punctuation
    words = re.findall(r"[A-Za-z0-9]+", title.lower())
    # remove common stop-ish words
    stop = {"the","and","of","in","—","ai","&"}
    tags = [w for w in words if len(w) > 2 and w not in stop]
    # dedupe preserving order
    seen = set()
    out = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def extract_most_recent_year(text: str):
    years = re.findall(r"(?:19|20)\d{2}", text or "")
    if not years:
        return None
    years_int = [int(y) for y in years]
    return max(years_int)

# ------------- Build enhanced KB -------------
kb_enhanced = []
for item in knowledge_base:
    new_item = dict(item)  # shallow copy
    text = new_item.get("text", "")
    title = new_item.get("title", "")
    new_item["summary"] = make_summary(text)
    new_item["tags"] = make_tags(title)
    new_item["year"] = extract_most_recent_year(text)
    # default priority
    new_item["priority"] = float(DEFAULT_PRIORITIES.get(new_item.get("id"), 0.0))
    kb_enhanced.append(new_item)

# ------------- Load model & embed -------------
print("Loading model:", MODEL_NAME)
model = SentenceTransformer(MODEL_NAME)

texts = [item["text"] for item in kb_enhanced]
title_texts = [ (item.get("title","") + " " + " ".join(item.get("tags",[]))).strip() for item in kb_enhanced ]

print("Embedding", len(texts), "text items...")
text_embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
print("Embedding", len(title_texts), "title items...")
title_embeddings = model.encode(title_texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

# Make sure embeddings are float32 and normalized
def normalize_rows(x):
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms

text_embeddings = normalize_rows(text_embeddings)
title_embeddings = normalize_rows(title_embeddings)

# Save metadata and embeddings
ids = [item["id"] for item in kb_enhanced]
with open(OUT_DIR / "kb_items.json", "w", encoding="utf-8") as f:
    json.dump(kb_enhanced, f, ensure_ascii=False, indent=2)

np.savez_compressed(OUT_DIR / "kb_embeddings.npz",
                    embeddings=text_embeddings,
                    title_embeddings=title_embeddings,
                    ids=np.array(ids, dtype=object))
print("Saved enhanced KB and embeddings to", OUT_DIR)
