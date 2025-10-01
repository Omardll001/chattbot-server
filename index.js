// server/index.js
/**
 * Small Express server that:
 *  - Provides /api/query that: finds nearest context via cosine similarity,
 *    then calls a local Ollama model (gemma3:4b) to produce an answer using the
 *    retrieved context snippets.
 *
 * Notes:
 *  - This preserves your original logic (vector store, cosine scoring, context
 *    building, sources) but replaces spawn(...) with execFile(...) for one-shot
 *    calls to the Ollama CLI. It also truncates very large prompts, adds a
 *    timeout and better error logging.
 *
 * Requirements:
 *   npm init -y
 *   npm i express cors body-parser
 *
 * Ollama:
 *   - Make sure ollama.exe path below matches your machine.
 *   - Tested with: C:\Users\omard\AppData\Local\Programs\Ollama\ollama.exe
 */

import express from "express";
import cors from "cors";
import bodyParser from "body-parser";
import { execFile } from "child_process";

const app = express();
app.use(cors());
app.use(bodyParser.json({ limit: "1mb" }));

// --- Knowledge base (canonical data) ---
const knowledgeBaseItems = [
  {
    id: "profile",
    title: "Profile summary",
    text:
      "Master of Science in Engineering in AI and Machine Learning at BTH (4th year). Strong programming skills in Python, C and Java. Experienced in machine learning, security in AI systems, genetic algorithms and object tracking (YOLO). Familiar with cloud technologies (Docker, GCP) and GitHub. Practical experience maintaining ERP systems and automotive/mechanical work. Interested in ML systems, software engineering and secure AI.",
  },
  {
    id: "contact",
    title: "Contact info",
    text:
      "Contact information: Email: omar.dalal.2001@gmail.com. Phone: +46 76-9481560. Address: Kungsmarksvägen 7, 37144 Karlskrona, Sweden.",
  },

  // Education
  {
    id: "education_bth",
    title: "Education — Blekinge Institute of Technology (BTH)",
    text:
      "M.Sc. in Engineering — AI & Machine Learning at Blekinge Institute of Technology (BTH). Study period: 2021 – expected 2026.",
  },
  {
    id: "education_highschool",
    title: "Education — Technology Programme (High School)",
    text: "Completed the Technology Programme in upper secondary school (Technology programme).",
  },

  // Work experience — separate items for each role
  {
    id: "humly",
    title: "Humly Solutions AB — QA & Technical Support (Workplace Solutions)",
    text:
      "Humly Solutions AB — December 2024 – June 2026. Responsible for quality assurance (QA) of new systems and firmware, performing thorough tests of software and hardware before production. Acted as a technical key resource in customer support handling troubleshooting, system analysis and guidance on system functionality and API integrations. Contributed across the stack from cloud-based backend services to physical devices.",
  },
  {
    id: "outliar",
    title: "Outliar — Python Developer (AI company, remote)",
    text:
      "Outliar — February 2025 – June 2025. Developed advanced Python programs to improve AI models' abilities to write, understand and analyze code. Projects included code generation, semantic understanding, refactoring, and code analysis pipelines to improve model precision and usability.",
  },
  {
    id: "meliox",
    title: "Meliox AB — AI Project Collaboration / Internship",
    text:
      "Meliox AB — January 2025 – June 2025. Led an AI project for sensor classification within smart building systems. Tasks included data collection, preprocessing, model selection, implementation and testing. The solution helped automate sensor identification and integration into company systems for energy management and automation.",
  },
  {
    id: "self_grocery",
    title: "Self-employed & System Developer — Grocery store (Östra Göinge)",
    text:
      "Self-employed & System Developer — July 2021 – June 2025. Responsible for accounting, auditing, financial maintenance and declarations. Built and maintained a custom system for the grocery store to track inventory, incomes, expenses and other shop-specific functionality, and handled ongoing system maintenance.",
  },
  {
    id: "automotive_mechanic",
    title: "Automotive Mechanic — Karlskrona",
    text:
      "Automotive Mechanic — June 2023 – June 2024. Diagnosed vehicles, identified and fixed problems, ordered and replaced parts, and ensured vehicles operated correctly through repairs and maintenance.",
  },
  {
    id: "assistant_nurse",
    title: "Assistant Nurse (Night Patrol) — Karlskrona Municipality",
    text:
      "Assistant Nurse, Night Patrol — November 2022 – May 2023. Provided personal care, support and assistance to clients in their homes as part of home care services.",
  },
  {
    id: "patient_guard",
    title: "Patient Guard — Hässleholm Hospital",
    text:
      "Patient Guard — October 2020 – August 2021. Temporary role supporting patients with special needs in daily activities.",
  },
  {
    id: "service_staff",
    title: "Service Staff — Hässleholm Hospital",
    text:
      "Service Staff — June 2020 – August 2020. Duties included transport responsibility in June and cleaning/facility maintenance in July and August.",
  },
  {
    id: "support_hemfixare",
    title: "Support Technician & Furniture Assembler — Hemfixare AB",
    text:
      "Support Technician & Furniture Assembler — December 2019 – August 2020. Assembled furniture and provided technical support for hardware and software.",
  },
  {
    id: "assembler_glentons",
    title: "Assembler — Glentons AB",
    text: "Assembler — November 2019 – January 2020. Assembled various sports trophies and awards.",
  },
  {
    id: "seasonal_agneberg",
    title: "Seasonal Worker — Agneberg Preschool (Hanaskog)",
    text: "Seasonal Work — June 2018 – August 2018. Assisted in kitchen duties: cleaning, dishwashing and food preparation.",
  },
  {
    id: "intern_pizzeria",
    title: "Intern — Vingården Pizzeria (Osby)",
    text: "Intern — November 2016 – December 2016. Internship as cook and dishwasher.",
  },

  // Skills sets as searchable items
  {
    id: "skills_frontend",
    title: "Frontend Skills",
    text:
      "Frontend technologies: React, JavaScript / TypeScript, Tailwind CSS, HTML, CSS, UI/UX. Experience building modern, responsive user interfaces and portfolio components.",
  },
  {
    id: "skills_backend",
    title: "Backend & General Programming Skills",
    text:
      "Backend / general: Node.js, Python, PHP, MySQL, pgAdmin, SQL, RESTful APIs, Flask, API Gateway, C, C++, Java, MongoDB. Experience implementing backend APIs and database-driven systems.",
  },
  {
    id: "skills_ml_data",
    title: "Machine Learning & Data Skills",
    text:
      "Machine learning & data: Machine Learning, TensorFlow, Keras, CNN, YOLO, Object Tracking, Dataset Management, Data Augmentation, Model Evaluation, Random Forest, CRISP-DM, Data Mining, Feature Engineering, Adversarial Defense, API Security, LLMs.",
  },
  {
    id: "skills_devops_tools",
    title: "DevOps & Tools",
    text:
      "DevOps & tools: Docker, Git/GitHub, GCP, MCP Server, phpMyAdmin, Jupyter Notebook, SolidWorks (CSWA). Familiar with containerization and cloud workflows.",
  },
  {
    id: "skills_other",
    title: "Other Technical & Business Skills",
    text:
      "Other skills: System engineering, Cloud management, ERP system maintenance, Auditing and financial management, Mechanics and vehicle engineering.",
  },

  // Projects split to individual entries + summary
  {
    id: "project_object_tracking",
    title: "Project — Object Tracking of Football Players",
    text:
      "Object Tracking of Football Players: Developed a tool for small football clubs to track player positions and performance during matches using the YOLO algorithm, dataset management, movement analysis and coach insights. GitHub: https://github.com/Omardll001/project_ObjectTracking",
  },
  {
    id: "project_rasts",
    title: "Project — Rasts (Swimrunner Service)",
    text:
      "Rasts – Service for swimrunners: Website and app for event registration, timing and performance tracking. Backend in Python & PHP, MySQL database, frontend in HTML/CSS. GitHub: https://github.com/MiranIsmail/EXPproject.github.io",
  },
  {
    id: "project_rag_ikea",
    title: "Project — RAG System for IKEA",
    text:
      "RAG System for IKEA: Retrieval-Augmented Generation tool with engineering standards knowledge base to help IKEA developers adopt standard workflows and reusable components; contextual recommendations, security compliance and scalable architecture.",
  },
  {
    id: "project_chatbot",
    title: "Project — AI Chatbot for Recruiters / Portfolio Chatbot",
    text:
      "AI Chatbot for Recruiters: Integrated an interactive chatbot into the portfolio that answers questions about skills, projects and background to provide a conversational and engaging experience.",
  },
  {
    id: "project_image_recognizer",
    title: "Project — Cloud-based AI Image Recognizer",
    text:
      "Cloud-based AI Image Recognizer: Flask APIs and a convolutional neural network that classifies fashion images; includes defenses against model extraction and adversarial attacks.",
  },
  {
    id: "project_brain_tumor",
    title: "Project — Brain Tumor Detector",
    text:
      "Brain Tumor Detector: Deep learning pipeline for brain tumor detection using CNNs with MRI preprocessing, training and evaluation and model interpretability.",
  },
  {
    id: "project_sensor_meliox",
    title: "Project — Sensor Classification Algorithm (Meliox AB)",
    text:
      "Sensor Classification Algorithm for Meliox AB: CRISP-DM applied to sensor data classification (temperature vs humidity) for smart buildings; feature engineering and Random Forest model used in deployment.",
  },
  {
    id: "projects_summary",
    title: "Projects — Summary",
    text:
      "Summary of projects: Object Tracking (YOLO), Rasts (swimrun service), RAG System for IKEA, AI Chatbot, Cloud-based AI Image Recognizer (Flask/CNN), Brain Tumor Detector, Sensor Classification for Meliox AB.",
  },

  // Qualifications and languages
  {
    id: "qualifications",
    title: "Qualifications & Certifications",
    text:
      "Driving licence: Class B. Forklift card: Indoor truck. Certified SOLIDWORKS Associate (CSWA). CPR training. Medical & insulin delegation.",
  },
  {
    id: "languages",
    title: "Languages",
    text: "Swedish (Fluent), English (Fluent), Arabic (Fluent), Turkish (Basic).",
  },

  // Add small administrative / contact notes (useful to show when user requests)
  {
    id: "misc_availability",
    title: "Availability & Contact Note",
    text:
      "For opportunities or project inquiries contact via email omar.dalal.2001@gmail.com. Open to internships, collaborations and junior/mid software and ML roles depending on fit.",
  },
];

// --- Vectorize text locally ---
function textToVector(text) {
  const words = text.toLowerCase().replace(/[^\w\s]/g,'').split(/\s+/);
  const freq = {};
  words.forEach(w => freq[w] = (freq[w]||0)+1);
  return freq;
}

// --- Cosine similarity for TF vectors ---
function cosineSimTF(a, b) {
  const allKeys = new Set([...Object.keys(a), ...Object.keys(b)]);
  let dot = 0, magA = 0, magB = 0;
  allKeys.forEach(k => {
    const va = a[k] || 0;
    const vb = b[k] || 0;
    dot += va*vb;
    magA += va*va;
    magB += vb*vb;
  });
  return dot / (Math.sqrt(magA) * Math.sqrt(magB) + 1e-12);
}

// --- Precompute KB vectors ---
let vectorStore = knowledgeBaseItems.map(item => ({
  ...item,
  vector: textToVector(item.text)
}));


// --- Health check ---
app.get("/health",(req,res)=>res.json({ok:true}));

// --- Query endpoint ---
app.post("/api/query", async (req, res) => {
  const { question, top_k = 3 } = req.body;
  if (!question || !question.trim()) return res.status(400).json({ error: "Missing question" });

  // --- Retrieve top-k relevant KB items using TF vectors ---
    const qVector = textToVector(question);
    const scored = vectorStore.map(item => ({ ...item, score: cosineSimTF(item.vector, qVector) }));
    scored.sort((a,b) => b.score - a.score);
    const top = scored.slice(0, top_k);


  const contextParts = top.map((t,i)=>`Source ${i+1} (${t.title}): ${t.text}`).join("\n\n---\n\n");

  // --- Protect prompt length (truncate if very large) ---
  const maxContextChars = 4000; // tune if needed
  const safeContext = contextParts.length > maxContextChars
    ? contextParts.slice(0, maxContextChars) + "\n\n...context truncated..."
    : contextParts;

  const fullPrompt = `
You are an assistant that answers questions about Omar Dalal using only the provided context.
${safeContext}
User question: "${question}"
Answer concisely and cite sources at the end.
`.trim();

  try {
    // Path to your local ollama executable
    const ollamaPath = "C:\\Users\\omard\\AppData\\Local\\Programs\\Ollama\\ollama.exe";

    // Build arguments: run gemma3:4b (no prompt arg)
    const args = ["run", "gemma3:4b"];

    // Debug logging
    console.log("Calling Ollama with prompt length:", fullPrompt.length);
    console.log("Prompt preview:", fullPrompt.slice(0, 300).replace(/\n/g, "\\n"));

    const timeoutMs = 120 * 1000; // 120s
    const maxBuffer = 20 * 1024 * 1024; // 20MB

    const child = execFile(
    ollamaPath,
    args,
    { timeout: timeoutMs, maxBuffer },
    (err, stdout, stderr) => {
        if (err) {
        console.error("Ollama execFile error:", err);
        if (stderr && stderr.toString().trim()) console.error("Ollama stderr:", stderr.toString());

        if (err.killed && err.signal === "SIGTERM") {
            return res.status(500).json({ error: "Ollama timed out or was killed (SIGTERM)" });
        }
        return res.status(500).json({ error: `Ollama error: ${err.message}` });
        }

        if (stderr && stderr.toString().trim()) {
        console.log("Ollama stderr (non-empty):", stderr.toString());
        }

        const answer = (stdout || "").toString().trim();
        if (!answer) {
        console.error("Ollama returned empty stdout. stderr:", stderr && stderr.toString());
        return res.status(500).json({ error: "Empty response from Ollama" });
        }

        return res.json({
        answer,
        sources: top.map((t, i) => ({ id: t.id, title: t.title, sourceNumber: i + 1 }))
        });
    }
    );

    // Write the prompt to stdin and close it
    child.stdin.write(fullPrompt);
    child.stdin.end();

    // Log the spawned child (pid) for debugging and watch if it exceeds timeout
    if (child && child.pid) {
      console.log("Spawned Ollama child, pid:", child.pid);
      
      // optional: attach listeners if you want streaming logs server-side
      child.stdout?.on("data", d => {
        // Not returning streaming to client — just log
        // console.log("ollama stdout chunk:", d.toString());
      });
      child.stderr?.on("data", d => {
        // console.error("ollama stderr chunk:", d.toString());
      });
      child.on("close", (code, signal) => {
        console.log("Ollama process closed. code=", code, "signal=", signal);
      });
    }
  } catch (err) {
    console.error("Query error:", err);
    return res.status(500).json({ error: err.message || "server error" });
  }
});

// --- Start server ---
const port = process.env.PORT || 5174;
app.listen(port, ()=>console.log(`Server listening on http://localhost:${port}`));
