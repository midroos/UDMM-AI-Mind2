# 🧠 UDMM Cognitive Agent

[🇬🇧 English](#-english) | [🇦🇪 العربية](#-arabic)

---

## 🇬🇧 English

This project aims to build a Cognitive Agent based on the Unified Dynamic Model of Mind (UDMM). The agent integrates an LLM (like OpenAI or Gemini), a RAG memory (based on FAISS), an Intent Manager, and a dynamic simulation engine into a single framework that reflects UDMM principles.

### 📂 Project Structure

```
udmm-agent/
├── src/udmm2/
│   ├── api/
│   │   └── app.py             # FastAPI server exposing the API endpoints
│   ├── intent/
│   │   └── hierarchical.py    # Hierarchical Intent Manager
│   ├── llm/                   # LLM Providers (OpenAI, Gemini, LlamaCPP, Echo)
│   ├── memory/
│   │   └── faiss_rag.py       # Semantic RAG memory using FAISS
│   ├── config.py              # General settings (API keys, model options...)
│   ├── simulation.py          # UDMM dynamics simulation engine
│   └── udmm_core.py           # Agent's core - integrates all components
│
├── scripts/
│   ├── index_data.py          # Script to index the knowledge base
│   └── verify_memory.py       # Script to verify the memory
│
├── data/
│   ├── knowledge_base.json    # The initial knowledge base (JSON)
│   └── episodic.json          # Episodic Memory
│
├── ui_streamlit.py            # Simple UI for interaction (Streamlit)
├── requirements.txt           # Required libraries
└── README.md                  # This file
```

### ⚙️ Core Features

-   **🔧 Multi-LLM Provider Support:**
    Supports OpenAI, Gemini, LlamaCPP (local) + an Echo model for testing.

-   **🧠 FAISS RAG Memory:**
    For storing and recalling facts/knowledge taught to the agent, with pre-indexing capabilities.

-   **🎯 Hierarchical Intent Manager:**
    A system to determine the type of question (informational, explanatory, social, existential).

-   **⚡ Dynamic Simulation:**
    Includes a simple simulation engine that models UDMM concepts like InfoTension and Perceptual Time, influencing the agent's behavior.

-   **🔗 UDMM Core Integration:**
    Combines input processing, LLM interpretation, memory interaction, and output generation.

-   **🌐 API Interface:**
    -   `/ask`: To ask questions.
    -   `/teach`: To add new knowledge to the agent's memory.
    -   `/status`: To view the current state of the agent and simulation.
    -   `/simulate`: To directly trigger a simulation step.

-   **🖥️ Streamlit UI:**
    For a simple and visual chat experience.

### 🚀 Getting Started

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Set API Keys (Optional):**
    Set the API keys you want to use as environment variables, or modify `src/udmm2/config.py`.
    ```bash
    export GEMINI_API_KEY="YOUR_KEY"
    export OPENAI_API_KEY="YOUR_KEY"
    ```

3.  **Index the Knowledge Base (Important):**
    To give the agent an initial memory, run the indexing script.
    ```bash
    python scripts/index_data.py
    ```
    This command will create the `data/faiss.index` and `data/faiss_meta.json` files.

4.  **Run the API Server:**
    ```bash
    uvicorn src.udmm2.api.app:app --reload
    ```
    You can access the interactive API docs at: `http://127.0.0.1:8000/docs`

5.  **Or run the Streamlit UI:**
    ```bash
    streamlit run ui_streamlit.py
    ```

### 🧪 Experimental UDMM

This version is a prototype that implements some UDMM principles:

-   **🧲 Attractors:**
    Simulated through the intent system + weighted values (knowledge, social...).

-   **⚡ Dynamic State:**
    Demonstrated in the cycle: Input → Simulation → LLM → Memory → Interpretation. The simulation state (tension, time) affects the LLM prompt.

-   **🔄 Dynamic Learning:**
    The system can be "taught" new information (`/teach`), which it will store in FAISS for later retrieval.

-   **🌀 Semantic Interaction:**
    The model attempts to connect stored knowledge with the current question (Retrieval Augmented).

### 📊 How to Test?

1.  **Ask the system a question from its memory:**
    ```bash
    curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d '{"question": "What is the capital of France?"}'
    ```

2.  **Teach it something new:**
    ```bash
    curl -X POST http://127.0.0.1:8000/teach -H "Content-Type: application/json" -d '{"question": "What is UDMM?", "answer": "A unified dynamic model of the mind."}'
    ```

3.  **Ask it again:**
    ```bash
    curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d '{"question": "What is UDMM?"}'
    ```
    It will retrieve what it learned.

### ⚠️ Current Limitations

-   Relies on an external LLM (OpenAI/Gemini) and does not yet generate its own knowledge.
-   The attractors, intents, and simulation are still rudimentary.
-   No advanced management of long-term context or complex temporal simulation yet.
-   The Streamlit UI is a basic prototype for experimentation.

---

## 🇦🇪 العربية

### 🧠 وكيل UDMM الإدراكي

هذا المشروع يهدف إلى بناء وكيل إدراكي (Cognitive Agent) مستند إلى النموذج الديناميكي الشامل للعقل (UDMM).
الوكيل يدمج بين LLM (مثل OpenAI أو Gemini) وذاكرة RAG (قائمة على FAISS) ونظام لإدارة النوايا (Intent Manager) ومحرك محاكاة ديناميكي في إطار واحد يعكس مبادئ UDMM.

### 📂 هيكل المشروع

```
udmm-agent/
├── src/udmm2/
│   ├── api/
│   │   └── app.py             # خادم FastAPI - يعرّض واجهات API
│   ├── intent/
│   │   └── hierarchical.py    # مدير النوايا (Hierarchical Intent Manager)
│   ├── llm/                   # مزودو LLM (OpenAI, Gemini, LlamaCPP, Echo)
│   ├── memory/
│   │   └── faiss_rag.py       # الذاكرة الدلالية RAG باستخدام FAISS
│   ├── config.py              # إعدادات عامة (مفاتيح API، خيارات النموذج...)
│   ├── simulation.py          # محرك محاكاة ديناميكيات UDMM
│   └── udmm_core.py           # قلب الوكيل - التكامل بين المكونات
│
├── scripts/
│   ├── index_data.py          # سكربت لفهرسة قاعدة المعرفة
│   └── verify_memory.py       # سكربت للتحقق من الذاكرة
│
├── data/
│   ├── knowledge_base.json    # قاعدة المعرفة الأولية (JSON)
│   └── episodic.json          # الذاكرة العرضية (Episodic Memory)
│
├── ui_streamlit.py            # واجهة بسيطة للتفاعل (Streamlit UI)
├── requirements.txt           # المكتبات المطلوبة
└── README.md                  # هذا الملف
```

### ⚙️ المزايا الأساسية

- **🔧 تعدد مزودي LLM:**
  دعم OpenAI و Gemini و LlamaCPP (محلي) + نموذج Echo للتجارب.

- **🧠 ذاكرة FAISS RAG:**
  لتخزين وتذكر الحقائق/المعارف التي يتم تعليمها للوكيل، مع إمكانية الفهرسة المسبقة.

- **🎯 مدير النوايا (Intents):**
  نظام هرمي لتحديد نوع السؤال (معلومة/تفسير/اجتماعي/وجودي).

- **⚡ محاكاة ديناميكية (Dynamic Simulation):**
  يتضمن محرك محاكاة بسيط ينمذج مفاهيم UDMM مثل التوتر المعلوماتي (InfoTension) والزمن الإدراكي، مما يؤثر على سلوك الوكيل.

- **🔗 تكامل UDMM Core:**
  الجمع بين الإدخال (Input)، التفسير عبر LLM، التفاعل مع الذاكرة، وتوليد المخرجات.

- **🌐 واجهة API:**
  - `/ask`: لطرح الأسئلة.
  - `/teach`: لإضافة معرفة جديدة إلى ذاكرة الوكيل.
  - `/status`: لعرض الحالة الحالية للوكيل والمحاكاة.
  - `/simulate`: لتشغيل خطوة محاكاة بشكل مباشر.

- **🖥️ واجهة Streamlit:**
  لتجربة المحادثة بشكل بصري وبسيط.

### 🚀 التشغيل

1.  **تثبيت المتطلبات:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **إعداد المفاتيح (اختياري):**
    قم بتعيين مفاتيح API التي تريد استخدامها كمتغيرات بيئة، أو قم بتعديل `src/udmm2/config.py`.
    ```bash
    export GEMINI_API_KEY="YOUR_KEY"
    export OPENAI_API_KEY="YOUR_KEY"
    ```

3.  **فهرسة قاعدة المعرفة (مهم):**
    لإعطاء الوكيل ذاكرة أولية، قم بتشغيل سكربت الفهرسة.
    ```bash
    python scripts/index_data.py
    ```
    سيقوم هذا الأمر بإنشاء ملفات `data/faiss.index` و `data/faiss_meta.json`.

4.  **تشغيل خادم API:**
    ```bash
    uvicorn src.udmm2.api.app:app --reload
    ```
    سيمكنك الوصول إلى الواجهة التفاعلية للـ API عبر: `http://127.0.0.1:8000/docs`

5.  **أو تشغيل واجهة Streamlit:**
    ```bash
    streamlit run ui_streamlit.py
    ```

### 🧪 التجريبية (Experimental UDMM)

هذه النسخة تعتبر بروتوتايب (Prototype) يطبق بعض مبادئ UDMM:

- **🧲 الجاذبات (Attractors):**
  تُحاكى من خلال نظام النوايا + قيم مرجحة (knowledge, social...).

- **⚡ الحالة الديناميكية (Dynamic State):**
  يظهر عند تفعيل الدورة بين الإدخال → المحاكاة → LLM → الذاكرة → التفسير. حالة المحاكاة (التوتر، الزمن) تؤثر على prompt الـ LLM.

- **🔄 التعلم الديناميكي:**
  يمكن "تعليم" النظام بمعلومة جديدة (`/teach`)، وسيخزنها في FAISS ليسترجعها لاحقًا.

- **🌀 التفاعل الدلالي:**
  النموذج يحاول ربط المعرفة المخزنة مع السؤال الحالي (Retrieval Augmented).

### 📊 كيف تختبر؟

1.  **اسأل النظام سؤالاً موجوداً في الذاكرة:**
    ```bash
    curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d '{"question": "ما هي عاصمة فرنسا؟"}'
    ```

2.  **علمه شيئًا جديدًا:**
    ```bash
    curl -X POST http://127.0.0.1:8000/teach -H "Content-Type: application/json" -d '{"question": "ما هو UDMM؟", "answer": "نموذج ديناميكي شامل للعقل."}'
    ```

3.  **أعد سؤاله:**
    ```bash
    curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d '{"question": "ما هو UDMM؟"}'
    ```
    سيسترجع ما تعلمه منك.

### ⚠️ حدود النسخة الحالية

-   تعتمد على LLM خارجي (OpenAI/Gemini) ولا تولد معرفة ذاتية بعد.
-   الجاذبات والنوايا والمحاكاة ما زالت أولية (rudimentary).
-   لا توجد حتى الآن إدارة متقدمة للسياق الطويل أو المحاكاة الزمنية المعقدة.
-   واجهة Streamlit مجرد نسخة بدائية للتجريب.
