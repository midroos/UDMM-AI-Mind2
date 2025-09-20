# 🧠 UDMM v4: Integrated Dynamic Core

[🇬🇧 English](#-english) | [🇦🇪 العربية](#-arabic)

---

## 🇬🇧 English

This project is a Python implementation of the **UDMM v4 architecture**, a self-contained, theoretically-grounded cognitive agent. This version integrates all core components—Dynamic Attractors, a Hierarchical Intent Manager, a Dynamic Memory Module, and an LLM Wrapper—into a single, powerful script.

### 🔬 Key Architectural Features

1.  **Integrated Core (`UDMMCore`):**
    -   A single class that orchestrates all cognitive functions.
    -   **Attractor Dynamics:** Models a 5D attractor state (Ego, Social, etc.) that evolves based on inputs and internal tension.
    -   **Intent Manager:** Manages a three-level hierarchy of goals (Structural, Self, Symbolic).
    -   **Dynamic Memory:** Implements a `MemoryModule` that restructures its "schemas" based on Informational Tension (IT), allowing it to learn and adapt.
    -   **LLM Wrapper:** Includes a simple wrapper for LLM calls (OpenAI or a fallback echo) that can be configured via environment variables.

2.  **Standalone & API-Ready:**
    -   The entire architecture is contained in `udmm_v4_core.py`, which can be run directly for simulation and testing.
    -   A pre-configured FastAPI server (`api/app.py`) exposes the core via `/ask` and `/status` endpoints.

3.  **Interactive UI & Visualization:**
    -   A Streamlit-based UI (`ui_streamlit.py`) provides the primary way to interact with the agent, offering a real-time chat and visualization of the agent's internal state.

### 📂 Project Structure

```
udmm-agent/
├── src/udmm2/
│   ├── api/
│   │   └── app.py          # Optional FastAPI server for the v4 agent
│   └── udmm_v4_core.py     # The complete, integrated UDMM v4 architecture
│
├── ui_streamlit.py         # The main interactive chat UI
├── requirements.txt        # Required libraries
└── README.md               # This file
```

### 🚀 Getting Started

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Interactive Chat UI (Recommended):**
    This launches a web-based chat that visualizes the agent's internal state.
    ```bash
    streamlit run ui_streamlit.py
    ```

### ⚙️ Other Ways to Run

-   **Standalone Simulation:**
    To understand the model's core dynamics, run the core file directly as a module.
    ```bash
    python -m src.udmm2.udmm_v4_core
    ```
    This saves a state visualization to `udmm_v4_state.json`.

-   **API Server (for developers):**
    ```bash
    uvicorn src.udmm2.api.app:app --reload
    ```
    Interact via `http://127.0.0.1:8000/docs`.

### ⚠️ Important Notes
-   **LLM Configuration:** To use a real LLM like OpenAI, set the `LLM_PROVIDER` and `OPENAI_API_KEY` environment variables. Otherwise, it will use a simple echo fallback.
-   **Stateful UI:** The Streamlit UI maintains the agent's state for your session. Closing the tab resets the agent's memory and state.

---

## 🇦🇪 العربية

### 🧠 UDMM v4: النواة الديناميكية المتكاملة

هذا المشروع هو تطبيق بلغة بايثون لمعمارية **UDMM v4**، وهو وكيل إدراكي متكامل ومبني على أسس نظرية. هذه النسخة تدمج جميع المكونات الأساسية - الجاذبات الديناميكية، مدير النوايا الهرمي، وحدة الذاكرة الديناميكية، ومغلف LLM - في سكربت واحد قوي.

### 🔬 الميزات المعمارية الأساسية

1.  **النواة المتكاملة (`UDMMCore`):**
    -   فئة واحدة تنسق جميع الوظائف الإدراكية.
    -   **ديناميكيات الجاذب:** تنمذج حالة جاذب خماسية الأبعاد (الأنا، الاجتماعي، إلخ) تتطور بناءً على المدخلات والتوتر الداخلي.
    -   **مدير النوايا:** يدير تسلسلًا هرميًا للأهداف من ثلاثة مستويات (بنيوي، ذاتي، رمزي).
    -   **الذاكرة الديناميكية:** تطبق `MemoryModule` التي تعيد هيكلة "مخططاتها" بناءً على التوتر المعلوماتي (IT)، مما يسمح لها بالتعلم والتكيف.
    -   **مغلف LLM:** يتضمن مغلفًا بسيطًا لاستدعاءات LLM (OpenAI أو echo كبديل) يمكن تكوينه عبر متغيرات البيئة.

2.  **جاهز للتشغيل المستقل وعبر API:**
    -   المعمارية بأكملها موجودة في `udmm_v4_core.py`، والذي يمكن تشغيله مباشرة للمحاكاة والاختبار.
    -   خادم FastAPI مُعد مسبقًا (`api/app.py`) يعرض النواة عبر نقاط النهاية `/ask` و `/status`.

3.  **واجهة مستخدم تفاعلية وتصور:**
    -   واجهة مستخدم مبنية على Streamlit (`ui_streamlit.py`) توفر الطريقة الأساسية للتفاعل مع الوكيل، وتقدم دردشة وتصورًا فوريًا للحالة الداخلية للوكيل.

### 📂 هيكل المشروع

```
udmm-agent/
├── src/udmm2/
│   ├── api/
│   │   └── app.py          # خادم FastAPI (اختياري)
│   └── udmm_v4_core.py     # معمارية UDMM v4 الكاملة والمتكاملة
│
├── ui_streamlit.py         # الواجهة التفاعلية الرئيسية
├── requirements.txt        # المكتبات المطلوبة
└── README.md               # هذا الملف
```

### 🚀 خطوات التشغيل

1.  **تثبيت المتطلبات:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **تشغيل الواجهة التفاعلية (موصى به):**
    تقوم هذه الطريقة بتشغيل واجهة دردشة على الويب تعرض الحالة الداخلية للوكيل.
    ```bash
    streamlit run ui_streamlit.py
    ```

### ⚙️ طرق تشغيل أخرى

-   **المحاكاة المستقلة:**
    لفهم ديناميكيات النموذج الأساسية، قم بتشغيل الملف الأساسي مباشرة كـ module.
    ```bash
    python -m src.udmm2.udmm_v4_core
    ```
    سيقوم هذا بحفظ تصور الحالة في `udmm_v4_state.json`.

-   **خادم الـ API (للمطورين):**
    ```bash
    uvicorn src.udmm2.api.app:app --reload
    ```
    تفاعل مع الواجهة عبر `http://127.0.0.1:8000/docs`.

### ⚠️ ملاحظات هامة
-   **إعداد LLM:** لاستخدام LLM حقيقي مثل OpenAI، قم بتعيين متغيرات البيئة `LLM_PROVIDER` و `OPENAI_API_KEY`. وإلا، سيتم استخدام وضع الـ echo البسيط.
-   **حالة الواجهة:** تحتفظ واجهة Streamlit بحالة الوكيل طوال مدة جلستك. إغلاق علامة التبويب يعيد تعيين ذاكرة الوكيل وحالته.
