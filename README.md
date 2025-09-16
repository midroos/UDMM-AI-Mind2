# 🧠 UDMM v4: Dynamic Attractor Architecture

[🇬🇧 English](#-english) | [🇦🇪 العربية](#-arabic)

---

## 🇬🇧 English

This project is a Python implementation of the **UDMM v4 architecture**, a significant evolution of the cognitive agent based on the Unified Dynamic Model of Mind. This version moves beyond simple component integration to a fully dynamic, theoretically-grounded system that models attractor states, hierarchical intentions, and informational tension as first-class computational elements.

### 🔬 Key Architectural Features

1.  **Dynamic Attractor System (`AttractorDynamics`):**
    -   Models attractors as vectors in a 5-dimensional space: (Ego, Social, Symbolic, Physical, Cultural).
    -   The attractor state evolves based on external inputs, internal dynamics, inter-attractor coupling, and self-regulating informational tension.

2.  **Hierarchical Intent Manager (`IntentManager`):**
    -   Implements a three-level intent hierarchy: (Structural, Self, Symbolic).
    -   Each level's dynamics are influenced differently by the core attractor state.

3.  **Contextual Prompt Builder (`PromptBuilder`):**
    -   Constructs a rich prompt for the LLM that includes the current attractor state, intent hierarchy, and retrieved memory context.

4.  **Integrated Core (`UDMMCore`):**
    -   The central class that orchestrates the entire process and maintains a detailed history of every time step for analysis and visualization.

5.  **Interactive UI & Visualization:**
    -   A Streamlit-based user interface for easy interaction and chatting.
    -   The UI visualizes the agent's internal attractor state in real-time.

### 📂 Project Structure

```
udmm-agent/
├── src/udmm2/
│   ├── api/
│   │   └── app.py             # Optional FastAPI server
│   └── udmm_v4_core.py        # Contains the entire UDMM v4 architecture
│
├── ui_streamlit.py            # The main interactive chat UI
├── requirements.txt           # Required libraries
└── README.md                  # This file
```

### 🚀 Getting Started

1.  **Install Dependencies:**
    Ensure you have Python 3.9+ and install the required libraries.
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Interactive Chat UI (Recommended Method):**
    This is the primary way to interact with the agent. It launches a web-based chat interface that includes real-time visualization of the agent's internal state.
    ```bash
    streamlit run ui_streamlit.py
    ```

### ⚙️ Other Ways to Run

-   **Standalone Simulation:**
    To understand the model's core dynamics, you can run a non-interactive simulation.
    ```bash
    python src/udmm2/udmm_v4_core.py
    ```
    This will save a visualization to `udmm_system_state.png` and a log to `udmm_history.json`.

-   **API Server (for developers):**
    You can also run the FastAPI server for programmatic interaction.
    ```bash
    uvicorn src.udmm2.api.app:app --reload
    ```

### ⚠️ Important Notes

-   **Prototype Architecture:** This implementation is an architectural prototype. The LLM call in `UDMMCore` is currently a placeholder. To make it a fully functional agent, you must replace the `_get_llm_response` method with a call to a real LLM.
-   **Stateful UI:** The Streamlit UI maintains the agent's state for the duration of your session. If you close the browser tab, the state will be reset.

---

## 🇦🇪 العربية

### 🧠 UDMM v4: معمارية الجاذبات الديناميكية

هذا المشروع هو تطبيق بلغة بايثون لمعمارية **UDMM v4**، والتي تمثل تطورًا جوهريًا للوكيل الإدراكي المبني على النموذج الديناميكي الشامل للعقل. هذه النسخة تتجاوز فكرة دمج المكونات البسيطة إلى نظام ديناميكي متكامل ومبني على أسس نظرية.

### 🔬 الميزات المعمارية الأساسية

1.  **نظام الجاذبات الديناميكية (`AttractorDynamics`):**
    -   ينمذج الجاذبات كمتجهات في فضاء خماسي الأبعاد: (الأنا، الاجتماعي، الرمزي، الجسدي، الثقافي).
    -   تتطور حالة الجاذب بناءً على المدخلات الخارجية، الديناميكيات الداخلية، والتوتر المعلوماتي.

2.  **مدير النوايا الهرمي (`IntentManager`):**
    -   يطبق تسلسلًا هرميًا للنوايا من ثلاثة مستويات: (بنيوي، ذاتي، رمزي).

3.  **باني الأوامر السياقي (`PromptBuilder`):**
    -   يبني أمرًا (prompt) غنيًا لـ LLM يتضمن الحالة الداخلية للوكيل وسياق الذاكرة.

4.  **النواة المتكاملة (`UDMMCore`):**
    -   الفئة المركزية التي تنسق العملية بأكملها وتحتفظ بتاريخ مفصل لكل خطوة.

5.  **واجهة تفاعلية وتصور:**
    -   واجهة مستخدم مبنية على Streamlit للتفاعل والدردشة بسهولة.
    -   تعرض الواجهة تصورًا بيانيًا لحالة الجاذب الداخلية للوكيل في الوقت الفعلي.

### 📂 هيكل المشروع

```
udmm-agent/
├── src/udmm2/
│   ├── api/
│   │   └── app.py             # خادم FastAPI (اختياري)
│   └── udmm_v4_core.py        # يحتوي على معمارية UDMM v4 بأكملها
│
├── ui_streamlit.py            # الواجهة التفاعلية الرئيسية
├── requirements.txt           # المكتبات المطلوبة
└── README.md                  # هذا الملف
```

### 🚀 خطوات التشغيل

1.  **تثبيت المتطلبات:**
    تأكد من أن لديك بايثون 3.9+ وقم بتثبيت المكتبات المطلوبة.
    ```bash
    pip install -r requirements.txt
    ```

2.  **تشغيل الواجهة التفاعلية (الطريقة الموصى بها):**
    هذه هي الطريقة الأساسية للتفاعل مع الوكيل. تقوم بتشغيل واجهة دردشة على الويب تتضمن تصورًا فوريًا للحالة الداخلية للوكيل.
    ```bash
    streamlit run ui_streamlit.py
    ```

### ⚙️ طرق تشغيل أخرى

-   **المحاكاة المستقلة:**
    لفهم ديناميكيات النموذج الأساسية، يمكنك تشغيل محاكاة غير تفاعلية.
    ```bash
    python src/udmm2/udmm_v4_core.py
    ```
    سيقوم هذا بحفظ تصور بياني في `udmm_system_state.png` وسجل في `udmm_history.json`.

-   **خادم الـ API (للمطورين):**
    يمكنك أيضًا تشغيل خادم FastAPI للتفاعل البرمجي.
    ```bash
    uvicorn src.udmm2.api.app:app --reload
    ```

### ⚠️ ملاحظات هامة

-   **نموذج معماري أولي:** استدعاء LLM في `UDMMCore` هو حاليًا عنصر نائب. لجعله وكيلاً وظيفيًا بالكامل، يجب عليك استبدال دالة `_get_llm_response` باستدعاء لـ LLM حقيقي.
-   **حالة الواجهة:** تحتفظ واجهة Streamlit بحالة الوكيل طوال مدة جلستك. إذا أغلقت علامة تبويب المتصفح، فستتم إعادة تعيين الحالة.
