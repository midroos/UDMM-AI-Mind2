# 🧠 UDMM v4: Dynamic Attractor Architecture

[🇬🇧 English](#-english) | [🇦🇪 العربية](#-arabic)

---

## 🇬🇧 English

This project is a Python implementation of the **UDMM v4 architecture**. This version moves beyond simple component integration to a fully dynamic, theoretically-grounded system that models attractor states, hierarchical intentions, and a dynamic memory module as first-class computational elements.

### 🔬 Key Architectural Features

1.  **Dynamic Attractor System (`AttractorDynamics`):**
    -   Models attractors as vectors in a 5-dimensional space.
    -   The attractor state evolves based on external inputs, internal dynamics, and self-regulating informational tension.

2.  **Hierarchical Intent Manager (`IntentManager`):**
    -   Implements a three-level intent hierarchy (Structural, Self, Symbolic).

3.  **Dynamic Memory Module (`MemoryModule`):**
    -   A new core component that replaces static memory.
    -   It calculates Informational Tension (IT) between the agent's prior and posterior beliefs.
    -   Based on an IT threshold, it either performs a "local update" (refining existing knowledge) or proposes a "restructuring" (creating a new memory schema).

4.  **Integrated Core & UI (`UDMMCore`, `ui_streamlit.py`):**
    -   A central `UDMMCore` orchestrates all components.
    -   A Streamlit-based UI allows for easy interaction and visualizes the agent's internal attractor state in real-time.

### 📂 Project Structure

```
udmm-agent/
├── src/udmm2/
│   ├── api/
│   │   └── app.py             # Optional FastAPI server
│   ├── memory_module.py       # The new Dynamic Memory Module
│   └── udmm_v4_core.py        # Contains the entire UDMM v4 architecture
│
├── ui_streamlit.py            # The main interactive chat UI
├── requirements.txt           # Required libraries
└── README.md                  # This file
```

### 🚀 Getting Started

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Interactive Chat UI (Recommended Method):**
    This is the primary way to interact with the agent and see its dynamic state.
    ```bash
    streamlit run ui_streamlit.py
    ```

### ⚙️ Other Ways to Run

-   **Standalone Simulation:**
    To understand the model's core dynamics without a UI, run the core file as a module.
    ```bash
    python -m src.udmm2.udmm_v4_core
    ```
    This will save a visualization to `udmm_v4_dynamic_memory_state.png` and a log to `udmm_v4_history.json`.

-   **API Server (for developers):**
    ```bash
    uvicorn src.udmm2.api.app:app --reload
    ```

### ⚠️ Important Notes
-   **Prototype Architecture:** This is an architectural prototype. The LLM call is a placeholder.
-   **Stateful UI:** The Streamlit UI maintains the agent's state for the duration of your session. Closing the tab resets the state.

---

## 🇦🇪 العربية

### 🧠 UDMM v4: معمارية الجاذبات الديناميكية

هذا المشروع هو تطبيق لمعمارية **UDMM v4**. هذه النسخة تتجاوز فكرة دمج المكونات البسيطة إلى نظام ديناميكي متكامل ومبني على أسس نظرية، ينمذج حالات الجاذبات، النوايا الهرمية، ووحدة الذاكرة الديناميكية كعناصر حسابية من الدرجة الأولى.

### 🔬 الميزات المعمارية الأساسية

1.  **نظام الجاذبات الديناميكية (`AttractorDynamics`):**
    -   ينمذج الجاذبات كمتجهات في فضاء خماسي الأبعاد.
    -   تتطور حالة الجاذب بناءً على المدخلات الخارجية، الديناميكيات الداخلية، والتوتر المعلوماتي.

2.  **مدير النوايا الهرمي (`IntentManager`):**
    -   يطبق تسلسلًا هرميًا للنوايا من ثلاثة مستويات (بنيوي، ذاتي، رمزي).

3.  **وحدة الذاكرة الديناميكية (`MemoryModule`):**
    -   مكون أساسي جديد يحل محل الذاكرة الثابتة.
    -   يقوم بحساب التوتر المعلوماتي (IT) بين معتقدات الوكيل السابقة واللاحقة.
    -   بناءً على عتبة التوتر، فإنه إما يقوم "بتحديث محلي" (صقل المعرفة الحالية) أو يقترح "إعادة هيكلة" (إنشاء مخطط ذاكرة جديد).

4.  **النواة المتكاملة والواجهة (`UDMMCore`, `ui_streamlit.py`):**
    -   `UDMMCore` مركزي ينسق جميع المكونات.
    -   واجهة مستخدم مبنية على Streamlit تسمح بالتفاعل السهل وتصور حالة الجاذب الداخلية للوكيل في الوقت الفعلي.

### 📂 هيكل المشروع

```
udmm-agent/
├── src/udmm2/
│   ├── api/
│   │   └── app.py             # خادم FastAPI (اختياري)
│   ├── memory_module.py       # وحدة الذاكرة الديناميكية الجديدة
│   └── udmm_v4_core.py        # يحتوي على معمارية UDMM v4 بأكملها
│
├── ui_streamlit.py            # الواجهة التفاعلية الرئيسية
├── requirements.txt           # المكتبات المطلوبة
└── README.md                  # هذا الملف
```

### 🚀 خطوات التشغيل

1.  **تثبيت المتطلبات:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **تشغيل الواجهة التفاعلية (الطريقة الموصى بها):**
    هذه هي الطريقة الأساسية للتفاعل مع الوكيل ورؤية حالته الديناميكية.
    ```bash
    streamlit run ui_streamlit.py
    ```

### ⚙️ طرق تشغيل أخرى

-   **المحاكاة المستقلة:**
    لفهم ديناميكيات النموذج الأساسية بدون واجهة، قم بتشغيل الملف الأساسي كـ module.
    ```bash
    python -m src.udmm2.udmm_v4_core
    ```
    سيقوم هذا بحفظ تصور بياني في `udmm_v4_dynamic_memory_state.png` وسجل في `udmm_v4_history.json`.

-   **خادم الـ API (للمطورين):**
    ```bash
    uvicorn src.udmm2.api.app:app --reload
    ```

### ⚠️ ملاحظات هامة
-   **نموذج معماري أولي:** استدعاء LLM هو حاليًا عنصر نائب.
-   **حالة الواجهة:** تحتفظ واجهة Streamlit بحالة الوكيل طوال مدة الجلسة. إغلاق علامة التبويب يعيد تعيين الحالة.
