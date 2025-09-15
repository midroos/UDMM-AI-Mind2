# 🧠 UDMM v4: Dynamic Attractor Architecture

[🇬🇧 English](#-english) | [🇦🇪 العربية](#-arabic)

---

## 🇬🇧 English

This project is a Python implementation of the **UDMM v4 architecture**, a significant evolution of the cognitive agent based on the Unified Dynamic Model of Mind. This version moves beyond simple component integration to a fully dynamic, theoretically-grounded system that models attractor states, hierarchical intentions, and informational tension as first-class computational elements.

### 🔬 Key Architectural Features

1.  **Dynamic Attractor System (`AttractorDynamics`):**
    -   Models attractors as vectors in a 5-dimensional space: (Ego, Social, Symbolic, Physical, Cultural).
    -   The attractor state evolves based on external inputs, internal dynamics, inter-attractor coupling, and self-regulating informational tension.
    -   Uses the Dirichlet distribution to ensure normalized, coherent states.

2.  **Hierarchical Intent Manager (`IntentManager`):**
    -   Implements a three-level intent hierarchy: (Structural, Self, Symbolic).
    -   Each level's dynamics are influenced differently by the core attractor state, allowing for complex, multi-layered goal-setting.

3.  **Contextual Prompt Builder (`PromptBuilder`):**
    -   Constructs a rich prompt for the LLM that includes the current attractor state, the full intent hierarchy, and retrieved memory context.
    -   Features a basic semantic memory system using `sentence-transformers` for encoding and retrieval.

4.  **Integrated Core (`UDMMCore`):**
    -   The central class that orchestrates the entire process: calculating tension, updating attractors and intents, retrieving memory, building the prompt, and calling the LLM (currently a placeholder).
    -   Maintains a detailed history of every time step for analysis and visualization.

5.  **Built-in Visualization:**
    -   The system can generate plots for the evolution of attractors, intents, and informational tension over time, providing a powerful tool for diagnostics and understanding the agent's internal state.

### 📂 Project Structure

```
udmm-agent/
├── src/udmm2/
│   ├── api/
│   │   └── app.py             # FastAPI server for the v4 agent
│   └── udmm_v4_core.py        # Contains the entire UDMM v4 architecture
│
├── requirements.txt           # Required libraries
└── README.md                  # This file
```

### 🚀 Getting Started

1.  **Install Dependencies:**
    Ensure you have Python 3.9+ and install the required libraries.
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Standalone Simulation:**
    The `udmm_v4_core.py` file can be run directly to simulate a conversation and visualize the results. This is the best way to understand the model's dynamics.
    ```bash
    python src/udmm2/udmm_v4_core.py
    ```
    This will:
    -   Run a predefined multi-turn conversation.
    -   Print the agent's placeholder response at each step.
    -   Save a visualization of the system's state to `udmm_system_state.png`.
    -   Save the detailed conversation history to `udmm_history.json`.

3.  **Run the API Server (Optional):**
    You can also interact with the UDMM v4 core via a simple FastAPI server.
    ```bash
    uvicorn src.udmm2.api.app:app --reload
    ```
    You can then send requests to the `/ask` endpoint:
    ```bash
    curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d '{"question": "Tell me about attractors."}'
    ```

### ⚠️ Important Notes

-   **Prototype Architecture:** This implementation is an architectural prototype. The LLM call in `UDMMCore` is currently a placeholder. To make it a fully functional agent, you must replace the `_get_llm_response` method with a call to a real LLM (e.g., OpenAI, Gemini).
-   **Simplified Dynamics:** The dynamics and coupling matrices in `AttractorDynamics` are initialized with random values for demonstration. For a more advanced implementation, these should be based on a studied model of how these concepts influence each other.
-   **Basic Memory:** The memory system is a simple list of embeddings. It can be upgraded to use the more robust FAISS indexing from previous versions for better performance.

---

## 🇦🇪 العربية

### 🧠 UDMM v4: معمارية الجاذبات الديناميكية

هذا المشروع هو تطبيق بلغة بايثون لمعمارية **UDMM v4**، والتي تمثل تطورًا جوهريًا للوكيل الإدراكي المبني على النموذج الديناميكي الشامل للعقل. هذه النسخة تتجاوز فكرة دمج المكونات البسيطة إلى نظام ديناميكي متكامل ومبني على أسس نظرية، حيث ينمذج حالات الجاذبات، النوايا الهرمية، والتوتر المعلوماتي كعناصر حسابية من الدرجة الأولى.

### 🔬 الميزات المعمارية الأساسية

1.  **نظام الجاذبات الديناميكية (`AttractorDynamics`):**
    -   ينمذج الجاذبات كمتجهات في فضاء خماسي الأبعاد: (الأنا، الاجتماعي، الرمزي، الجسدي، الثقافي).
    -   تتطور حالة الجاذب بناءً على المدخلات الخارجية، الديناميكيات الداخلية، التزاوج بين الجاذبات، والتوتر المعلوماتي ذاتي التنظيم.
    -   يستخدم توزيع ديريخليت لضمان حالات طبيعية ومتماسكة.

2.  **مدير النوايا الهرمي (`IntentManager`):**
    -   يطبق تسلسلًا هرميًا للنوايا من ثلاثة مستويات: (بنيوي، ذاتي، رمزي).
    -   تتأثر ديناميكيات كل مستوى بشكل مختلف بحالة الجاذب الأساسية، مما يسمح بتحديد أهداف معقدة ومتعددة الطبقات.

3.  **باني الأوامر السياقي (`PromptBuilder`):**
    -   يبني أمرًا (prompt) غنيًا لـ LLM يتضمن حالة الجاذب الحالية، والتسلسل الهرمي الكامل للنوايا، وسياق الذاكرة المسترجع.
    -   يتميز بنظام ذاكرة دلالي أساسي يستخدم `sentence-transformers` للترميز والاسترجاع.

4.  **النواة المتكاملة (`UDMMCore`):**
    -   الفئة المركزية التي تنسق العملية بأكملها: حساب التوتر، تحديث الجاذبات والنوايا، استرجاع الذاكرة، بناء الأمر، واستدعاء LLM (حاليًا كعنصر نائب).
    -   تحتفظ بتاريخ مفصل لكل خطوة زمنية للتحليل والتصور.

5.  **التصور المدمج (Built-in Visualization):**
    -   يمكن للنظام إنشاء رسوم بيانية لتطور الجاذبات والنوايا والتوتر المعلوماتي بمرور الوقت، مما يوفر أداة قوية للتشخيص وفهم الحالة الداخلية للوكيل.

### 📂 هيكل المشروع

```
udmm-agent/
├── src/udmm2/
│   ├── api/
│   │   └── app.py             # خادم FastAPI لوكيل v4
│   └── udmm_v4_core.py        # يحتوي على معمارية UDMM v4 بأكملها
│
├── requirements.txt           # المكتبات المطلوبة
└── README.md                  # هذا الملف
```

### 🚀 خطوات التشغيل

1.  **تثبيت المتطلبات:**
    تأكد من أن لديك بايثون 3.9+ وقم بتثبيت المكتبات المطلوبة.
    ```bash
    pip install -r requirements.txt
    ```

2.  **تشغيل المحاكاة المستقلة:**
    يمكن تشغيل ملف `udmm_v4_core.py` مباشرة لمحاكاة محادثة وتصور النتائج. هذه هي أفضل طريقة لفهم ديناميكيات النموذج.
    ```bash
    python src/udmm2/udmm_v4_core.py
    ```
    سيقوم هذا الأمر بما يلي:
    -   تشغيل محادثة محددة مسبقًا متعددة الأدوار.
    -   طباعة استجابة الوكيل النائبة في كل خطوة.
    -   حفظ تصور لحالة النظام في ملف `udmm_system_state.png`.
    -   حفظ سجل المحادثة المفصل في ملف `udmm_history.json`.

3.  **تشغيل خادم الـ API (اختياري):**
    يمكنك أيضًا التفاعل مع نواة UDMM v4 عبر خادم FastAPI بسيط.
    ```bash
    uvicorn src.udmm2.api.app:app --reload
    ```
    بعد ذلك، يمكنك إرسال طلبات إلى نقطة النهاية `/ask`:
    ```bash
    curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d '{"question": "أخبرني عن الجاذبات."}'
    ```

### ⚠️ ملاحظات هامة

-   **نموذج معماري أولي:** هذا التطبيق هو نموذج معماري أولي. استدعاء LLM في `UDMMCore` هو حاليًا عنصر نائب. لجعله وكيلاً وظيفيًا بالكامل، يجب عليك استبدال دالة `_get_llm_response` باستدعاء لـ LLM حقيقي (مثل OpenAI, Gemini).
-   **ديناميكيات مبسطة:** تم تهيئة مصفوفات الديناميكيات والتزاوج في `AttractorDynamics` بقيم عشوائية للعرض التوضيحي. لتطبيق أكثر تقدمًا، يجب أن تستند هذه المصفوفات إلى نموذج مدروس لكيفية تأثير هذه المفاهيم على بعضها البعض.
-   **ذاكرة أساسية:** نظام الذاكرة هو قائمة بسيطة من التضمينات (embeddings). يمكن ترقيته لاستخدام فهرسة FAISS الأكثر قوة من الإصدارات السابقة لتحسين الأداء.
