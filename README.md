# 🧠 UDMM v4: Gemini-Powered Agent

[🇬🇧 English](#-english) | [🇦🇪 العربية](#-arabic)

---

## 🇬🇧 English

This project is a Python implementation of the **UDMM v4 architecture**, a self-contained, theoretically-grounded cognitive agent powered exclusively by **Google's Gemini LLM**. This version integrates all core components—Dynamic Attractors, a Hierarchical Intent Manager, and a Dynamic Memory Module—into a single, powerful script that interacts with the Gemini API.

### 🔬 Key Architectural Features

1.  **Gemini-Exclusive Core (`UDMMCore`):**
    -   A single class that orchestrates all cognitive functions.
    -   **Attractor Dynamics:** Models a 5D attractor state that evolves based on inputs and internal tension.
    -   **Intent Manager:** Manages a three-level hierarchy of goals.
    -   **Dynamic Memory:** Implements a `MemoryModule` that restructures its "schemas" based on Informational Tension (IT).
    -   **Gemini LLM Wrapper:** Includes a dedicated wrapper for making API calls to the Gemini series of models.

2.  **Interactive UI with API Key Management:**
    -   A Streamlit-based user interface is the primary way to interact with the agent.
    -   It features a secure sidebar input for your Gemini API key, which is required to activate the agent.
    -   The UI visualizes the agent's internal attractor state in real-time.

### 📂 Project Structure

```
udmm-agent/
├── src/udmm2/
│   ├── api/
│   │   └── app.py          # Optional FastAPI server
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

2.  **Run the Interactive Chat UI:**
    This is the only way to run the agent with full functionality.
    ```bash
    streamlit run ui_streamlit.py
    ```

3.  **Enter Your API Key:**
    -   The application will open in your web browser.
    -   On the sidebar, find the text field labeled "Gemini API Key".
    -   Enter your key and press Enter. The agent will initialize and be ready to chat.

### ⚙️ Other Ways to Run

-   **API Server (for developers):**
    This requires setting the `GEMINI_API_KEY` as an environment variable before running.
    ```bash
    # For Windows (PowerShell)
    $env:GEMINI_API_KEY="your_key_here"
    uvicorn src.udmm2.api.app:app --reload

    # For Linux/macOS
    export GEMINI_API_KEY="your_key_here"
    uvicorn src.udmm2.api.app:app --reload
    ```

---

## 🇦🇪 العربية

### 🧠 UDMM v4: وكيل يعمل بنموذج Gemini

هذا المشروع هو تطبيق بلغة بايثون لمعمارية **UDMM v4**، وهو وكيل إدراكي متكامل ومبني على أسس نظرية، يعمل حصريًا بواسطة **نموذج Gemini LLM من Google**. هذه النسخة تدمج جميع المكونات الأساسية - الجاذبات الديناميكية، مدير النوايا الهرمي، ووحدة الذاكرة الديناميكية - في سكربت واحد قوي يتفاعل مع Gemini API.

### 🔬 الميزات المعمارية الأساسية

1.  **نواة حصرية لـ Gemini (`UDMMCore`):**
    -   فئة واحدة تنسق جميع الوظائف الإدراكية.
    -   **ديناميكيات الجاذب:** تنمذج حالة جاذب خماسية الأبعاد تتطور بناءً على المدخلات والتوتر الداخلي.
    -   **مدير النوايا:** يدير تسلسلًا هرميًا للأهداف من ثلاثة مستويات.
    -   **الذاكرة الديناميكية:** تطبق `MemoryModule` التي تعيد هيكلة "مخططاتها" بناءً على التوتر المعلوماتي (IT).
    -   **مغلف Gemini LLM:** يتضمن مغلفًا مخصصًا لإجراء استدعاءات API لنماذج سلسلة Gemini.

2.  **واجهة مستخدم تفاعلية مع إدارة مفتاح API:**
    -   واجهة مستخدم مبنية على Streamlit هي الطريقة الأساسية للتفاعل مع الوكيل.
    -   تتميز بإدخال آمن في الشريط الجانبي لمفتاح Gemini API الخاص بك، وهو مطلوب لتفعيل الوكيل.
    -   تعرض الواجهة تصورًا بيانيًا لحالة الجاذب الداخلية للوكيل في الوقت الفعلي.

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

2.  **تشغيل الواجهة التفاعلية:**
    هذه هي الطريقة الوحيدة لتشغيل الوكيل بكامل وظائفه.
    ```bash
    streamlit run ui_streamlit.py
    ```

3.  **إدخال مفتاح الـ API الخاص بك:**
    -   سيتم فتح التطبيق في متصفح الويب الخاص بك.
    -   في الشريط الجانبي، ابحث عن حقل النص المسمى "Gemini API Key".
    -   أدخل مفتاحك واضغط على Enter. سيتم تهيئة الوكيل ويكون جاهزًا للدردشة.

### ⚙️ طرق تشغيل أخرى

-   **خادم الـ API (للمطورين):**
    يتطلب هذا تعيين `GEMINI_API_KEY` كمتغير بيئة قبل التشغيل.
    ```bash
    # لنظام Windows (PowerShell)
    $env:GEMINI_API_KEY="your_key_here"
    uvicorn src.udmm2.api.app:app --reload

    # لنظام Linux/macOS
    export GEMINI_API_KEY="your_key_here"
    uvicorn src.udmm2.api.app:app --reload
    ```
