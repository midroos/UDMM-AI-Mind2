# 🧠 UDMM Cognitive Agent

هذا المشروع يهدف إلى بناء وكيل إدراكي (Cognitive Agent) مستند إلى النموذج الديناميكي الشامل للعقل (UDMM).
الوكيل يدمج بين LLM (مثل OpenAI أو Gemini) وذاكرة RAG (قائمة على FAISS) ونظام لإدارة النوايا (Intent Manager) ومحرك محاكاة ديناميكي في إطار واحد يعكس مبادئ UDMM.

---

# 📂 هيكل المشروع

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

---

# ⚙️ المزايا الأساسية

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

---

# 🚀 التشغيل

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

---

# 🧪 التجريبية (Experimental UDMM)

هذه النسخة تعتبر بروتوتايب (Prototype) يطبق بعض مبادئ UDMM:

- **🧲 الجاذبات (Attractors):**
  تُحاكى من خلال نظام النوايا + قيم مرجحة (knowledge, social...).

- **⚡ الحالة الديناميكية (Dynamic State):**
  يظهر عند تفعيل الدورة بين الإدخال → المحاكاة → LLM → الذاكرة → التفسير. حالة المحاكاة (التوتر، الزمن) تؤثر على prompt الـ LLM.

- **🔄 التعلم الديناميكي:**
  يمكن "تعليم" النظام بمعلومة جديدة (`/teach`)، وسيخزنها في FAISS ليسترجعها لاحقًا.

- **🌀 التفاعل الدلالي:**
  النموذج يحاول ربط المعرفة المخزنة مع السؤال الحالي (Retrieval Augmented).

---

# 📊 كيف تختبر؟

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

---

# ⚠️ حدود النسخة الحالية

-   تعتمد على LLM خارجي (OpenAI/Gemini) ولا تولد معرفة ذاتية بعد.
-   الجاذبات والنوايا والمحاكاة ما زالت أولية (rudimentary).
-   لا توجد حتى الآن إدارة متقدمة للسياق الطويل أو المحاكاة الزمنية المعقدة.
-   واجهة Streamlit مجرد نسخة بدائية للتجريب.
