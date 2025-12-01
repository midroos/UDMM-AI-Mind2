import tkinter as tk
from tkinter import scrolledtext, simpledialog, messagebox, font
import threading

from src.udmm2.agent import UDMM_Agent

class ChatApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🧠 الوكيل الديناميكي الكامل (UDMM + AAR) - نسخة Tkinter")
        self.geometry("900x600")

        # Initialize Agent
        self.agent = UDMM_Agent("tkinter_agent")

        # --- Main Layout ---
        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Conversation Area (Left)
        conversation_frame = tk.Frame(main_frame)
        conversation_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.chat_log = scrolledtext.ScrolledText(conversation_frame, state='disabled', wrap=tk.WORD, font=("Arial", 10))
        self.chat_log.pack(fill=tk.BOTH, expand=True)

        input_frame = tk.Frame(conversation_frame)
        input_frame.pack(fill=tk.X, pady=(10, 0))

        self.entry_box = tk.Entry(input_frame, font=("Arial", 11))
        self.entry_box.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.entry_box.bind("<Return>", self.send_message)

        send_button = tk.Button(input_frame, text="إرسال", command=self.send_message)
        send_button.pack(side=tk.RIGHT, padx=(5, 0))

        # Status Panel (Right)
        status_frame = tk.Frame(main_frame, width=250)
        status_frame.pack(side=tk.RIGHT, fill=tk.Y)
        status_frame.pack_propagate(False)

        tk.Label(status_frame, text="📊 الحالة الداخلية للوكيل", font=("Arial", 12, "bold")).pack(pady=5, anchor="w")

        self.status_vars = {
            "energy": tk.StringVar(value="الطاقة: 1.00"),
            "arousal": tk.StringVar(value="الإثارة: 0.30"),
            "a_r": tk.StringVar(value="الوجدان الخام: 0.20"),
            "it": tk.StringVar(value="التوتر المعلوماتي: 0.00"),
            "kl": tk.StringVar(value="تباعد KL: 0.00"),
            "attractor": tk.StringVar(value="الحوض الحالي: stable")
        }

        for key, var in self.status_vars.items():
            tk.Label(status_frame, textvariable=var, font=("Arial", 10)).pack(anchor="w", pady=2)

        reset_button = tk.Button(status_frame, text="🚨 إعادة تعيين الوكيل", command=self.reset_agent)
        reset_button.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        # Initial state update
        self.update_status_display()

    def add_message(self, sender: str, message: str):
        self.chat_log.configure(state='normal')
        if sender == "user":
            self.chat_log.insert(tk.END, f"👤 أنت: {message}\n\n", "user_tag")
        else:
            self.chat_log.insert(tk.END, f"🤖 الوكيل: {message}\n\n", "agent_tag")
        self.chat_log.configure(state='disabled')
        self.chat_log.yview(tk.END)

        # Tag configuration for colors
        self.chat_log.tag_config("user_tag", foreground="blue")
        self.chat_log.tag_config("agent_tag", foreground="green")

    def send_message(self, event=None):
        msg = self.entry_box.get()
        if not msg.strip():
            return

        self.add_message("user", msg)
        self.entry_box.delete(0, tk.END)

        # Run agent response in a separate thread to avoid freezing the UI
        threading.Thread(target=self.get_agent_response, args=(msg,), daemon=True).start()

    def get_agent_response(self, msg: str):
        response, _ = self.agent.generate_response(msg)
        self.after(0, self.add_message, "assistant", response)
        self.after(0, self.update_status_display)

    def update_status_display(self):
        state = self.agent.get_detailed_state()
        self.status_vars["energy"].set(f"⚡ الطاقة: {state['body'][0]:.2f}")
        self.status_vars["arousal"].set(f"🔥 الإثارة: {state['body'][1]:.2f}")
        self.status_vars["a_r"].set(f"❤️ الوجدان الخام: {state['affect']['A_r']:.2f}")
        self.status_vars["it"].set(f"🧠 التوتر المعلوماتي: {state['meta']['IT']:.2f}")
        self.status_vars["kl"].set(f"🔀 تباعد KL: {state['meta']['KL_divergence']:.2f}")
        self.status_vars["attractor"].set(f"🌊 الحوض الحالي: {state.get('attractor', 'غير معروف')}")

    def reset_agent(self):
        if messagebox.askyesno("تأكيد", "هل أنت متأكد أنك تريد إعادة تعيين الوكيل؟ سيتم فقدان كل الذاكرة الحالية."):
            self.agent = UDMM_Agent("tkinter_agent")
            self.chat_log.configure(state='normal')
            self.chat_log.delete(1.0, tk.END)
            self.chat_log.configure(state='disabled')
            self.update_status_display()
            messagebox.showinfo("نجاح", "تم إعادة تعيين الوكيل بنجاح!")


if __name__ == "__main__":
    app = ChatApplication()
    app.mainloop()
