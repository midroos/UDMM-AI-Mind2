
import tkinter as tk
from tkinter import scrolledtext, messagebox, simpledialog
from src.udmm2.udmm_core import UDMMAgent
import threading

class ChatApplication:
    def __init__(self, master):
        self.master = master
        master.title("UDMM Agent Chat")
        master.geometry("700x500")

        self.agent = UDMMAgent()

        # Chat history display
        self.chat_history = scrolledtext.ScrolledText(master, state='disabled', wrap=tk.WORD, bg="#f0f0f0", font=("Arial", 10))
        self.chat_history.pack(padx=10, pady=10, expand=True, fill=tk.BOTH)

        # User input frame
        input_frame = tk.Frame(master, pady=5)
        input_frame.pack(fill=tk.X, padx=10, pady=5)

        self.user_input = tk.Entry(input_frame, font=("Arial", 12))
        self.user_input.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        self.user_input.bind("<Return>", self.send_message)

        # Send button
        self.send_button = tk.Button(input_frame, text="إرسال", command=self.send_message, font=("Arial", 10, "bold"), bg="#4CAF50", fg="white")
        self.send_button.pack(side=tk.RIGHT)

        # Teach button
        self.teach_button = tk.Button(input_frame, text="علّم", command=self.teach_agent, font=("Arial", 10), bg="#2196F3", fg="white")
        self.teach_button.pack(side=tk.RIGHT, padx=5)

    def add_message(self, sender, message):
        self.chat_history.configure(state='normal')
        self.chat_history.insert(tk.END, f"{sender}: {message}\n\n")
        self.chat_history.configure(state='disabled')
        self.chat_history.yview(tk.END)

    def send_message(self, event=None):
        user_text = self.user_input.get()
        if user_text.strip():
            self.add_message("أنت", user_text)
            self.user_input.delete(0, tk.END)

            # Run agent in a separate thread to avoid freezing the UI
            threading.Thread(target=self.get_agent_response, args=(user_text,)).start()

    def get_agent_response(self, user_text):
        try:
            response = self.agent.perceive_and_answer(user_text)
            self.master.after(0, self.add_message, "الوكيل", response['response'])
        except Exception as e:
            self.master.after(0, messagebox.showerror, "Error", f"An error occurred: {e}")

    def teach_agent(self):
        try:
            question = simpledialog.askstring("علّم الوكيل", "أدخل السؤال:", parent=self.master)
            if question:
                answer = simpledialog.askstring("علّم الوكيل", f"أدخل الإجابة عن:\n'{question}'", parent=self.master)
                if answer:
                    self.agent.teach(question, answer)
                    messagebox.showinfo("تم", "تم تعليم الوكيل بنجاح.", parent=self.master)
        except Exception as e:
            messagebox.showerror("خطأ", f"حدث خطأ أثناء التعليم: {e}", parent=self.master)

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatApplication(root)
    root.mainloop()
