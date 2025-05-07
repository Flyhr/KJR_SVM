import tkinter as tk
from tkinter import scrolledtext
import requests

# DeepSeek API 信息
API_URL = "https://api.deepseek.com/v1/chat/completions"
API_KEY = "sk-075b9706ee4e42beae3c2961581fe906"  # 替换为你的 DeepSeek API 密钥

# 定义调用 DeepSeek API 的函数
def call_deepseek_api(prompt):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.RequestException as e:
        return f"请求出错: {e}"
    except KeyError:
        return "无法解析 API 响应"

# 定义发送消息的函数
def send_message():
    user_input = input_box.get()
    if user_input:
        # 在聊天窗口显示用户消息
        chat_box.insert(tk.END, f"你: {user_input}\n")
        # 调用 DeepSeek API 获取回复
        response = call_deepseek_api(user_input)
        # 在聊天窗口显示回复消息
        chat_box.insert(tk.END, f"DeepSeek: {response}\n")
        # 清空输入框
        input_box.delete(0, tk.END)

# 创建主窗口
root = tk.Tk()
root.title("DeepSeek 聊天窗口")

# 创建聊天窗口
chat_box = scrolledtext.ScrolledText(root, width=80, height=20)
chat_box.pack(padx=10, pady=10)

# 创建输入框
input_box = tk.Entry(root, width=60)
input_box.pack(padx=10, pady=5)

# 创建发送按钮
send_button = tk.Button(root, text="发送", command=send_message)
send_button.pack(pady=5)

# 运行主循环
root.mainloop()