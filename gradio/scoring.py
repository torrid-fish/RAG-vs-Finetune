import gradio as gr
import random
import json
import os

data = None
used_model = None
# Load the selected file and prepare the data
def load_data(selected_file):
    global data
    with open("data/test.jsonl", "r") as f:
        data = [json.loads(line) for line in f]
    
    with open(selected_file, "r") as f:
        results = [json.loads(line) for line in f]
        for i, d in enumerate(data):
            d["output"] = results[i]["output"]
        
    global used_model
    used_model = selected_file.split("/")[-1].replace(".jsonl", "")
        
    return gr.update(visible=True), gr.update(value=f"已載入 {selected_file}", visible=True), \
        gr.update(visible=False), gr.update(visible=False)  


index = 0
point = 0
    
# This function will load the next question and corresponding model output
def next():
    global index
    selected = data[index]
    index += 1
    
    return gr.update(label=f"問題 {index}/{len(data)}", value=selected["question"]), gr.update(value=selected["answer"]), \
        gr.update(label=f"模型輸出", value=selected["output"]), \
        gr.update(visible=True), gr.update(visible=True), \
        gr.update(visible=False)

# Function to handle user selection (correct or incorrect)
def btn_clicked(is_correct):
    global index
    if is_correct:
        global point
        point += 1
        
    with open(f"log/scores_{used_model}.jsonl", "a+") as f:
        f.write(json.dumps({"idx": index, "correct": is_correct}) + "\n")
        
    if index == len(data):
        return gr.update(visible=False), gr.update(visible=False), \
        gr.update(visible=False), \
        gr.update(visible=False), gr.update(visible=False), \
        gr.update(visible=False), \
        gr.update(visible=True, value=f"## 結束！正確率: {point / len(data)}")
    
    selected = data[index]
    index += 1
    
    return gr.update(label=f"問題 {index}/{len(data)}", value=selected["question"]), gr.update(value=selected["answer"]), \
        gr.update(label=f"模型輸出", value=selected["output"]), \
        gr.update(visible=True), gr.update(visible=True), \
        gr.update(visible=False), \
        gr.update(visible=False)

# Function to load the Gradio interface
def gradio():
    files = [f"output/{file}" for file in os.listdir("output") if file.endswith(".jsonl")]
    files.sort()
    
    with gr.Blocks() as demo:
        gr.Markdown("## 模型評分系統")
        
        selected_file = gr.Dropdown(choices=files, label="選擇要評分的檔案")
        load_btn = gr.Button("載入檔案")
        info_text = gr.Markdown(visible=False)
        
        question_text = gr.TextArea(label="問題", interactive=False)
        with gr.Row():
            reference_text = gr.TextArea(label="參考答案", interactive=False)
            output_text = gr.TextArea(label="模型輸出", interactive=False)
        
        next_btn = gr.Button("下一題", visible=False)
        result_text = gr.Markdown(visible=False)
        
        with gr.Column():
            correct_btn = gr.Button("✅ 正確", visible=False)
            incorrect_btn = gr.Button("❌ 錯誤", visible=False)
        
        load_btn.click(lambda file: load_data(file), inputs=[selected_file], outputs=[
            next_btn, info_text, selected_file, load_btn
        ])
        
        next_btn.click(lambda: next(), outputs=[
            question_text, reference_text, output_text, 
            correct_btn, incorrect_btn, next_btn
        ])
        
        correct_btn.click(lambda: btn_clicked(1), outputs=[
            question_text, reference_text, output_text, 
            correct_btn, incorrect_btn, next_btn,
            result_text
        ])
        
        incorrect_btn.click(lambda: btn_clicked(0), outputs=[
            question_text, reference_text, output_text, 
            correct_btn, incorrect_btn, next_btn,
            result_text
        ])
    
        demo.launch()

if __name__ == "__main__":
    gradio()
