import gradio as gr
import random
import json

def load_data():
    with open("data/test.jsonl", "r") as f:
        data = [json.loads(line) for line in f]
    
    with open("output/normal_chain_results.jsonl", "r") as f:
        results = [json.loads(line) for line in f]
        for i, d in enumerate(data):
            d["outputs"] = [(results[i]["output"], "normal_chain_results")]
           
    with open("output/rag_chain_embedding_openai_results.jsonl", "r") as f:
        results = [json.loads(line) for line in f]
        for i, d in enumerate(data):
            d["outputs"].append((results[i]["output"], "rag_chain_embedding_openai_results"))
            
    with open("output/finetuned_chain_ckpt_200_results.jsonl", "r") as f:
        results = [json.loads(line) for line in f]
        for i, d in enumerate(data):
            d["outputs"].append((results[i]["output"], "finetuned_chain_ckpt_200_results"))
            
    return data      

option1_model = None
option2_model = None
option3_model = None

index = None
    
def next(data):
    import random
    selected = random.choice(data)
        
    outputs = selected["outputs"]
    random.shuffle(outputs)
    
    global option1_model, option2_model, option3_model
    option1_model = outputs[0][1]
    option2_model = outputs[1][1]
    option3_model = outputs[2][1]
    
    global index
    index = selected["idx"]
    
    return gr.update(label=f"問題 {index+1}", value=selected["question"]), gr.update(value=selected["answer"]), \
        gr.update(label=f"模型A", value=outputs[0][0]), gr.update(label=f"模型B", value=outputs[1][0]), gr.update(label=f"模型C", value=outputs[2][0]), \
        gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), \
        gr.update(visible=False)
        

def btn_clicked(idx):
    
    if idx != -1:
        if idx == 1:
            log = f"{index} {option1_model}"
        elif idx == 2:
            log = f"{index} {option2_model}"
        elif idx == 3:
            log = f"{index} {option3_model}"
        else:
            raise ValueError("Invalid index")
        
        with open("log/blind_selection.log", "a") as f:
            f.write(log + "\n")
    
    return gr.update(visible=False), gr.update(label=option1_model), \
        gr.update(visible=False), gr.update(label=option2_model), \
        gr.update(visible=False), gr.update(label=option3_model), \
        gr.update(visible=False), \
        gr.update(visible=True), 

def gradio(data):
    with gr.Blocks() as demo:
        gr.Markdown("## 模型盲測系統")
        
        question_text = gr.TextArea(label="問題")
        reference_text = gr.TextArea(label="參考答案")
        
        gr.Markdown("請對下列模型的輸出，選擇您認為最好的回應：")
        with gr.Row():
            output1 = gr.TextArea(label="模型A")
            output2 = gr.TextArea(label="模型B")
            output3 = gr.TextArea(label="模型C")
        
        next_btn = gr.Button("下一題")
        
        with gr.Row():
            choice1_btn = gr.Button("A", visible=False)
            choice2_btn = gr.Button("B", visible=False)
            choice3_btn = gr.Button("C", visible=False)
        
        all_bad_btn = gr.Button("都不夠好", visible=False)
            
        next_btn.click(lambda: next(data), outputs=[ 
            question_text, reference_text, 
            output1, output2, output3, 
            choice1_btn, choice2_btn, choice3_btn, all_bad_btn,
            next_btn
        ])
        choice1_btn.click(lambda: btn_clicked(1), outputs=[
            choice1_btn, output1, 
            choice2_btn, output2, 
            choice3_btn, output3, 
            all_bad_btn, next_btn
        ])
        choice2_btn.click(lambda: btn_clicked(2), outputs=[
            choice1_btn, output1, 
            choice2_btn, output2, 
            choice3_btn, output3, 
            all_bad_btn, next_btn
        ])
        choice3_btn.click(lambda: btn_clicked(3), outputs=[
            choice1_btn, output1, 
            choice2_btn, output2, 
            choice3_btn, output3, 
            all_bad_btn, next_btn
        ])
        all_bad_btn.click(lambda: btn_clicked(-1), outputs=[
            choice1_btn, output1, 
            choice2_btn, output2, 
            choice3_btn, output3, 
            all_bad_btn, next_btn
        ])
    
        demo.launch()

if __name__ == "__main__":
    data = load_data()
    gradio(data)
