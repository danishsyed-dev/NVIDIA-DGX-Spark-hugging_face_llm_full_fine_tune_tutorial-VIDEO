"""
FoodExtract Demo - Extract food and drink items from text
Using fine-tuned Gemma 3 270M model
"""

import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Model configuration
MODEL_ID = "danishali11903/gemma-3-270m-it-FoodExtract-v1"  # Update with your model ID

# System prompt for the model
SYSTEM_PROMPT = """You are a helpful assistant that extracts food and drink information from text.
For each input, respond with:
- food_or_drink: 1 if food/drink is present, 0 if not
- tags: category tags (fi=food items, di=drink items, re=recipe, me=menu, etc.)
- foods: comma-separated list of food items
- drinks: comma-separated list of drink items"""

# Load model and tokenizer (done at startup)
print("Loading model...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    MODEL_LOADED = True
    print("Model loaded successfully!")
except Exception as e:
    MODEL_LOADED = False
    print(f"Error loading model: {e}")


def extract_food_drinks(text: str) -> str:
    """Extract food and drink items from the input text."""
    
    if not MODEL_LOADED:
        return "⚠️ Model not loaded. Please check the model ID and try again."
    
    if not text.strip():
        return "Please enter some text to analyze."
    
    # Format the prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text}
    ]
    
    # Tokenize
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    )
    
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    
    return response


# Example inputs
examples = [
    ["A plate of rice cakes, salmon, cottage cheese and small cherry tomatoes with a cup of tea."],
    ["I had a delicious burger with fries and a chocolate milkshake for lunch."],
    ["The recipe calls for 2 cups of flour, 1 egg, and a glass of milk."],
    ["Just grabbed a coffee from Starbucks."],
    ["Tonight's menu: grilled chicken, mashed potatoes, steamed vegetables, and sparkling water."]
]

# Create the Gradio interface
with gr.Blocks(
    title="🍕 FoodExtract - Food & Drink Extractor",
    theme=gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="amber"
    )
) as demo:
    
    gr.Markdown("""
    # 🍕 FoodExtract - Food & Drink Extractor
    
    Extract food and drink items from any text using a fine-tuned **Gemma 3 270M** model.
    
    This model was trained on the [FoodExtract-1k](https://huggingface.co/datasets/mrdbourke/FoodExtract-1k) dataset
    to identify and categorize food/drink mentions in text.
    """)
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="📝 Input Text",
                placeholder="Enter text containing food or drink mentions...",
                lines=4
            )
            submit_btn = gr.Button("🔍 Extract", variant="primary", size="lg")
        
        with gr.Column():
            output_text = gr.Textbox(
                label="📊 Extraction Results",
                lines=6,
                interactive=False
            )
    
    gr.Examples(
        examples=examples,
        inputs=input_text,
        outputs=output_text,
        fn=extract_food_drinks,
        cache_examples=False
    )
    
    gr.Markdown("""
    ---
    ### 🏷️ Output Format
    - **food_or_drink**: 1 if food/drink present, 0 otherwise
    - **tags**: Category tags (fi=food items, di=drink items, re=recipe, me=menu, etc.)
    - **foods**: Comma-separated list of food items
    - **drinks**: Comma-separated list of drink items
    
    ### 📚 Resources
    - [Fine-tuning Tutorial](https://github.com/danishsyed-dev/NVIDIA-DGX-Spark-hugging_face_llm_full_fine_tune_tutorial-VIDEO)
    - [Model on Hugging Face](https://huggingface.co/danishali11903/gemma-3-270m-it-FoodExtract-v1)
    """)
    
    submit_btn.click(
        fn=extract_food_drinks,
        inputs=input_text,
        outputs=output_text
    )

# Launch
if __name__ == "__main__":
    demo.launch()
