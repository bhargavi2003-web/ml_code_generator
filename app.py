
import os
import gradio as gr

from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    LiteLLMModel
)

# 1. Configure Google Gemini API Key
os.environ["GOOGLE_API_KEY"] = "AQ.Ab8RN6JBLOOnLbWlYaV2Q3YxGNA0iBlCSnKwQ-j6jNZV3wAKXQ"

# 2. Initialize Gemini Model
model = LiteLLMModel(
    model_id="gemini/gemini-3.8-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)

# 3. Create ML Code Generation Agent
ml_code_agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()],
    additional_authorized_imports=[
        "pandas",
        "numpy",
        "sklearn",
        "json"
    ],
    model=model
)

# 4. Define Function to Generate ML Code
def generate_ml_code(dataset_file, user_prompt):

    if dataset_file is None:
        return "Please upload a dataset file."

    if not user_prompt.strip():
        return "Please enter your machine learning task."

    file_path = dataset_file

    prompt = f"""
    You are a Machine Learning coding assistant.

    Dataset path: {file_path}

    User Request: {user_prompt}

    Instructions:
    1. Load the dataset using pandas.
    2. Analyze the dataset.
    3. Perform the requested machine learning task.
    4. Generate complete Python code.
    5. Display the final results and evaluation metrics.
    """

    try:
        response = ml_code_agent.run(prompt)
        return str(response)

    except Exception as e:
        return f"Error: {str(e)}"

# 5. Create Simple Gradio Interface
iface = gr.Interface(
    fn=generate_ml_code,
    inputs=[
        gr.File(label="Upload Dataset"),
        gr.Textbox(
            lines=3,
            label="Enter Your Machine Learning Task",
            placeholder="Example: Build a Logistic Regression model"
        )
    ],
    outputs=gr.Textbox(
        label="Generated ML Code",
        lines=15
    ),
    title="ML Code Generator",
    description="Upload your dataset and enter a machine learning task to generate Python code."
)

# 6. Launch Application
if __name__ == "__main__":
    iface.launch()