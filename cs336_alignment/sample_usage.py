from vllm import LLM, SamplingParams

MODEL_PATH = '/kun-data/assignment5-alignment/models/Qwen/Qwen2.5-Math-1.5B'
# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create a sampling params object, stopping generation on newline.
sampling_params = SamplingParams(
    temperature=0.7, 
    top_p=0.95, 
    max_tokens=1024, 
    stop=["\n"]
)
# Create an LLM.
llm = LLM(model=MODEL_PATH)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
