from openai import OpenAI

# Your Router Config
ROUTER_IP = "http://43.157.202.81:20128/v1"
ROUTER_KEY = "sk-6b3ac6ef8e3b70c9-qy4p0x-767d185d"
MODEL_ID = "MiniMax-M3"  

client = OpenAI(
    base_url=ROUTER_IP,
    api_key=ROUTER_KEY
)

def get_minimax_response(prompt_text):
    """
    A wrapper to bypass the router's buffering bug by forcing a stream 
    and manually compiling the chunks into a final string.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.7,
            stream=True # ALWAYS TRUE for this router/model combo
        )
        
        compiled_text = ""
        
        for chunk in response:
            # Safely extract the text delta, defaulting to "" if it's missing
            delta = chunk.choices[0].delta.content or ""
            compiled_text += delta
            
        return compiled_text

    except Exception as e:
        return f"[ERROR] {e}"

# --- Test it out ---
print("Sending request...")
prompt = "You are a helpful assistant. Hello, are you receiving my messages? Please reply with only the word: YES."

final_answer = get_minimax_response(prompt)

print("\n--- COMPILED RESULT ---")
print(final_answer)