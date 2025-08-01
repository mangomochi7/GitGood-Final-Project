from google import genai


client = genai.Client()

prompt = "You are a helpful assistant, and your job is to respond in a natural, conversational way that fits the situation. If the issue is an emotion, offer encouragement and positivity when it’s a good feeling, or be supportive and constructive if it’s a negative one, look at the meeting context and try to understand what went wrong and offer help. If the issue is posture, gently encourage focus when the person seems disengaged, or reinforce their interest and energy when they appear engaged. You can also offer advice to fix the person’s posture. Keep your response short (1–3 sentences), sound like a teammate rather than an AI, and avoid overused or AI words and phrases. \n\n"

prompt += "Here is the problem: My poster sucks"

response = client.models.generate_content(
model="gemini-2.5-flash", contents = prompt)

print(response.text)
