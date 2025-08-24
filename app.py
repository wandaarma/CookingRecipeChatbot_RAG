import streamlit as st
import pandas as pd
import faiss
import pickle
import gdown
import numpy as np
from sentence_transformers import SentenceTransformer
from streamlit.components.v1 import html

st.set_page_config(page_title="Recipe Chatbot", page_icon="🍲", layout="wide")

st.markdown(
    """
<style>
    body {
        background-color: #F7F9FC; 
        overflow-y: hidden; 
        font-family: 'Arial', sans-serif;
    }
    
    .block-container {
        padding: 2rem; 
        height: 100vh; 
    }
    
    [data-testid="column"]:nth-of-type(2) > div > [data-testid="stVerticalBlock"] {
        max-width: 400px; 
        height: 80vh; 
        margin: 0 auto; 
        border: 1px solid #D0D0D0; 
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); 
        background-color: #FFFFFF; 
        display: flex; 
        flex-direction: column;
    }
    
    .chat-header {
        background: linear-gradient(90deg, rgba(75,5,159,1) 0%, rgba(111,75,242,1) 100%);
        color: #FFFFFF; 
        padding: 1rem; 
        border-top-left-radius: 12px; 
        border-top-right-radius: 12px;
        font-size: 1.2rem; 
        font-weight: bold; 
        display: flex; 
        align-items: center; 
    }
        
    .chat-header .icon {
        margin-right: 10px; 
        font-size: 1.5rem; 
    }
    
    .message-container-div {
        padding: 1rem; 
        font-size: 0.95rem;
        height: 300px;        
        overflow-y: auto;     
        background-color: #FAFAFA; 
    }
    
    .message {
        margin-bottom: 1rem; 
        padding: 0.8rem 1rem; 
        border-radius: 16px; 
        max-width: 80%;
        word-wrap: break-word; 
        display: inline-block; 
        clear: both; 
    }
    
    .user-message-wrapper {
        text-align: right; 
    }
    
    .user-message {
        background-color: #dcf8c6; 
        border-bottom-right-radius: 4px; 
        border-top-left-radius: 12px; 
        text-align: left;
    }
    
    .bot-message {
        background-color: #ffffff;
        border: 1px solid #D0D0D0;
        border-bottom-left-radius: 4px;
        border-top-right-radius: 12px; 
    }
    
    .input-area-wrapper {
        padding: 0.5rem 1rem;
        border-top: 1px solid #e0e0e0;
        background-color: #f0f0f0;
        flex-shrink: 0;
        border-radius: 0 0 12px 12px; 
    }

    .stButton>button {
        width: 100%;
        background-color: #4B059F; 
        color: white;
        font-size: 1rem;
        border: none;
        border-radius: 8px;
        padding: 10px; 
        transition: background-color 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #6A0DAD;  
        color: #F0EFFF;
    }

    .input-area-wrapper form {
        margin-bottom: 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_data():
    gdown.download("https://drive.google.com/uc?id=1KSwxoIoD1WzEwTI97SfUyJT0o8-Htd34", "cleaned_recipes.pkl", quiet=False)
    gdown.download("https://drive.google.com/uc?id=1A73z7jBdr424A6zeC84z7m2J8siNbHjl", "recipe_texts.pkl", quiet=False)
    gdown.download("https://drive.google.com/uc?id=1vWiFvPYLuo5qQunWWEfQILwN-92eWNbu", "recipe_faiss.index", quiet=False)

    df = pd.read_pickle("cleaned_recipes.pkl")
    with open("recipe_texts.pkl", "rb") as f:
        corpus = pickle.load(f)
    index = faiss.read_index("recipe_faiss.index")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return df, corpus, index, model
df, corpus, index, model = load_data()



def search_recipe_by_title(query, top_k=3):
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec, dtype="float32"), top_k)
    results = [
        df.iloc[idx] for dist, idx in zip(D[0], I[0]) if idx != -1 and dist <= 1.5
    ]
    return results if results else None


def format_list(raw_text):
    if isinstance(raw_text, str):
        cleaned_text = raw_text.strip("[]").replace('"', "").replace("'", "")
        items = [x.strip() for x in cleaned_text.split(",") if x.strip()]
        return "\n".join([f"- {i}" for i in items])
    return raw_text


def format_steps(raw_text):
    if isinstance(raw_text, str):
        items = [x.strip() for x in raw_text.split(". ") if x.strip()]
        return "\n".join([f"{idx+1}. {i}" for idx, i in enumerate(items)])
    return raw_text


def auto_scroll():
    js = """
    <script>
        function scroll(element){ element.scrollTop = element.scrollHeight }
        var element = window.parent.document.querySelector('[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] > [data-testid="stContainer"]');
        if (element) { scroll(element); }
    </script>
    """
    html(js, height=0)


# --------------------
# Session State Initialization
# --------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! 👋 What would you like to do today? You can choose an option below or type your question directly.",
        }
    ]
if "mode" not in st.session_state:
    st.session_state.mode = "menu"

col1, col2 = st.columns([0.8, 1.2])

# --------------------
# Left Column: Info
# --------------------
with col1:
    # Header
    st.markdown(
        "<h1 style='color: #4B059F;'>🍲 Recipe Chatbot</h1>", 
        unsafe_allow_html=True
    )
    # Welcome Message
    st.markdown("<h3>Welcome Recipian!</h3>", unsafe_allow_html=True)
    st.write(
        "I'm here to help you with your cooking adventures. "
        "You can ask me for **ingredients** or **cooking steps** for various recipes. "
        "Just use the chat on the right to get started!"
    )
    # Recipe of the Day
    st.markdown("<h3>Recipe of the Day</h3>", unsafe_allow_html=True)
    st.image("aglio.jpg", width=150)
    st.write("🌟 **Spaghetti Aglio e Olio**: A simple and quick Italian pasta dish made with garlic, olive oil, and chili flakes.") 
    st.write("Try it out and let me know your thoughts!")

# --------------------
# Right Column: Chat
# --------------------
with col2:
    # Header
    st.markdown(
        """<div class="chat-header"><span class="icon">🤖</span> Recipe Assistant</div>""",
        unsafe_allow_html=True,
    )

    # Messages container
    message_container = st.container()
    with message_container:
        messages_html = '<div class="message-container-div">'
        for message in st.session_state.messages:
            wrapper_class = (
                "user-message-wrapper"
                if message["role"] == "user"
                else "bot-message-wrapper"
            )
            message_class = (
                "user-message" if message["role"] == "user" else "bot-message"
            )
            content_html = message["content"].replace("\n", "<br>")
            messages_html += f'<div class="{wrapper_class}"><div class="message {message_class}">{content_html}</div></div>'
        messages_html += "</div>"

        st.markdown(messages_html, unsafe_allow_html=True)


    # Input area
    input_container = st.container()
    with input_container:
        st.markdown('<div class="input-area-wrapper">', unsafe_allow_html=True)

        # Row tombol aksi (dua tombol sejajar, sama lebar)
        btn_col1, btn_col2 = st.columns([1, 1])
        if btn_col1.button("Check Ingredients 🥕", use_container_width=True):
            st.session_state.mode = "waiting_ingredient"
            st.session_state.messages.append({"role": "user", "content": "Check Ingredients"})
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Sure! What is the name of the recipe you want to check?",
            })
            st.rerun()
        if btn_col2.button("Check Cooking Steps 📝", use_container_width=True):
            st.session_state.mode = "waiting_direction"
            st.session_state.messages.append({"role": "user", "content": "Check Cooking Steps"})
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Sure! What is the name of the recipe you want to check?",
            })
            st.rerun()

        # Row input (text box besar + tombol Send kecil)
        with st.form(key="chat_form", clear_on_submit=True):
            form_col1, form_col2 = st.columns([4, 1])  
            prompt = form_col1.text_input(
                "Type your message...",
                placeholder="Type your message here...",
                label_visibility="collapsed",
            )
            submitted = form_col2.form_submit_button("Send", use_container_width=True)

            if submitted and prompt:
                # Main Chat Logic
                st.session_state.messages.append({"role": "user", "content": prompt})
                user_lower = prompt.lower()
                bot_reply = ""

                # Mode tunggu input resep
                if st.session_state.mode in ["waiting_ingredient", "waiting_direction"]:
                    results = search_recipe_by_title(prompt, top_k=5)
                    if results:
                        first_recipe = results[0]
                        st.session_state.last_results = results
                        st.session_state.last_index = 0

                        if st.session_state.mode == "waiting_ingredient":
                            bot_reply = f"Here are the ingredients for **{first_recipe['title']}**:\n\n{format_list(first_recipe['ingredients'])}"
                            st.session_state.mode_type = "ingredient"
                        else:
                            bot_reply = f"Here are the cooking steps for **{first_recipe['title']}**:\n\n{format_steps(first_recipe['directions'])}"
                            st.session_state.mode_type = "step"

                        st.session_state.messages.append(
                            {"role": "assistant", "content": bot_reply}
                        )
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": "Do you want to see another recipe? (yes/no)",
                            }
                        )
                        st.session_state.mode = "ask_another"
                    else:
                        bot_reply = "Sorry, I couldn’t find that recipe. 😔 Please try another name."
                        st.session_state.messages.append(
                            {"role": "assistant", "content": bot_reply}
                        )

                # Mode yes/no
                elif st.session_state.mode == "ask_another":
                    if "yes" in user_lower:
                        st.session_state.last_index += 1
                        if st.session_state.last_index < len(
                            st.session_state.last_results
                        ):
                            recipe = st.session_state.last_results[
                                st.session_state.last_index
                            ]
                            if st.session_state.mode_type == "ingredient":
                                bot_reply = f"Here are the ingredients for **{recipe['title']}**:\n\n{format_list(recipe['ingredients'])}"
                            else:
                                bot_reply = f"Here are the cooking steps for **{recipe['title']}**:\n\n{format_steps(recipe['directions'])}"
                            st.session_state.messages.append(
                                {"role": "assistant", "content": bot_reply}
                            )
                            st.session_state.messages.append(
                                {
                                    "role": "assistant",
                                    "content": "Do you want to see another recipe? (yes/no)",
                                }
                            )
                            st.session_state.mode = "ask_another"
                        else:
                            st.session_state.messages.append(
                                {
                                    "role": "assistant",
                                    "content": "No more similar recipes found. 😔",
                                }
                            )
                            st.session_state.mode = "menu"
                    else:
                        bot_reply = "Okay! You can type another recipe name anytime. 🍳"
                        st.session_state.messages.append(
                            {"role": "assistant", "content": bot_reply}
                        )
                        st.session_state.mode = "menu"

                # Default small talk / smart reply
                else:
                    if "ingredient" in user_lower:
                        results = search_recipe_by_title(
                            user_lower.replace("ingredients for", "")
                            .replace("ingredients", "")
                            .strip(),
                            top_k=5,
                        )
                        if results:
                            first_recipe = results[0]
                            st.session_state.last_results = results
                            st.session_state.last_index = 0
                            st.session_state.mode_type = "ingredient"
                            st.session_state.mode = "ask_another"
                            bot_reply = f"Here are the ingredients for **{first_recipe['title']}**:\n\n{format_list(first_recipe['ingredients'])}"
                            st.session_state.messages.append(
                                {"role": "assistant", "content": bot_reply}
                            )
                            st.session_state.messages.append(
                                {
                                    "role": "assistant",
                                    "content": "Do you want to see another recipe? (yes/no)",
                                }
                            )
                        else:
                            bot_reply = "Sorry, I couldn’t find the ingredients for that recipe. 😔"
                            st.session_state.mode = "waiting_ingredient"

                    elif "step" in user_lower or "direction" in user_lower:
                        results = search_recipe_by_title(
                            user_lower.replace("steps for", "")
                            .replace("directions for", "")
                            .replace("steps", "")
                            .replace("directions", "")
                            .strip(),
                            top_k=5,
                        )
                        if results:
                            first_recipe = results[0]
                            st.session_state.last_results = results
                            st.session_state.last_index = 0
                            st.session_state.mode_type = "step"
                            st.session_state.mode = "ask_another"
                            bot_reply = f"Here are the cooking steps for **{first_recipe['title']}**:\n\n{format_steps(first_recipe['directions'])}"
                            st.session_state.messages.append(
                                {"role": "assistant", "content": bot_reply}
                            )
                            st.session_state.messages.append(
                                {
                                    "role": "assistant",
                                    "content": "Do you want to see another recipe? (yes/no)",
                                }
                            )
                        else:
                            bot_reply = "Sorry, I couldn’t find the cooking steps for that recipe. 😔"
                            st.session_state.mode = "waiting_direction"
                    else:
                        # Small talk
                        if "hello" in user_lower or "hi" in user_lower or "hai" in user_lower or "hey" in user_lower:
                            bot_reply = "Hi there! 😊 How can I help you with your cooking today?"
                        elif "thank" in user_lower:
                            bot_reply = "You're welcome! Let me know if you need anything else. 🍀"
                        elif "how to" in user_lower or "make" in user_lower:
                            results = search_recipe_by_title(user_lower.replace("how to", "").replace("make", "").strip(), top_k=5)
                            if results:
                                first_recipe = results[0]
                                st.session_state.last_results = results
                                st.session_state.last_index = 0
                                bot_reply = f"To make **{first_recipe['title']}**, you have to prepare these ingredients:\n\n{format_list(first_recipe['ingredients'])}\n\nAfter preparing the ingredients, you can follow these cooking steps:\n\n{format_steps(first_recipe['directions'])}"
                                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
                                st.session_state.messages.append({"role": "assistant", "content": "Do you want to see another recipe? (yes/no)"})
                                st.session_state.mode = "ask_another"
                            else:
                                bot_reply = "Sorry, I couldn’t find that recipe. 😔"
                                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
                                st.session_state.mode = "waiting_direction"
                        elif "ingredient" in user_lower:
                            results = search_recipe_by_title(user_lower.replace("ingredients for", "").replace("ingredients", "").strip(), top_k=5)
                            if results:
                                first_recipe = results[0]
                                st.session_state.last_results = results
                                st.session_state.last_index = 0
                                bot_reply = f"Here are the ingredients for **{first_recipe['title']}**:\n\n{format_list(first_recipe['ingredients'])}"
                                st.session_state.messages.append({"role": "assistant", "content": "Do you want to see another recipe? (yes/no)"})
                                st.session_state.mode = "ask_another"
                            else:
                                bot_reply = "Sorry, I couldn’t find the ingredients for that recipe. 😔"
                                st.session_state.mode = "waiting_ingredient"
                                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
                        elif "step" in user_lower or "direction" in user_lower:
                            results = search_recipe_by_title(user_lower.replace("steps for", "").replace("directions for", "").replace("steps", "").replace("directions", "").strip(), top_k=5)
                            if results:
                                first_recipe = results[0]
                                st.session_state.last_results = results
                                st.session_state.last_index = 0
                                bot_reply = f"Here are the directions for **{first_recipe['title']}**:\n\n{format_steps(first_recipe['directions'])}"
                                st.session_state.messages.append({"role": "assistant", "content": "Do you want to see another recipe? (yes/no)"})
                                st.session_state.mode = "ask_another"
                            else:
                                bot_reply = "Sorry, I couldn’t find the cooking steps for that recipe. 😔"
                                st.session_state.mode = "waiting_direction"
                                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
                        else:
                            bot_reply = "I can help you with ingredients or cooking steps. Just mention the recipe name. 🍳"
                        st.session_state.messages.append(
                            {"role": "assistant", "content": bot_reply}
                        )
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    auto_scroll()