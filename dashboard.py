# Dashbaord; run with streamlit run dashbaord.py
# chatbot parts adapted from https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps

import streamlit as st
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl 

from agent import run, _schema
from load_chroma import load_chroma_db, rag, add_pos_example
from dotenv import load_dotenv
from graph import generate_graph
from graph import fix_graph


# styling for graphs so it matches the overall design
plt.style.use("dark_background")
mpl.rcParams.update({
    "figure.facecolor": "#0E1117",
    "axes.facecolor": "#0E1117",
    "savefig.facecolor": "#0E1117",
})


# get secrets and rag
load_dotenv()
db = load_chroma_db("./chroma_db", "business_questions", "snowflake-arctic-embed2")
results = []

# login + styling 
# unsafe_allow_html=True tags are needed to overwrite streamlit design and add custom html
st.markdown('<style>' + open('style.css').read() + '</style>', unsafe_allow_html=True)
def login_screen():
    st.title("REGIOCAST Conversational Analytics Agent")
    st.header("Bitte melden Sie sich an, um auf das Tool zugreifen zu können")
    st.button("Mit Microsoft anmelden", on_click=st.login)

# set icons for chat interface
USER_AVATAR = ":material/person:"
ASSISTANT_AVATAR = ":material/smart_toy:"

def avatar_for(role):
    return USER_AVATAR if role == "user" else ASSISTANT_AVATAR

def run_agent(prompt):
    return run(prompt)

# messages while agent runs in the background
def show_running_status(future, placeholder):
    messages = itertools.cycle([
        "SQL wird generiert...",
        "Daten werden abgerufen...",
        "Daten werden verarbeitet...",
        "Antwort wird generiert...",
        "Die Antwort dauert etwas länger, aber ich arbeite daran ...",
        "Keine Sorge, ich arbeite noch an der Antwort...",
        "Bitte entschuldigen Sie die Verzögerung...",
    ])
    while not future.done():
        placeholder.markdown(
                f"""
                <div class="custom-toast-wrapper">
                    <div class="custom-toast">
                        {next(messages)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        time.sleep(2.5) # pause between messages
    placeholder.empty()


# rating; positive rating gets saved in chroma db
def handle_rating(i, rating_value):
    st.session_state.messages[i]["rating"] = rating_value
    if rating_value == "up":
        message = st.session_state.messages[i]
        question = message.get("prompt")
        sql = message.get("sql")
        if question and sql:
            add_pos_example(db, question, sql)

def save_rating(i):
    message = st.session_state.messages[i]
    rating = message.get("rating")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.button(
            "👍",
            key=f"thumbs_up_{i}",
            disabled=rating is not None,
            on_click=handle_rating,
            args=(i, "up"),
        )
    with col2:
        st.button(
            "👎",
            key=f"thumbs_down_{i}",
            disabled=rating is not None,
            on_click=handle_rating,
            args=(i, "down"),
        )
    if rating == "up":
        st.markdown(
            "<div style='color: gray; font-size: 0.9rem;'>Danke für Ihr Feedback. Ihre Antwort: 👍</div>",
            unsafe_allow_html=True
        )
    elif rating == "down":
        st.markdown(
            "<div style='color: gray; font-size: 0.9rem;'>Danke für Ihr Feedback. Ihre Antwort: 👎</div>",
            unsafe_allow_html=True
        )

# interface display; only show dashbaord after login
if not st.user.is_logged_in:
    login_screen()
    
else:
    # sidebar with tips section
    with st.sidebar:
        st.title("REGIOCAST Conversational Analytics Agent")
        st.caption(f"Willkommen {st.user.name}")
        st.button("Abmelden", on_click=st.logout)
        with st.expander("💡 Tipps", expanded=True):
            st.markdown("""
            - Wenn nach Hörerzahlen gefragt wird, zählt der Bot einzigartige ListenerIDs, für alle Sessions sollte explizit nach Sessions gefragt werden
            - Für genauere Ergebnisse sollten Zeiträume genannt werden
            - Die Tabelle enthält die Ground Truth, weitere Informationen sind KI generiert
            """)
    with st.container(key="top_header"):
        left, right = st.columns([3, 1])
        with left:
            st.title("REGIOCAST Conversational Analytics Agent")
            st.caption(f"Willkommen {st.user.name}")
    

    if "messages" not in st.session_state:
        st.session_state.messages = []
    # display message history with all expand menues
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"], avatar=avatar_for(message["role"])):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                if message.get("sql"):
                    with st.expander("SQL anzeigen"):
                        st.code(message["sql"], language="sql")
                if message["table"] is not None:
                    with st.expander("Ganze Tabelle anzeigen"):
                        st.dataframe(message["table"])
                if message.get("fig"):
                    with st.expander("Graph anzeigen"):
                        st.pyplot(message["fig"])
                with st.expander("Helfen sie uns, besser zu werden und bewerten Sie die erhaltene Antwort"):
                    save_rating(i)

    # input box for the user
    if prompt := st.chat_input("Was möchten Sie zu den Session Logs wissen?"):
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)

        # get answer and handle errors
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            status = True
            toast_placeholder = st.empty()
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    # execute agent
                    future = executor.submit(run, prompt)
                    show_running_status(future, toast_placeholder)
                    result = future.result()
                answer = result["answer"]
                sql = result["sql"]
                if "Leider konnte keine" in answer:
                    status = False
            except Exception as e:
                status = False
                toast_placeholder.empty()
                answer = "Etwas ist schief gelaufen. Bitte probieren Sie es noch einmal"
                print(e)

            # variables for expand menues
            fig = None
            code_fig = None
            table = None

            # table display
            if result.get("columns") and result.get("rows"):
                try:
                    table = pd.DataFrame(result["rows"], columns=result["columns"])
                except:
                    table = None

            # graph generation with columns and rows from result
            if result.get("columns") and result.get("rows"):
                try: 
                    fig, code_fig = generate_graph(
                        result["columns"],
                        result["rows"],
                        prompt
                    )
                except Exception as e:
                    fig = None
                    try: 
                        fig = fix_graph(
                            error=e, 
                            columns= result["columns"], 
                            rows= result["rows"], 
                            user_input= prompt, 
                            old_code=code_fig)
                        print(f"Graph generation failed: {e}")
                    except Exception as e2:
                        print(f"Graph fix failed: {e2}")
                        fig = None
            
            # save answer with features (sql, graph, table)
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "rating": None,
                "sql": sql,
                "table": table,
                "fig": fig,
                "prompt": prompt,
                "saved_to_chroma": False,
            })

            # display answer with all features that were returned, do not display features that are  None
            assistant_index = len(st.session_state.messages) - 1
            st.write(answer)
            if sql:
                with st.expander("SQL anzeigen"):
                    st.code(sql, language="sql")
            if table is not None:
                with st.expander("Ganze Tabelle anzeigen"):
                    st.dataframe(table)
            if fig is not None:
                with st.expander("Graph anzeigen"):
                    st.pyplot(fig)
            with st.expander("Helfen sie uns, besser zu werden und bewerten Sie die erhaltene Antwort"):
                save_rating(assistant_index)