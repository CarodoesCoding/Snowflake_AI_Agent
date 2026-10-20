import os
import re
import json
import pandas as pd
from datetime import date
from groq import Groq
from dotenv import load_dotenv
from load_chroma import load_chroma_db, rag

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

_vectordb = load_chroma_db(
    "./chroma_db", "business_questions", "snowflake-arctic-embed2"
)

_cost_tracker = {
    "input_tokens": 0,
    "output_tokens": 0,
    "total_cost": 0.0,
    "calls": 0,
    "calls_detail": [],
}


def _clean_sql(sql: str) -> str:
    """entfernt Markdown-Formatierung und überflüssige Zeichen aus der generierten SQL"""
    sql = sql.strip()
    sql = re.sub(r"^```[\w]*\n?", "", sql)
    sql = re.sub(r"\n?```$", "", sql)
    if sql.lower().startswith("sql"):
        sql = sql[3:]
    return sql.strip()


def _get_few_shot(question: str) -> str:
    """holt ähnliche Fragen und deren SQL aus der Vektordatenbank, um sie als Beispiele für die SQL-Generierung zu verwenden"""
    examples = rag(question, _vectordb, k=3)
    return "\n\n".join(
        [f"Question: {ex['question']}\nSQL: {ex['sql']}" for ex in examples]
    )


def _track_usage(usage, label: str):
    """trackt die Token-Nutzung und Kosten für die einzelnen Schritte des Agenten"""
    input_tokens, output_tokens = usage.prompt_tokens, usage.completion_tokens
    cost = (input_tokens * 0.11 + output_tokens * 0.34) / 1_000_000
    _cost_tracker["input_tokens"] += input_tokens
    _cost_tracker["output_tokens"] += output_tokens
    _cost_tracker["total_cost"] += cost
    _cost_tracker["calls"] += 1
    _cost_tracker["calls_detail"].append(
        {
            "function": label,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
        }
    )
    print(
        f"Token usage for {label}: {input_tokens} in / {output_tokens} out -> ${cost:.6f}"
    )


def reset_cost_tracker():
    """setzt den Kosten-Tracker zurück, wird vor jeder neuen Frage ausgeführt"""
    global _cost_tracker
    _cost_tracker = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_cost": 0.0,
        "calls": 0,
        "calls_detail": [],
    }


def get_cost_summary() -> dict:
    """returns cost summary"""
    return _cost_tracker.copy()


def generate_sql(question: str, schema: str) -> str:
    """LLM call für die Generierung von SQL basierend auf der Frage, dem Schema und ähnlichen Beispielen"""
    today = date.today().strftime("%Y-%m-%d")
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.1,
        max_tokens=500,
        messages=[
            {
                "role": "system",
                "content": f"""Du bist ein SQL-Experte für Snowflake.
                Heutiges Datum: {today}.
                Generiere ausschließlich gültiges Snowflake-SQL. Keine Erklärungen, keine Markdown-Formatierung, nur reines SQL.
                Verwende KEINE Umlaute in Spaltennamen oder Aliasen.
                Nutze die Tabelle SESSION_LOGS_CLEAN_V2.
                Schema: {schema}
                Beispiele: {_get_few_shot(question)}""",
            },
            {"role": "user", "content": question},
        ],
    )
    _track_usage(response.usage, "generate_sql")
    return _clean_sql(response.choices[0].message.content)


def fix_sql(schema: str, question: str, bad_sql: str, error: str) -> str:
    """LLM call für die Korrektur von fehlerhaftem SQL basierend auf der ursprünglichen Frage, dem fehlerhaften SQL, der Fehlermeldung und ähnlichen Beispielen"""
    today = date.today().strftime("%Y-%m-%d")
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.1,
        max_tokens=500,
        messages=[
            {
                "role": "system",
                "content": f"""Du bist ein SQL-Experte für Snowflake.
                Heutiges Datum: {today}.
                Korrigiere NUR den konkreten Fehler in der SQL-Abfrage. Verändere Logik oder Struktur NICHT.
                Gib nur reines SQL zurück, keine Erläuterungen.
                Schema: {schema}
                Beispiele: {_get_few_shot(question)}""",
            },
            {
                "role": "user",
                "content": f"""Original Frage: {question}
                Fehlerhafte SQL: {bad_sql}
                Snowflake Fehler: {error}
                Gib nur die korrigierte SQL zurück.""",
            },
        ],
    )
    _track_usage(response.usage, "fix_sql")
    return _clean_sql(response.choices[0].message.content)


def validate_response(columns: list, rows: list) -> dict:
    """validiert die SQL-Antwort auf mögliche Probleme wie leere Ergebnisse, falsche Datentypen oder ausschließlich NULL-Werte"""
    if not rows:
        return {
            "ok": False,
            "type": "empty",
            "issue": "Query erfolgreich aber keine Zeilen zurückgegeben",
        }
    if all(all(v is None for v in row) for row in rows):
        return {
            "ok": False,
            "type": "all_null",
            "issue": "Alle zurückgegebenen Werte sind NULL",
        }
    numeric_keywords = [
        "COUNT",
        "SUM",
        "AVG",
        "TOTAL",
        "ANZAHL",
        "SESSIONS",
        "LISTENERS",
        "DURATION",
        "MINUTES",
        "HOURS",
        "SECONDS",
    ]
    for i, col in enumerate(columns):
        if any(kw in numeric_keywords for kw in col.upper().split("_")):
            first_val = rows[0][i]
            if isinstance(first_val, str) and not first_val.replace(".", "").isdigit():
                return {
                    "ok": False,
                    "type": "wrong_type",
                    "issue": f"Spalte '{col}' sollte numerisch sein, enthält aber: '{first_val}'",
                }
    return {"ok": True, "type": "", "issue": ""}


def generate_nl_answer(question: str, columns: list, rows: list, schema: str) -> str:
    """LLM call für die Generierung einer natürlichen Sprache Antwort basierend auf der ursprünglichen Frage und den SQL-Ergebnissen"""
    results_text = " | ".join(columns) + "\n"
    results_text += "\n".join([" | ".join(map(str, row)) for row in rows[:20]])
    if len(rows) > 20:
        results_text += f"\n... (truncated, +{len(rows) - 20} more rows)"

    schema_context = (
        f"Verwende die Terminologie des Unternehmens. Schema:\n{schema}\n"
        if schema
        else ""
    )

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.3,
        max_tokens=300,
        messages=[
            {
                "role": "system",
                "content": f"Du bist ein hilfreicher Data-Analyst. Beantworte die Frage präzise basierend auf den Daten. Hebe wichtige Zahlen hervor. {schema_context}",
            },
            {
                "role": "user",
                "content": f"Frage: {question}\nQuery Ergebnis: {results_text}\nBeantworte in 1-3 Sätzen, kurz und präzise.",
            },
        ],
    )
    _track_usage(response.usage, "generate_nl_answer")
    return response.choices[0].message.content.strip()


def generate_graph_code(question: str, columns: list, rows: list) -> str:
    """Generiert Python-Code für einen Matplotlib-Graphen basierend auf SQL-Ergebnissen."""

    df_preview = pd.DataFrame(rows, columns=columns).head(20).to_string(index=False)

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.1,
        max_tokens=900,
        messages=[
            {
                "role": "system",
                "content": """
                    Du bist ein Python-Datenvisualisierungs-Assistent.
                    Gib ausschließlich valides JSON zurück.

                    Format:
                    {"code": "..."}

                    Der Code muss exakt diese Funktion definieren:

                    def generate_plot(df):
                        return fig

                    Regeln:
                    - Nutze nur matplotlib.
                    - Importiere matplotlib.pyplot as plt.
                    - Kein plt.show().
                    - Keine Dateien speichern.
                    - Kein fig.savefig().
                    - Gib immer ein Matplotlib Figure-Objekt zurück.
                    - Verwende ausschließlich Spalten, die im DataFrame vorhanden sind.
                    - Der Code darf keine Erklärungen, Markdown oder Kommentare außerhalb des JSON enthalten.
                    """,
            },
            {
                "role": "user",
                "content": f"""
                    User Question:
                    {question}

                    DataFrame columns:
                    {columns}

                    DataFrame preview:
                    {df_preview}
                    """,
            },
        ],
    )

    _track_usage(response.usage, "generate_graph_code")

    content = response.choices[0].message.content.strip()
    return json.loads(content)["code"]


def fix_graph_code(
    question: str, columns: list, rows: list, code: str, error: str
) -> str:
    """Correction Loop für den Matplotlib-Graphen."""

    df_preview = pd.DataFrame(rows, columns=columns).head(20).to_string(index=False)

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.1,
        max_tokens=900,
        messages=[
            {
                "role": "user",
                "content": """Du bist ein Python-Coding-Assistent.
                                Gib ausschließlich valides JSON zurück.

                                Format:
                                {"code": "..."}

                                Der Code muss exakt diese Funktion definieren:

                                def generate_plot(df):
                                    return fig

                                Regeln:
                                - Nutze nur matplotlib.
                                - Importiere matplotlib.pyplot as plt.
                                - Kein plt.show().
                                - Keine Dateien speichern.
                                - Kein fig.savefig().
                                - Gib immer ein Matplotlib Figure-Objekt zurück.
                                - Verwende ausschließlich Spalten, die im DataFrame vorhanden sind.
                                - Wenn Spalten aus der vorherigen Version fehlen oder unklar sind, wähle automatisch passende vorhandene Spalten aus df.
                                - Der Code darf keine Erklärungen, Markdown oder Kommentare außerhalb des JSON enthalten.""",
            },
            {
                "role": "user",
                "content": "User Frage:\n"
                + str(question)
                + "\n\nDataFrame columns:\n"
                + str(columns)
                + "\n\nDataFrame preview:\n"
                + str(df_preview)
                + "\n\nFehlerhafter Code:\n"
                + str(code)
                + "\n\nFehler:\n"
                + str(error),
            },
        ],
    )

    _track_usage(response.usage, "generate_graph_code")

    content = response.choices[0].message.content.strip()
    return json.loads(content)["code"]
