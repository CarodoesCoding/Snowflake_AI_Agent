import pandas as pd
from llm import generate_graph_code
from llm import fix_graph_code


def generate_graph(columns: list, rows: list, user_input: str):
    if not columns or not rows:
        return None

    df = pd.DataFrame(rows, columns=columns)

    code = generate_graph_code(
        question=user_input,
        columns=columns,
        rows=rows
    )

    namespace = {}
    exec(code, namespace)

    fig = namespace["generate_plot"](df)
    return fig, code

def fix_graph(error: str, columns: list, rows: list, user_input: str, old_code: str,):
    if not columns or not rows:
        return None

    df = pd.DataFrame(rows, columns=columns)

    new_code = fix_graph_code(
        question= user_input, 
        columns= columns, 
        rows= rows, 
        code= old_code,
        error=error)

    namespace = {}
    exec(new_code, namespace)

    fig = namespace["generate_plot"](df)
    return fig