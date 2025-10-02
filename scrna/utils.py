from graphviz import Digraph
from sklearn.pipeline import Pipeline
import polars
import gtfparse
import polars
import pandas as pd
import json
import numpy as np
from graphviz import Digraph
from sklearn.pipeline import Pipeline
import json
from compact_json import Formatter
import html
from sklearn.base import clone

def concat_pipelines(pipes):
    _pipes = [clone(pipe) for pipe in pipes]
    for pipe in _pipes[1:]:
        _pipes[0].steps.extend(pipe.steps)
    return _pipes[0]


def save_ensembl_annotations(file, folder):

    df = gtfparse.read_gtf(file)
    filename = file.rsplit('/', 1)[1].rsplit('.',1)[0]
    s = df.filter((polars.col('feature') == 'gene')).select(['gene_id', 'gene_name'])
    print(folder+'/'+filename+'.csv')
    s.select(polars.all().str.to_lowercase()).write_csv(folder+'/'+filename+'.csv')


def save_object(obj, filename):
    import pickle
    with open(filename, 'wb') as outp: 
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def get_cell_cycle_genes(filename):
    
    cell_cycle_genes = pd.read_csv(filename, sep = '\t')
    
    s_genes = cell_cycle_genes['S'].dropna().str.lower().tolist()
    g2m_genes = cell_cycle_genes['G2.M'].dropna().str.lower().tolist()
    

    return s_genes, g2m_genes


class NumpyEncoder(json.JSONEncoder):
    """
    JSON encoder that:
      - converts numpy scalars to Python scalars
      - converts numpy arrays to lists
      - converts any other object to a short string with its class name
    """
    def default(self, obj):
        # NumPy ints
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16,
                            np.int32, np.int64, np.uint8, np.uint16,
                            np.uint32, np.uint64)):
            return int(obj)
        # NumPy floats
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        # NumPy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Generic objects → "<ClassName>"
        try:
            name = getattr(obj, "__name__", obj.__class__.__name__)
            return f"<{name}>"
        except Exception:
            return json.JSONEncoder.default(self, obj)


def pipeline_to_json(pipe, json_path=None):
    """
    Build and return a JSON string describing a sklearn Pipeline.
    Steps set to None or 'passthrough' are marked; others include get_params(deep=False).
    If json_path is provided, writes the same JSON string using NumpyEncoder.
    """
    if not isinstance(pipe, Pipeline):
        raise ValueError("pipe must be an sklearn.pipeline.Pipeline")

    steps = []
    for name, est in pipe.steps:
        if est is None:
            steps.append({"name": name, "kind": "none", "class_name": None, "params": None})
            continue
        if isinstance(est, str) and est == "passthrough":
            steps.append({"name": name, "kind": "passthrough", "class_name": None, "params": None})
            continue  # <- fixed stray token
        steps.append({
            "name": name,
            "kind": "estimator",
            "class_name": f"{est.__class__.__module__}.{est.__class__.__name__}",
            "params": est.get_params(deep=False)
        })

    obj = {"steps": steps}
    json_str = json.dumps(obj, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    if json_path is not None:
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_str)

    return json_str

def pipeline_to_svg(pipeline_or_json, svg_path="pipeline", graph_kwargs=None,
                    show_params=True):
    """
    Vertical SVG of one sklearn Pipeline (or its JSON), or a CHAIN of pipelines.
    - Accepts a Pipeline, JSON string/path, or a list/tuple of any of those.
    - Skips steps set to None / 'passthrough'.
    - Params pretty-printed with compact_json (dicts indented, lists inline).
    - Float rule: <=2 decimals -> plain (e.g. 0.75, 1.0); else -> scientific with 2 decimals (e.g. 1.26e-02).
    - Layout: rounded outer box per node; centered estimator name; left-aligned params.
    """
    import json, html
    from graphviz import Digraph
    from sklearn.pipeline import Pipeline
    from compact_json import Formatter

    formatter = Formatter()
    formatter.indent_spaces = 2
    formatter.max_inline_complexity = 2
    formatter.max_inline_length = 70

    def _load_as_json(obj):
        if isinstance(obj, Pipeline):
            return json.loads(pipeline_to_json(obj))
        if isinstance(obj, str):
            try:
                return json.loads(obj)   # JSON string
            except json.JSONDecodeError:
                with open(obj, "r", encoding="utf-8") as f:
                    return json.load(f)  # JSON file
        raise ValueError("Each item must be a Pipeline, JSON string, or JSON file path.")

    def _format_value(v):
        if isinstance(v, float):
            x = float(v)
            if abs(x - round(x, 2)) < 1e-12:
                s = f"{round(x, 2):.2f}".rstrip("0")
                if s.endswith("."):
                    s += "0"
                return s
            return f"{x:.2e}"
        if isinstance(v, list):
            return [_format_value(x) for x in v]
        if isinstance(v, dict):
            return {k: _format_value(val) for k, val in v.items()}
        return v

    def _format_params(params):
        if not show_params or not isinstance(params, dict) or not params:
            return []
        formatted = {
            k: _format_value(v) for k, v in params.items()
            if (v is not None and v != []) and k != 'plot'
        }
        if not formatted:
            return []
        txt = formatter.serialize(formatted)
        return txt.splitlines()

    def _visible_steps(data):
        # returns [(step_name, class_short, params_dict_shown)]
        steps = []
        for s in data.get("steps", []):
            if s.get("kind") in ("none", "passthrough"):
                continue
            name = s.get("name", "")
            cls = (s.get("class_name") or "Estimator").split(".")[-1]
            params = s.get("params", {}) or {}
            steps.append((name, cls, params))
        return steps

    # Normalize input to a list of pipeline JSON dicts
    if isinstance(pipeline_or_json, (list, tuple)):
        pipelines = [ _load_as_json(obj) for obj in pipeline_or_json ]
    else:
        pipelines = [ _load_as_json(pipeline_or_json) ]

    # Build visible steps per pipeline (in order)
    chains = [ _visible_steps(d) for d in pipelines ]

    dot = Digraph(format="svg", **(graph_kwargs or {}))
    dot.attr("graph", rankdir="TB", nodesep="0.15", ranksep="0.25")
    dot.attr("node", shape="box", style="rounded", fontname="Nimbus Sans")

    # Render nodes and connect within/between pipelines
    prev_last_node_id = None
    for pi, steps in enumerate(chains):
        visible_ids = []
        for name, cls, params in steps:
            node_id = f"p{pi}_{name}"  # unique internal id per pipeline index
            params_lines = _format_params(params)

            # HTML label: centered header, left-aligned params
            html_label = ['<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="2">']
            html_label.append(f'<TR><TD ALIGN="CENTER"><B>{html.escape(cls)}</B></TD></TR>')
            for ln in params_lines:
                html_label.append(f'<TR><TD ALIGN="LEFT">{html.escape(ln)}</TD></TR>')
            html_label.append("</TABLE>>")

            dot.node(node_id, label="".join(html_label))
            visible_ids.append(node_id)

        # connect within this pipeline
        for i in range(len(visible_ids) - 1):
            dot.edge(visible_ids[i], visible_ids[i + 1], arrowhead="none")

        # connect chain: previous pipeline tail -> this pipeline head
        if prev_last_node_id and visible_ids:
            dot.edge(prev_last_node_id, visible_ids[0], arrowhead="none")

        if visible_ids:
            prev_last_node_id = visible_ids[-1]
    return dot.render(filename=svg_path, cleanup=True)
    


def _get_visible_steps(pipeline_json):
    """
    Iterate the visible (non-passthrough/none) steps of a pipeline JSON.

    Parameters
    ----------
    pipeline_json : dict
        A dict with key "steps" as produced by `pipeline_to_json`.

    Yields
    ------
    (step_name, class_name, params_dict)
        - step_name  : str
        - class_name : str (tail of 'module.ClassName')
        - params_dict: dict of shallow params (may be empty)
    """
    for step in pipeline_json.get("steps", []):
        if step.get("kind") in ("none", "passthrough"):
            continue
        step_name = step.get("name", "")
        class_name = (step.get("class_name") or "Estimator").split(".")[-1]
        params = step.get("params") or {}
        yield (step_name, class_name, params)


def _get_child_specs(spec_node):
    """
    Iterate only the *child* sub-specs of a node, skipping control keys.

    Parameters
    ----------
    spec_node : dict
        A node from the nested spec tree.

    Yields
    ------
    (attach_key, child_spec)
        - attach_key : str, the key on the parent under which the child is attached
                       (e.g., 'step_name' or 'step_name:1')
        - child_spec : dict, must contain 'trunk' and 'name_trunk'
    """
    for key, val in spec_node.items():
        if key in ("trunk", "name_trunk", "links"):
            continue
        if isinstance(val, dict) and ("trunk" in val) and ("name_trunk" in val):
            yield key, val


def _collect_columns(spec_root):
    """
    Flatten the nested spec into a linear list of columns (pipelines) for drawing.

    Parameters
    ----------
    spec_root : dict
        Root of the nested spec.

    Returns
    -------
    list[tuple[str, dict, dict]]
        A list of (pipeline_name, pipeline_json_dict, spec_node_for_pipeline)
        in DFS order. The DFS is iterative and cycle-protected.

    Raises
    ------
    ValueError
        On cycle detection or absurdly deep nesting.
    """
    columns, stack = [], [(spec_root, 0)]
    seen, MAX_DEPTH = set(), 1000

    while stack:
        node, depth = stack.pop()
        node_hash = id(node)

        if node_hash in seen:
            raise ValueError("Cycle detected in spec; aborting.")
        seen.add(node_hash)
        if depth > MAX_DEPTH:
            raise ValueError("Spec nesting too deep; aborting.")

        pipeline_name = node["name_trunk"]
        pipeline_json = json.loads(pipeline_to_json(node["trunk"]))
        columns.append((pipeline_name, pipeline_json, node))

        for _, child in _get_child_specs(node):
            stack.append((child, depth + 1))
    return columns


def _get_html_label(title, lines, text_color=None):
    """
    Construct an HTML-like Graphviz label for a node (title + left-aligned lines).

    Parameters
    ----------
    title : str
        The estimator class name to display in bold at the top of the node.
    lines : list[str]
        Preformatted parameter lines (already broken by the caller).
    text_color : str or None
        Optional color (name or #hex). If given, colors both title and lines.

    Returns
    -------
    str
        A label string suitable for Graphviz HTML-like labels.
    """
    open_font  = f'<FONT COLOR="{text_color}">' if text_color else ''
    close_font = '</FONT>' if text_color else ''
    parts = ['<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="2">']
    parts.append(f'<TR><TD ALIGN="CENTER"><B>{open_font}{html.escape(title)}{close_font}</B></TD></TR>')
    for ln in lines:
        parts.append(f'<TR><TD ALIGN="LEFT">{open_font}{html.escape(ln)}{close_font}</TD></TR>')
    parts.append("</TABLE>>")
    return "".join(parts)

def _format_value(v):
    if isinstance(v, float):
        x = float(v)
        if abs(x - round(x, 2)) < 1e-12:
            s = f"{round(x, 2):.2f}".rstrip("0")
            if s.endswith("."):
                s += "0"
            return s
        return f"{x:.2e}"
    if isinstance(v, list):
        return [_format_value(x) for x in v]
    if isinstance(v, dict):
        return {k: _format_value(val) for k, val in v.items()}
    return v

def experiment_repr(spec, svg_path=None, show_params=True, graph_kwargs=None, align_last=False):
    """
    Render nested pipelines. If align_last=True, skip all top alignments and
    align each child pipeline's LAST node with the LAST node of its parent
    (or with the LAST node of the pipe referenced in child['align'] = 'pipe:step').
    """
    import re
    from graphviz import Digraph
    from compact_json import Formatter

    # Pretty-printer for parameter blocks
    formatter = Formatter()
    formatter.indent_spaces = 2
    formatter.max_inline_complexity = 2
    formatter.max_inline_length = 70

    # 0) Preprocess and draw column nodes
    columns = _collect_columns(spec)

    dot = Digraph(format="svg", **(graph_kwargs or {}))
    dot.attr("graph", rankdir="TB", nodesep="0.25", ranksep="0.45", splines="ortho")
    dot.attr("node", shape="box", style="rounded", fontname="Nimbus Sans")

    # Registries
    node_ids = {}              # (pipeline_name, step_name) -> node_id
    first_step_id = {}         # pipeline_name -> first node_id in that column
    next_step_after = {}       # (pipeline_name, step_name) -> next step_name or None
    class_name_by_id = {}      # node_id -> estimator class name
    param_lines_by_id = {}     # node_id -> list[str] parameter lines
    last_step_id_by_pipe = {}  # pipeline_name -> last node_id (for caption & align_last)

    for pipeline_name, pipeline_json, _node_spec in columns:
        prev_node_id = None
        prev_step_name = None

        for idx, (step_name, class_name, params) in enumerate(_get_visible_steps(pipeline_json)):
            node_id = f"{pipeline_name}__{step_name}"
            node_ids[(pipeline_name, step_name)] = node_id
            if idx == 0:
                first_step_id[pipeline_name] = node_id

            if show_params and params:
                filtered = {k: v for k, v in params.items() if v is not None and v != [] and k != "plot"}
                if filtered:
                    # apply your float formatting recursively
                    formatted = {k: _format_value(v) for k, v in filtered.items()}
                    txt = formatter.serialize(formatted)
                    # remove quotes around numeric-like tokens (produced by _format_value)
                    txt = re.sub(r'"(-?(?:\d+(?:\.\d+)?)(?:[eE][+\-]?\d+)?)"', r'\1', txt)
                    param_lines = txt.splitlines()
                else:
                    param_lines = []
            else:
                param_lines = []

            class_name_by_id[node_id] = class_name
            param_lines_by_id[node_id] = param_lines

            dot.node(node_id, label=_get_html_label(class_name, param_lines), group=pipeline_name)

            if prev_node_id:
                dot.edge(prev_node_id, node_id, arrowhead="none", weight="50")
                next_step_after[(pipeline_name, prev_step_name)] = step_name

            prev_node_id, prev_step_name = node_id, step_name

        # mark terminal next-step
        if prev_step_name is not None and (pipeline_name, prev_step_name) not in next_step_after:
            next_step_after[(pipeline_name, prev_step_name)] = None
        if prev_node_id:
            last_step_id_by_pipe[pipeline_name] = prev_node_id

    # Caption under each pipeline column
    for pipeline_name, last_id in last_step_id_by_pipe.items():
        caption_id = f"{pipeline_name}__CAPTION"
        dot.node(caption_id, label=f"<<B>{html.escape(pipeline_name)}</B>>",
                 shape="plaintext", group=pipeline_name)
        dot.edge(last_id, caption_id, style="invis")

    # 1) ALIGN PASS
    #    - If align_last=False: align TOP as before (next step after branching, or explicit 'align')
    #    - If align_last=True:  skip TOP alignment entirely
    #    - If align_last=True:  also align LAST(child) ↔ LAST(parent or align pipe)
    stack, seen = [(spec, 0)], set()
    while stack:
        node, depth = stack.pop()
        if id(node) in seen:     continue
        if depth > 1000:         raise ValueError("Spec nesting too deep during alignment.")
        seen.add(id(node))

        parent_pipe = node["name_trunk"]
        for attach_key, child in _get_child_specs(node):
            parent_step = attach_key.split(":", 1)[0]

            # Determine anchor pipe for LAST↔LAST when align_last=True
            if child.get("align") and ":" in child["align"]:
                anchor_pipe_for_last = child["align"].split(":", 1)[0]
            else:
                anchor_pipe_for_last = parent_pipe

            # TOP alignment only if align_last is False
            if not align_last:
                if child.get("align"):
                    ref = child["align"]
                    if ":" in ref:
                        anchor_pipe_top, anchor_step_top = ref.split(":", 1)
                    else:
                        anchor_pipe_top, anchor_step_top = parent_pipe, ref
                    top_anchor_id = node_ids.get((anchor_pipe_top, anchor_step_top))
                else:
                    nxt = next_step_after.get((parent_pipe, parent_step))
                    target_step = nxt if nxt is not None else parent_step
                    top_anchor_id = node_ids.get((parent_pipe, target_step))

                child_first_id = first_step_id.get(child["name_trunk"])
                if top_anchor_id and child_first_id:
                    with dot.subgraph() as sg:
                        sg.attr(rank="same")
                        sg.node(top_anchor_id); sg.node(child_first_id)

            # LAST↔LAST alignment if requested
            if align_last:
                parent_last_id = last_step_id_by_pipe.get(anchor_pipe_for_last)
                child_last_id  = last_step_id_by_pipe.get(child["name_trunk"])
                if parent_last_id and child_last_id:
                    with dot.subgraph() as sg2:
                        sg2.attr(rank="same")
                        sg2.node(parent_last_id); sg2.node(child_last_id)

            stack.append((child, depth + 1))

    # 2) WIRE PASS: connect parent step node → child first node (no arrowhead)
    stack, seen = [(spec, 0)], set()
    while stack:
        node, depth = stack.pop()
        if id(node) in seen:     continue
        if depth > 1000:         raise ValueError("Spec nesting too deep during wiring.")
        seen.add(id(node))

        parent_pipe = node["name_trunk"]
        for attach_key, child in _get_child_specs(node):
            parent_step = attach_key.split(":", 1)[0]
            parent_node_id = node_ids.get((parent_pipe, parent_step))
            child_first_id = first_step_id.get(child["name_trunk"])
            if parent_node_id and child_first_id:
                dot.edge(parent_node_id, child_first_id, arrowhead="none", constraint="false")
            stack.append((child, depth + 1))

    # 3) LINK PASS: red cross-links; color destination node text + border red
    red_destination_nodes = set()
    stack, seen = [(spec, 0)], set()
    while stack:
        node, depth = stack.pop()
        if id(node) in seen:     continue
        if depth > 1000:         raise ValueError("Spec nesting too deep during links.")
        seen.add(id(node))

        for _, child in _get_child_specs(node):
            links = child.get("links")
            if links:
                if isinstance(links, (str, bytes)):
                    links = [links]
                for expr in links:
                    if "->" not in expr:
                        continue
                    src_ref, dst_ref = [t.strip() for t in expr.split("->", 1)]

                    # resolve source
                    if ":" in src_ref:
                        src_pipe, src_step = src_ref.split(":", 1)
                    else:
                        src_pipe, src_step = child["name_trunk"], src_ref
                    src_id = node_ids.get((src_pipe, src_step))

                    # resolve destination
                    if ":" in dst_ref:
                        dst_pipe, dst_step = dst_ref.split(":", 1)
                    else:
                        dst_pipe, dst_step = child["name_trunk"], dst_ref
                    dst_id = node_ids.get((dst_pipe, dst_step))

                    if src_id and dst_id:
                        dot.edge(src_id, dst_id, color="red", penwidth="1.5",
                                 arrowhead="normal", constraint="false")
                        red_destination_nodes.add(dst_id)

            stack.append((child, depth + 1))

    for dst_id in red_destination_nodes:
        dot.node(dst_id,
                 label=_get_html_label(class_name_by_id[dst_id],
                                       param_lines_by_id[dst_id],
                                       text_color="red"),
                 color="red")

    if svg_path:
        dot.render(filename=svg_path, cleanup=True)
    return dot