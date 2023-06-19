import os
import subprocess


def _dot_var(v, verbose=False):
    name = "" if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ": "
        name += str(v.shape) + " " + str(v.dtype)

    dot_var = f"{id(v)} [label=\"{name}\", color=orange, style=filled]\n"
    return dot_var


def _dot_fun(f):
    txt = f"{id(f)} [label=\"{ f.__class__.__name__}\", color=lightblue, style=filled, shape=box]\n"

    for x in f.inputs:
        txt += f"{id(x)} -> {id(f)}\n"
    for y in f.outputs:
        txt += f"{id(f)} -> {id(y())}\n"

    return txt


def get_dot_graph(output, verbose=False):
    dot_txt = ""
    funcs = []
    seen_set = set()

    def add_func(func):
        if func not in seen_set:
            seen_set.add(func)
            funcs.append(func)
    add_func(output.creator)

    output.name = "output"
    dot_txt += _dot_var(output, verbose=verbose)

    while funcs:
        func = funcs.pop()
        dot_txt += _dot_fun(func)

        for x in func.inputs:
            dot_txt += _dot_var(x, verbose=verbose)
            if x.creator is not None:
                add_func(x.creator)

    return f"digraph g {{\n{dot_txt} }}"


def plot_dot_graph(output, verbose=True, out_path="./"):
    dot_graph = get_dot_graph(output, verbose)

    dot_path = os.path.join(out_path,"tmp_dot.dot")
    with open(dot_path, "w") as fp:
        fp.write(dot_graph)

    png_path = os.path.join(out_path,"tmp_dot.png")
    cmd = f"dot {dot_path} -T png -o {png_path}"
    subprocess.run(cmd, shell=True)
