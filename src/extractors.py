import re

from pylatexenc.latex2text import LatexNodes2Text
from pylatexenc.latexwalker import LatexEnvironmentNode, LatexWalker


def extract_code(text, include_language=False):
    code_pattern = r"```([\w\s]*)\n(.*?)\n```"
    code_chunks = re.findall(code_pattern, text, re.DOTALL)
    if include_language:
        return code_chunks
    else:
        return [code for _, code in code_chunks]


def remove_code(text):
    code_pattern = r"```([\w\s]*)\n(.*?)\n```"
    text = re.sub(code_pattern, "", text, re.DOTALL)
    return text


def extract_math(text, min_symbols=2):
    math_chunks = []

    w = LatexWalker(text)

    (nodelist, pos, len_) = w.get_latex_nodes()

    # """
    # Example:
    # >>> from pylatexenc.latexwalker import LatexWalker, LatexEnvironmentNode
    # >>> w = LatexWalker(r"""
    # ... \textbf{Hi there!} Here is \emph{a list}:
    # ... \begin{enumerate}[label=(i)]
    # ... \item One
    # ... \item Two
    # ... \end{enumerate}
    # ... and $x$ is a variable.
    # ... """)
    # >>> (nodelist, pos, len_) = w.get_latex_nodes(pos=0)
    # >>> nodelist[0]
    # LatexCharsNode(pos=0, len=1, chars='\n')
    # >>> nodelist[1]
    # LatexMacroNode(pos=1, len=18, macroname='textbf',
    # nodeargd=ParsedMacroArgs(argnlist=[LatexGroupNode(pos=8, len=11,
    # nodelist=[LatexCharsNode(pos=9, len=9, chars='Hi there!')],
    # delimiters=('{', '}'))], argspec='{'), macro_post_space='')
    # >>> nodelist[5].isNodeType(LatexEnvironmentNode)
    # True
    # >>> nodelist[5].environmentname
    # 'enumerate'
    # >>> nodelist[5].nodeargd.argspec
    # '['
    # >>> nodelist[5].nodeargd.argnlist
    # [LatexGroupNode(pos=60, len=11, nodelist=[LatexCharsNode(pos=61, len=9,
    # chars='label=(i)')], delimiters=('[', ']'))]
    # >>> nodelist[7].latex_verbatim()
    # '$x$'
    # """

    # extract all possible formats of equations or math nodes
    for node in nodelist:
        if isinstance(node, LatexEnvironmentNode):
            if node.environmentname in [
                "equation",
                "align",
                "align*",
                "eqnarray",
                "eqnarray*",
                "displaymath",
                # TODO: not sure if there's a better way to do this
            ]:
                math = node.latex_verbatim().strip("$").strip()
                if math.count("\\") >= min_symbols:
                    math_chunks.append(math)
        else:
            math = node.latex_verbatim().strip("$").strip()
            if math.count("\\") >= min_symbols:
                math_chunks.append(math)

    math_chunks = [math.strip("$").strip() for math in math_chunks]
    math_chunks = [math for math in math_chunks if math]

    # remove duplicates but keep order
    math_chunks = list(dict.fromkeys(math_chunks))

    # math_chunks = [math for math in math_chunks if math.count("\\") >= min_commands]

    return math_chunks


def remove_latex(text):
    # math_patterns = [
    #     # inline math
    #     r"\$.*?\$",
    #     # display math
    #     r"\$\$.*?\$\$",
    # ]

    text = LatexNodes2Text().latex_to_text(text)

    return text
