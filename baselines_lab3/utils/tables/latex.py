from typing import List, Tuple


class LatexTableWriter:
    def __init__(self, indent=4):
        self.lines = []
        self.indent = " "*indent
        self.current_indent = 0
        self.table_width = 0

    def begin_table(self, placement="ht"):
        self._line("\\begin{{table}}[{}]".format(placement))

    def end_table(self):
        self._line("\\end{table}")

    def begin_center(self):
        self._line("\\begin{center}")

    def end_center(self):
        self._line("\\end{center}")

    def begin_tabular(self, table_format):
        self._line("\\begin{{tabular}}{{{}}}".format(table_format))
        self.table_width = len(table_format)

    def add_toprule(self):
        self._line("\\toprule")

    def add_midrule(self):
        self._line("\\midrule")

    def add_bottomrule(self):
        self._line("\\bottomrule")

    def add_content(self, content):
        self._line(content)

    def add_row(self, row: List[str]):
        assert len(row) == self.table_width, "Row length must fit the table width!"
        self._line(" & ".join(row) + " \\\\")

    def add_spanned_header(self, header: List[Tuple[str, int, int]]):
        header_line = ""
        cmidrule_line = ""
        position = 0
        for i, (title, start, stop) in enumerate(header):
            header_line += " & "*(start - position - 1)
            header_line += "\\multicolumn{{{length}}}{{c}}{{{title}}}".format(length=(stop-start+1), title=title)
            cmidrule_line += "\\cmidrule(lr){{{start}-{end}}}".format(start=start, end=stop)
            position = stop + 1
            if i < len(header):
                header_line += " & "
        self._line(header_line + "\\\\")
        self._line(cmidrule_line)

    def add_dense_header(self, fieldnames):
        line = ""
        for i, col in enumerate(fieldnames):
            line += "\\multicolumn{{1}}{{c}}{{{col}}}".format(col=col)
            if i < len(fieldnames) - 1:
                line += " & "
        self._line(line + "\\\\")

    def end_tabular(self):
        self._line("\\end{tabular}")

    def _line(self, content):
        if "\\end" in content:
            self.current_indent -= 1
        self.lines.append(self.indent*self.current_indent + content + "\n")
        if "\\begin" in content:
            self.current_indent += 1

    def write(self, output):
        with open(output, "w") as f:
            f.writelines(self.lines)