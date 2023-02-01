"""
Sphinx extension that adds `ipython-with-element_reprs` directive.
"""
from typing import List

import docutils.nodes
import sphinx.addnodes
import sphinx.application
import sphinx.util.docutils
import sphinx.util.logging

logger = sphinx.util.logging.getLogger(__name__)

ELEMENT_REPR_TO_TITLE = {
    "int": "Integer",
    "poly": "Polynomial",
    "power": "Power",
}


class IPythonWithReprsDirective(sphinx.util.docutils.SphinxDirective):
    has_content = True
    required_arguments = 1
    optional_arguments = 1
    option_spec = {
        "name": str,
    }

    def run(self) -> List[docutils.nodes.Node]:
        self.assert_has_content()

        # Parse input parameters
        element_reprs = self.arguments[0].split(",")
        titles = [ELEMENT_REPR_TO_TITLE[element_repr] for element_repr in element_reprs]
        field = self.options.get("name", "GF")

        ws = "    "
        new_lines = [
            ".. md-tab-set::",
            "",
        ]

        for element_repr, title in zip(element_reprs, titles):
            new_lines += [
                f"{ws}.. md-tab-item:: {title}",
                "",
                f"{ws}{ws}.. ipython:: python",
                "",
            ]

            # Set the Galois field element representation
            first_code_line = self.content[0]
            if first_code_line.startswith(f"{field} = "):
                assert "repr=" not in first_code_line
                if element_repr == "int":
                    new_first_code_line = first_code_line
                else:
                    items = first_code_line.rsplit(")", 1)
                    assert len(items) == 2
                    items.insert(1, f', repr="{element_repr}")')
                    new_first_code_line = "".join(items)
                new_lines += [
                    f"{ws}{ws}{ws}{new_first_code_line}",
                ]
            else:
                new_lines += [
                    f"{ws}{ws}{ws}@suppress",
                    f'{ws}{ws}{ws}{field}.repr("{element_repr}")',
                    f"{ws}{ws}{ws}{first_code_line}",
                ]

            # Add the raw python code
            for code_line in self.content[1:]:
                new_lines += [
                    f"{ws}{ws}{ws}{code_line}",
                ]

            # Reset the element representation
            new_lines += [
                f"{ws}{ws}{ws}@suppress",
                f"{ws}{ws}{ws}{field}.repr()",
                "",
            ]

        self.state_machine.insert_input(new_lines, self.state_machine.input_lines.source(0))

        return []


def setup(app: sphinx.application.Sphinx):
    app.add_directive("ipython-with-reprs", IPythonWithReprsDirective)

    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
