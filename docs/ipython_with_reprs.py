"""
Sphinx extension that adds `ipython-with-reprs` directive.
"""
from typing import List

import docutils.nodes
import sphinx.addnodes
import sphinx.application
import sphinx.util.docutils
import sphinx.util.logging

logger = sphinx.util.logging.getLogger(__name__)

DISPLAY_MODE_TO_TITLE = {
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
        display_modes = self.arguments[0].split(",")
        titles = [DISPLAY_MODE_TO_TITLE[mode] for mode in display_modes]
        field = self.options.get("name", "GF")

        ws = "    "
        new_lines = [
            ".. md-tab-set::",
            ""
        ]

        for mode, title in zip(display_modes, titles):
            new_lines += [
                f"{ws}.. md-tab-item:: {title}",
                "",
                f"{ws}{ws}.. ipython:: python",
                "",
            ]

            # Set the Galois field display mode
            first_code_line = self.content[0]
            if first_code_line.startswith(f"{field} = "):
                assert "display=" not in first_code_line
                if mode == "int":
                    new_first_code_line = first_code_line
                else:
                    items = first_code_line.rsplit(")", 1)
                    assert len(items) == 2
                    items.insert(1, f", display=\"{mode}\")")
                    new_first_code_line = "".join(items)
                new_lines += [
                    f"{ws}{ws}{ws}{new_first_code_line}",
                ]
            else:
                new_lines += [
                    f"{ws}{ws}{ws}@suppress",
                    f"{ws}{ws}{ws}{field}.display(\"{mode}\")",
                    f"{ws}{ws}{ws}{first_code_line}",
                ]

            # Add the raw python code
            for code_line in self.content[1:]:
                new_lines += [
                    f"{ws}{ws}{ws}{code_line}",
                ]

            # Reset the Galois field display mode
            new_lines += [
                f"{ws}{ws}{ws}@suppress",
                f"{ws}{ws}{ws}{field}.display()",
                ""
            ]

        self.state_machine.insert_input(new_lines, self.state_machine.input_lines.source(0))

        return []


def setup(app: sphinx.application.Sphinx):
    app.add_directive("ipython-with-reprs", IPythonWithReprsDirective)

    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
