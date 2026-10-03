import nbformat
import sys

def create_notebook(filename, cells_content):
    nb = nbformat.v4.new_notebook()
    cells = []
    for cell_type, content in cells_content:
        if cell_type == 'markdown':
            cells.append(nbformat.v4.new_markdown_cell(content))
        elif cell_type == 'code':
            cells.append(nbformat.v4.new_code_cell(content))
    nb.cells = cells
    with open(filename, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"Created {filename}")

if __name__ == '__main__':
    # We will import this from other scripts to generate specific notebooks
    pass
