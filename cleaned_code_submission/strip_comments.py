import os
import ast
import glob
import shutil

def remove_docstrings(node):
    if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef, ast.Module)):
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body.pop(0)
    for child in ast.iter_child_nodes(node):
        remove_docstrings(child)

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()
    try:
        parsed = ast.parse(source)
        remove_docstrings(parsed)
        unparsed = ast.unparse(parsed)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(unparsed)
            f.write('\n')
    except Exception as e:
        print(f'Error processing {filepath}: {e}')

def main():
    target_dir = 'submission'
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    py_files = glob.glob('*.py')
    py_files.extend(glob.glob('models/*.py'))
    py_files.extend(glob.glob('utils/*.py'))
    for f in py_files:
        dest = os.path.join(target_dir, f)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(f, dest)
        process_file(dest)
        print(f'Processed {f}')
if __name__ == '__main__':
    main()
