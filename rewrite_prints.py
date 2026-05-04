import re

def process_file(filepath, start_marker, end_marker, out_filename_expr):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    in_run = False
    new_lines = []
    
    for line in lines:
        if start_marker in line:
            in_run = True
            new_lines.append(line)
            # inject file open
            indent = line[:len(line) - len(line.lstrip())]
            new_lines.append(f'{indent}stats_path = {out_filename_expr}\n')
            new_lines.append(f'{indent}with open(stats_path, "w", encoding="utf-8") as f:\n')
            continue
            
        if in_run and end_marker in line:
            in_run = False
            
        if in_run:
            # Need to indent everything inside the with block by 4 spaces
            # but wait, it's easier to just pass file=f to print
            pass

# Actually, an easier way is to just use a regular expression
# to replace `print(` with `print(file=f, ` between def run and return results.
# Wait, let's just do it directly.
