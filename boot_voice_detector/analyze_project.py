import os
def show_structure(root='.', indent='', max_depth=3, current_depth=0):
    if current_depth > max_depth:
        return
    try:
        items = sorted(os.listdir(root))
    except PermissionError:
        return
    for name in items:
        path = os.path.join(root, name)
        if os.path.isdir(path):
            print(indent + '📁 ' + name + '/')
            if current_depth < max_depth:
                show_structure(path, indent + '  ', max_depth, current_depth + 1)
        elif any(name.endswith(ext) for ext in ['.csv', '.py', '.yml', '.yaml', '.txt', '.md', '.yaml', '.yml', '.md', '.example']):
            emoji = '🐍' if name.endswith('.py') else '📄'
            print(indent + emoji + ' ' + name)
show_structure()