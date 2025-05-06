import re
elements = "2.029 turbo"

match = re.match(r'^(\d+)\s*([a-zA-Z]+)$', elements)
print(match)
print(match[2])
