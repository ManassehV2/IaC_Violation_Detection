import re

def enhanced_terraform_preprocessing(code: str) -> str:
    code = re.sub(r'\bresource\s+"([^"]+)"\s+"([^"]+)"', r'<RESOURCE> \1 \2', code)
    code = re.sub(r'\bdata\s+"([^"]+)"\s+"([^"]+)"',    r'<DATA> \1 \2', code)
    code = re.sub(r'\bprovider\s+"([^"]+)"',             r'<PROVIDER> \1', code)
    code = re.sub(r'\bmodule\s+"([^"]+)"',               r'<MODULE> \1', code)

    patterns = [
        (r'0\.0\.0\.0/0',        '<SECURITY_OPEN>'),
        (r'\*',                  '<WILDCARD>'),
        (r'public',              '<PUBLIC_ACCESS>'),
        (r'internet',            '<INTERNET_ACCESS>'),
        (r'encryption.*false',   '<NO_ENCRYPTION>'),
        (r'admin',               '<ADMIN_ACCESS>'),
        (r'root',                '<ROOT_ACCESS>')
    ]
    for pat, tok in patterns:
        code = re.sub(pat, tok, code, flags=re.IGNORECASE)

    return re.sub(r'\s+', ' ', code).strip()
