from jinja2 import Template


def load_jinja_prompt_template(path: str) -> Template:
    with open(path, encoding="utf-8") as f:
        template = f.read()
    return Template(template)
