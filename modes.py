"""Modes: a working style for the agent. Each mode has its own instructions and its own set of tools.
"General" is the normal agent with every tool. The tool limit is enforced in code (agent.py refuses
a tool that the current mode does not include), not just requested in the instructions."""

DEFAULT_MODE = "general"

# Tools that every mode has: look around, remember things, calculate, search your documents, tasks.
_COMMON = frozenset({
    "get_current_time", "list_files", "read_file", "fetch_url", "calculate",
    "remember", "forget", "project_remember", "project_forget",
    "knowledge_search", "knowledge_list", "knowledge_add",
    "add_task", "list_tasks", "complete_task", "delete_task",
})

MODES = {
    "general": {
        "label": "General",
        "description": "The normal agent with every tool.",
        "instruction": "",
        "tools": None,  # None = no limit
    },
    "coder": {
        "label": "Coder",
        "description": "Builds and fixes code. Runs and checks projects. No pictures or office documents.",
        "instruction": (
            "Work like a careful senior developer. Before you change anything, read the files involved "
            "and say in one sentence what you think the cause is. Change as little as needed, in the "
            "real project files. Run the right check after every change (test_flask_site, "
            "check_js_project, test_web_page or run_python) and report what it showed. Never say a "
            "fix works unless a check passed or I confirmed it."
        ),
        "tools": _COMMON | {"write_file", "run_python", "install_packages", "test_flask_site",
                            "test_web_page", "check_js_project", "db_schema", "db_query"},
    },
    "writer": {
        "label": "Writer",
        "description": "Letters, reports, CVs, slides and pictures. Cannot run or install code.",
        "instruction": (
            "Write clearly and in my voice. Ask about the audience only when you cannot guess it. Give "
            "complete drafts, not outlines, unless I ask for an outline, and avoid filler. For letters, "
            "reports, CVs and similar, make the document with create_docx or create_pdf with one title "
            "only. When you edit my text, keep my meaning and say what you changed in one short line."
        ),
        "tools": _COMMON | {"write_file", "create_docx", "create_pdf", "create_pptx", "create_xlsx",
                            "generate_image", "create_svg", "search_images"},
    },
    "tutor": {
        "label": "Tutor",
        "description": "Teaches step by step and checks your understanding. Can run small examples.",
        "instruction": (
            "Teach, do not just answer. Start from what I already know, explain one idea at a time in "
            "plain words, give one short example, then ask me one question to check I understood. When "
            "I ask for the answer to an exercise, give it and then explain the steps. Use calculate for "
            "arithmetic and run_python for small examples I can see run. Be patient and encouraging "
            "without flattery."
        ),
        "tools": _COMMON | {"write_file", "create_pdf", "create_docx", "run_python", "search_images"},
    },
    "finance": {
        "label": "Finance",
        "description": "Careful numbers, budgets and spreadsheets. Uses the calculator for every sum.",
        "instruction": (
            "Be exact with numbers. Use calculate for every calculation, show the formula and your "
            "assumptions, and say which currency you use (naira, written with the sign N or the word "
            "naira, unless I say otherwise). Use 'about' for estimates. Never invent a rate, price or "
            "percentage: if you do not have a source, say so and ask me for the figure. You are not a "
            "licensed adviser: give information and options with their risks, never guarantees. Put "
            "tables and budgets in a spreadsheet with create_xlsx."
        ),
        "tools": _COMMON | {"write_file", "create_xlsx", "create_docx", "create_pdf", "db_schema", "db_query"},
    },
}


def allows(mode_id: str, tool_name: str) -> bool:
    spec = MODES.get(mode_id) or MODES[DEFAULT_MODE]
    tools = spec["tools"]
    return tools is None or tool_name in tools


def list_modes() -> list:
    return [{"id": mode_id, "label": spec["label"], "description": spec["description"]}
            for mode_id, spec in MODES.items()]