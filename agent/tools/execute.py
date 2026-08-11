from .filesystem import list_files, read_file
from .search import grep_code
from .filesystem import edit_file
from .shell import run_tests
from .git import git_diff

def execute_tool(name, args, *, repo_path, test_command):
    if name == "list_files":
        return list_files(repo_path, args.get("path", "."))
    if name == "read_file":
        return read_file(repo_path, args["path"])
    if name == "grep_code":
        return grep_code(repo_path, args["query"])
    if name == "edit_file":
        return edit_file(repo_path, **args)
    if name == "run_tests":
        return run_tests(repo_path, test_command)
    if name == "git_diff":
        return git_diff(repo_path)

    return f"Error: unknown tool: {name}"