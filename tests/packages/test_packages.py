import os
import importlib
import ast
import pytest

def check_package(package_name):
    try:
        importlib.import_module(package_name)
        print(f"{package_name} is correctly installed.")
        return True
    except ImportError:
        print(f"{package_name} is not installed.")
        return False

def check_imports_in_file(file_path):
    with open(file_path, "r") as file:
        tree = ast.parse(file.read(), filename=file_path)

    imported_packages = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_packages.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            imported_packages.add(node.module)

    return imported_packages

@pytest.mark.my_test
def main():
    # Directory containing your project files
    project_dir = "."

    # List to store all imported packages
    all_imported_packages = set()

    # Traverse through all Python files in the project directory
    for root, _, files in os.walk(project_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                imported_packages = check_imports_in_file(file_path)
                all_imported_packages.update(imported_packages)

    # Check each imported package
    all_packages_installed = all(check_package(package) for package in all_imported_packages)

    if all_packages_installed:
        print("All required packages are installed.")
    else:
        print("Some required packages are missing.")

if __name__ == "__main__":
    main()
