#!/bin/bash

help_message() {
    echo "Usage: $0 [option]"
    echo "Options:"
    echo "  --gc                Perform basic garbage collection"
    echo "  --aggressive        Perform aggressive garbage collection"
    echo "  --prune             Remove unreferenced objects (Not recommended)"
    echo "  --large-files       Identify large files in the repository"
    echo "  --clean-branches    Remove merged branches"
    echo "  --help              Show this help message"
}

# Check if the first argument is a recognized option
case "$1" in
    --gc|--aggressive|--prune|--large-files|--clean-branches|--help)
        OPTION="$1"
        ;;
    *)
        BASE_DIR=${1:-$HOME}
        if [ ! -d "$BASE_DIR" ]; then
            echo "Error: Directory $BASE_DIR does not exist."
            exit 1
        fi
        ;;
esac

# Function to create a backup of the .git folder
git_backup() {
    if [ -d .git ]; then
        TIMESTAMP=$(date +%Y%m%d%H%M%S)
        BACKUP_DIR=".git-backup-$TIMESTAMP"
        echo "Creating backup of .git folder at $BACKUP_DIR..."
        cp -r .git "$BACKUP_DIR"
    fi
}

# Function to print .git size before and after cleaning
print_git_size() {
    echo "Size of .git before:"
    du -sh .git
}

# Function to perform garbage collection
git_gc() {
    git_backup
    print_git_size
    echo "Running basic garbage collection..."
    git gc --prune=now
    echo "Size of .git after:"
    du -sh .git
}

# Function to perform aggressive garbage collection
git_aggressive_gc() {
    git_backup
    print_git_size
    echo "Running aggressive garbage collection..."
    git gc --aggressive --prune=now
    echo "Size of .git after:"
    du -sh .git
}

# Function to remove unreferenced objects (commented out for safety)
# git_prune() {
#     git_backup
#     print_git_size
#     echo "Removing unreferenced objects..."
#     git reflog expire --expire=now --all
#     git gc --prune=now
#     echo "Size of .git after:"
#     du -sh .git
# }

# Function to find large files
git_large_files() {
    echo "Identifying large files..."
    git rev-list --objects --all | sort -k 2 -n | tail -n 10
}

# Function to remove merged branches
git_clean_branches() {
    echo "Removing merged branches..."
    git branch --merged | grep -v "\*" | xargs -n 1 git branch -d
}

# Main script execution
case "$OPTION" in
    --gc)
        git_gc
        ;;
    --aggressive)
        git_aggressive_gc
        ;;
    # --prune)
    #     git_prune
    #     ;;
    --large-files)
        git_large_files
        ;;
    --clean-branches)
        git_clean_branches
        ;;
    --help)
        help_message
        ;;
    *)
        echo "Invalid option. Use --help for usage information."
        exit 1
        ;;
esac