archive() {
    local destination="elsto@archive.mpcdf.mpg.de"
    local default_path="/ghi/r/e/elsto/"
    local auto_yes=false
    local recursive=false
    local verbose=false

    # Function to display help message
    show_help() {
        echo "Usage: archive [-r] [-y] [-v] [-h] <file/folder> [destination path]"
        echo
        echo "Options:"
        echo "  -r            Archive recursively (use for directories)"
        echo "  -y            Automatically confirm overwriting existing files"
        echo "  -v            Verbose mode (shows the command without running it)"
        echo "  -h            Show this help message"
        echo
        echo "Arguments:"
        echo "  <file/folder>   The file or folder to archive"
        echo "  [destination]   Optional remote path (defaults to ${destination}:${default_path})"
        echo
        echo "Example:"
        echo "  archive file.txt                                                         # Archive file.txt to the default path"
        echo "  archive -r folder   ${destination}:${default_path}           # Archive folder recursively to a specific path"
        echo "  archive -y file.txt ${destination}:${default_path}/folder/   # Skip overwrite check"
        echo
    }

    # Parse options: -r for recursive, -y for auto-confirmation, -v for verbose, -h for help
    while getopts "hryv" opt; do
        case ${opt} in
            r)
                recursive=true
                ;;
            y)
                auto_yes=true
                ;;
            v)
                verbose=true
                ;;
            h)
                show_help
                return 0
                ;;
            *)
                show_help
                return 1
                ;;
        esac
    done
    shift $((OPTIND-1))

    # Get source file/folder and optional destination path
    local source="$1"
    local remote_path="${2:-$default_path}"

    # Ensure source is provided
    if [ -z "$source" ]; then
        echo "Error: No source file/folder specified."
        show_help
        return 1
    fi

    # Define the target file/folder path on the remote server
    local filename=$(basename "$source")
    local remote_file_path="${remote_path}${filename}"

    # Function to check if a file exists on the remote server
    file_exists_on_remote() {
        ssh ${destination} "[ -e '$remote_file_path' ]"
    }

    # Validate whether the source exists as a file or directory
    if [ "$recursive" = true ]; then
        if [ ! -d "$source" ]; then
            echo "Error: '$source' is not a valid directory."
            return 1
        fi
    else
        if [ ! -f "$source" ]; then
            echo "Error: '$source' is not a valid file."
            return 1
        fi
    fi

    # Display or run the command
    scp_command="scp"
    if [ "$recursive" = true ]; then
        scp_command+=" -r"
    fi
    scp_command+=" \"$source\" ${destination}:\"$remote_path\""

    # Verbose mode: just print the command and skip execution
    if [ "$verbose" = true ]; then
        echo "Command to be run: $scp_command"
        return 0
    fi

    # Check if file exists on the remote and prompt unless -y is specified
    if $auto_yes || ! file_exists_on_remote; then
        echo "Running: $scp_command"
        eval $scp_command
    else
        echo "Warning: $remote_file_path already exists on the remote server."
        read -p "Do you want to overwrite it? [y/N] " confirm
        if [[ "$confirm" =~ ^[Yy]$ ]]; then
            echo "Running: $scp_command"
            eval $scp_command
        else
            echo "Skipping upload."
        fi
    fi
}
