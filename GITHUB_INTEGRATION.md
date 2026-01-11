# GitHub Integration for Monty

## What's New

Monty now includes GitHub integration that allows you to:

1. **Project-specific subdirectories**: Each project is created in its own subdirectory under `output/`
2. **Automatic GitHub push**: After completion, push your project to a new GitHub repository
3. **Optional cleanup**: Delete local files after successful push to save disk space

## How It Works

### Project Subdirectories

When Monty builds a project, it automatically creates a subdirectory based on the project name:

- Project Name: "My Awesome API"
- Output Directory: `./output/my-awesome-api/`

This keeps multiple projects organized and separate.

### GitHub Push Workflow

After Monty completes all user stories:

1. **Prompt**: You'll be asked if you want to push to GitHub
2. **Repository Creation**: A new public repository is created on your GitHub account
3. **Initial Commit**: All project files are committed with proper `.gitignore`
4. **Push**: Code is pushed to the new repository
5. **Cleanup Option**: Choose to delete local files or keep them

### Prerequisites

To use GitHub integration, you need:

1. **GitHub CLI installed**:
   ```bash
   # macOS
   brew install gh

   # Linux
   # See https://github.com/cli/cli#installation
   ```

2. **GitHub authentication**:
   ```bash
   gh auth login
   ```

## Usage

### Automatic (via Wizard)

1. Run the setup wizard:
   ```bash
   ./setup.sh
   ```

2. Configure your project as usual

3. When Monty completes, you'll see:
   ```
   Would you like to push this project to GitHub?
   This will create a new repository and optionally clean up local files (y/N):
   ```

4. Answer `y` to push to GitHub

5. Choose whether to delete local files:
   ```
   Delete local files after successful push? (y/N):
   ```

### Manual Push

If you chose not to push during completion, you can push later:

```bash
./scripts/github-push.sh "./output/your-project-name" "Your Project Name" [yes|no]
```

Parameters:
- `project_dir`: Path to the project directory
- `project_name`: Human-readable project name
- `cleanup`: "yes" to delete local files after push (optional, default: "no")

## Repository Naming

Projects are converted to GitHub-friendly names:
- Spaces become hyphens
- Special characters are removed
- Uppercase becomes lowercase

Examples:
- "My Awesome API!" → `my-awesome-api`
- "Real-Time Chat App" → `real-time-chat-app`

## What Gets Pushed

The GitHub repository includes:
- All project code
- Docker Compose configuration
- README with setup instructions
- `.gitignore` with sensible defaults
- Environment variable examples (`.env.example`)

## What's Ignored

The `.gitignore` automatically excludes:
- Python cache files (`__pycache__`, `*.pyc`)
- Virtual environments (`venv/`, `.env`)
- IDE files (`.vscode/`, `.idea/`)
- Secret files (`*.key`, `*.pem`, credentials)
- Database files (`*.db`, `*.sqlite`)

## Cleanup Behavior

When cleanup is enabled:
- The entire local project directory is deleted
- Only the GitHub repository remains
- You can clone it anytime with: `git clone https://github.com/username/repo-name`

When cleanup is disabled:
- Local files remain untouched
- You have both local and remote copies
- Can continue development locally

## Troubleshooting

### "GitHub CLI not found"

Install the GitHub CLI:
```bash
brew install gh  # macOS
```

### "GitHub authentication required"

Authenticate with GitHub:
```bash
gh auth login
```

### "Repository already exists"

The script will ask if you want to overwrite. Choose carefully as this deletes the existing repository.

### Manual Repository Management

If you need more control:
```bash
# View your repositories
gh repo list

# Delete a repository
gh repo delete username/repo-name --yes

# Clone a repository
gh repo clone username/repo-name
```

## Security Notes

- Repositories are created as **public** by default
- Never commit secrets or API keys
- Use environment variables for sensitive data
- Always include `.env` in `.gitignore`

## Benefits

1. **Version Control**: All projects are properly versioned
2. **Backup**: Projects are safely stored on GitHub
3. **Sharing**: Easy to share projects with others
4. **Portfolio**: Build a portfolio of AI-generated projects
5. **Disk Space**: Clean up local files while keeping GitHub backup

---

"Excellent..." - Monty