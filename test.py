import sys

# Save original arguments
original_argv = sys.argv.copy()

# Set your desired arguments
sys.argv = ['main.py', '--config', 'config.yaml']

# Run the script
exec(open('main.py').read())

# Restore original arguments
sys.argv = original_argv