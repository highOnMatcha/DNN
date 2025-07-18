#!/bin/bash
# Set up environment variables for database connection
# Usage: source set_db_env.sh

# IMPORTANT: Copy this file to set_db_env.sh and replace with your actual values
# Never commit the actual set_db_env.sh file with real credentials to git

export DB_HOST="your_database_host"
export DB_PORT="5432"
export DB_NAME="your_database_name"
export DB_USER="your_username"
export DB_PASSWORD="your_password"

echo "Database environment variables set."
echo "You can now run your Python script safely."
echo ""
echo "REMINDER: Make sure to:"
echo "1. Replace the placeholder values with your actual credentials"
echo "2. Keep set_db_env.sh in .gitignore to prevent committing secrets"
echo "3. Use 'source set_db_env.sh' before running your Python scripts"
