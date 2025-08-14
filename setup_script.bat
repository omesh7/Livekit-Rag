@echo off
title AWS S3 Vectors Portfolio Setup

echo =======================================
echo Setting up AWS S3 Vectors for Portfolio Assistant
echo =======================================
echo.

REM Check if AWS CLI is installed
aws --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] AWS CLI is not installed. Install it from: https://aws.amazon.com/cli/
    pause
    exit /b 1
)

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python 3.8+ is not installed. Please install it.
    pause
    exit /b 1
)

REM Install Python dependencies
echo Installing required Python packages...
uv add boto3 python-dotenv llama-index llama-index-embeddings-huggingface llama-index-llms-google-genai
echo.

REM AWS credentials setup reminder
echo Configuring AWS credentials...
echo Run: aws configure
echo Required permissions:
echo   - s3:CreateBucket
echo   - s3:ListBucket
echo   - s3:GetObject
echo   - s3:PutObject
echo   - s3vectors:CreateVectorIndex
echo   - s3vectors:PutVectors
echo   - s3vectors:QueryVectors
echo   - s3vectors:DescribeVectorIndex
echo.

REM Check AWS credentials
aws sts get-caller-identity >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] AWS credentials not configured or invalid.
    echo Run: aws configure
    pause
    exit /b 1
) else (
    echo AWS credentials verified!
)

REM Test the Python setup
echo.
echo Testing S3 Vectors setup...
python s3_vector_storage.py

if %errorlevel% equ 0 (
    echo.
    echo [SUCCESS] S3 Vectors setup complete!
    echo Your portfolio data has been embedded and stored in AWS S3 Vectors.
    echo.
    echo To run the voice assistant:
    echo python query_engine_s3.py
) else (
    echo.
    echo [ERROR] Setup failed. Please check the error messages above.
    echo Possible issues:
    echo   1. AWS region does not support S3 Vectors
    echo   2. Missing permissions
    echo   3. No documents to embed in data directory
)

echo.
pause
