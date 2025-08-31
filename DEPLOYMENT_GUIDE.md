# 🚀 Vercel Deployment Guide

## Files to Upload to Vercel:

### 1. Root Directory Files:
- `index.html` (main webpage)
- `vercel.json` (configuration)
- `requirements.txt` (Python dependencies)
- `package.json` (project metadata)

### 2. API Directory:
- `api/predict.py` (serverless function)

## 🌐 Vercel Website Deployment Steps:

### Step 1: Go to vercel.com
1. Visit https://vercel.com
2. Click "Sign Up" (use GitHub for easier integration)
3. Verify your email

### Step 2: Create New Project
1. Click "New Project" on dashboard
2. Choose "Import Git Repository" OR "Browse All Templates"

### Step 3A: If Using GitHub
1. Connect your GitHub account
2. Import the repository with your project files
3. Vercel will auto-detect the configuration

### Step 3B: If Uploading Files Directly
1. Create a ZIP file with all project files
2. Upload the ZIP file to Vercel
3. Extract and configure

### Step 4: Configure Project
```
Project Name: personalized-workout-recommender
Framework: Other
Root Directory: ./
Build Command: (leave empty)
Output Directory: (leave empty)
```

### Step 5: Deploy
1. Click "Deploy"
2. Wait 2-3 minutes
3. Get your live URL!

## 🎯 Expected Result:
Your app will be live at: https://personalized-workout-recommender.vercel.app

## 🔧 Troubleshooting:
- If Python errors: Check requirements.txt
- If API errors: Check api/predict.py syntax
- If frontend errors: Check index.html paths

## 📱 Testing Your Deployment:
1. Visit your live URL
2. Enter a fitness prompt
3. Click "Find My Best Workout Style"
4. See your recommendation!
