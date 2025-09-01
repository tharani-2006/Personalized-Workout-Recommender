"""
Test script to verify Vercel deployment readiness
Run this before deploying to catch any issues
"""

import json
import os
import sys

def check_file_structure():
    """Check if all required files exist"""
    print("📁 Checking file structure...")
    
    required_files = [
        'index.html',
        'vercel.json', 
        'requirements.txt',
        'package.json',
        'api/predict.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - MISSING!")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def check_vercel_config():
    """Check vercel.json configuration"""
    print("\n⚙️ Checking Vercel configuration...")
    
    try:
        with open('vercel.json', 'r') as f:
            config = json.load(f)
        
        # Check required fields
        if 'version' in config and config['version'] == 2:
            print("   ✅ Version 2 configuration")
        else:
            print("   ❌ Missing or incorrect version")
            return False
            
        if 'routes' in config:
            print("   ✅ Routes configured")
        else:
            print("   ❌ Missing routes configuration")
            return False
            
        return True
        
    except Exception as e:
        print(f"   ❌ Error reading vercel.json: {e}")
        return False

def check_api_function():
    """Check API function syntax"""
    print("\n🔧 Checking API function...")
    
    try:
        # Try to import and basic syntax check
        sys.path.append('api')
        
        with open('api/predict.py', 'r') as f:
            content = f.read()
        
        # Check for required components
        if 'class handler(BaseHTTPRequestHandler)' in content:
            print("   ✅ Handler class found")
        else:
            print("   ❌ Missing handler class")
            return False
            
        if 'def do_POST(self)' in content:
            print("   ✅ POST method found")
        else:
            print("   ❌ Missing POST method")
            return False
            
        if 'def do_OPTIONS(self)' in content:
            print("   ✅ CORS handling found")
        else:
            print("   ❌ Missing CORS handling")
            return False
            
        return True
        
    except Exception as e:
        print(f"   ❌ Error checking API function: {e}")
        return False

def check_frontend():
    """Check frontend configuration"""
    print("\n🌐 Checking frontend...")
    
    try:
        with open('index.html', 'r') as f:
            content = f.read()
        
        # Check for API endpoint
        if '/api/predict' in content:
            print("   ✅ API endpoint correctly configured")
        else:
            print("   ❌ API endpoint not found or incorrect")
            return False
            
        # Check for required HTML elements
        if 'id="workoutForm"' in content:
            print("   ✅ Form element found")
        else:
            print("   ❌ Missing form element")
            return False
            
        return True
        
    except Exception as e:
        print(f"   ❌ Error checking frontend: {e}")
        return False

def main():
    """Run all deployment readiness checks"""
    print("🚀" + "="*50)
    print("🧪 VERCEL DEPLOYMENT READINESS CHECK")
    print("🚀" + "="*50)
    
    checks = [
        ("File Structure", check_file_structure),
        ("Vercel Config", check_vercel_config), 
        ("API Function", check_api_function),
        ("Frontend", check_frontend)
    ]
    
    passed_checks = 0
    
    for check_name, check_function in checks:
        if check_function():
            passed_checks += 1
        else:
            print(f"\n❌ {check_name} check failed!")
    
    print("\n" + "🏆" + "="*50)
    print(f"📊 DEPLOYMENT READINESS: {passed_checks}/{len(checks)} CHECKS PASSED")
    print("🏆" + "="*50)
    
    if passed_checks == len(checks):
        print("🎉 ALL CHECKS PASSED! Ready for Vercel deployment! 🚀")
        print("\n📋 Next Steps:")
        print("   1. Go to vercel.com")
        print("   2. Create new project")
        print("   3. Upload these files or connect GitHub repo")
        print("   4. Deploy and get your live URL!")
    else:
        print("⚠️ Some checks failed. Fix the issues above before deploying.")

if __name__ == '__main__':
    main()
