#!/usr/bin/env python3
"""
Simple execution wrapper for the Cultural Analysis Dashboard
Runs the advanced dashboard with automatic setup and error handling
"""

import sys
import subprocess
import os
from pathlib import Path

def install_requirements():
    """Install required packages if not available"""
    required_packages = ['plotly', 'pandas']
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    """Main execution function"""
    print("🎵 Cultural Analysis Dashboard Runner")
    print("=" * 50)
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    if current_dir.name != 'cultural_analysis':
        cultural_dir = current_dir / 'cultural_analysis'
        if cultural_dir.exists():
            os.chdir(cultural_dir)
            print(f"Changed directory to: {cultural_dir}")
        else:
            print("❌ Error: Please run this script from the cultural_analysis directory")
            print("   or from the parent directory containing cultural_analysis/")
            return 1
    
    # Install requirements
    print("\n📦 Checking and installing requirements...")
    try:
        install_requirements()
        print("✅ All requirements satisfied")
    except Exception as e:
        print(f"❌ Error installing requirements: {e}")
        return 1
    
    # Run the advanced dashboard
    print("\n🚀 Launching Advanced Cultural Analysis Dashboard...")
    try:
        from advanced_cultural_analysis_dashboard import AdvancedCulturalAnalysisVisualizer
        
        print("📊 Initializing visualizer...")
        visualizer = AdvancedCulturalAnalysisVisualizer()
        
        print("📈 Creating comprehensive dashboard...")
        dashboard_path = visualizer.create_advanced_dashboard()
        
        print(f"\n✅ Dashboard created successfully!")
        print(f"📁 Location: {dashboard_path[0]}")
        print(f"📊 Insights: {dashboard_path[1]}")
        print(f"🌐 All charts are now embedded in a single page!")
        
        # Try to open in browser (suppress terminal warnings)
        try:
            import webbrowser
            import os
            # Suppress GLib warnings by redirecting stderr temporarily
            with open(os.devnull, 'w') as devnull:
                import subprocess
                # Use subprocess to avoid terminal warnings
                subprocess.Popen([webbrowser.get().name, dashboard_path[0]], 
                               stderr=devnull, stdout=devnull)
            print("🚀 Opening dashboard in browser...")
        except Exception as e:
            print(f"⚠️ Could not auto-open browser: {e}")
            print(f"📁 Please manually open: {dashboard_path[0]}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error running dashboard: {e}")
        print(f"📝 Try running directly: python advanced_cultural_analysis_dashboard.py")
        return 1

if __name__ == "__main__":
    exit_code = main()
    
    if exit_code == 0:
        print("\n🎉 Analysis complete! Check your browser for the interactive dashboard.")
    else:
        print("\n💡 For troubleshooting, check the COMBINED_ANALYSIS_README.md file")
    
    input("\nPress Enter to exit...")
    sys.exit(exit_code) 