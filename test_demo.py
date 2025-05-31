#!/usr/bin/env python3

import asyncio
import os
from interactive_demo import run_quick_demo, run_interactive_demo

def main():
    print("🎯 AI Urban Planning Demo")
    print("========================")
    print("Choose your demo mode:")
    print("1. Quick Demo (pre-defined questions)")
    print("2. Interactive Demo (full consultation)")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ")
    
    if choice == "1":
        print("\n🚀 Running Quick Demo...")
        asyncio.run(run_quick_demo())
    elif choice == "2":
        print("\n🚀 Running Interactive Demo...")
        asyncio.run(run_interactive_demo())
    elif choice == "3":
        print("👋 Goodbye!")
        return
    else:
        print("❌ Invalid choice")
        main()

if __name__ == "__main__":
    main()