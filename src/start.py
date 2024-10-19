# app.py
import subprocess
def main():
    print("Starting the app!")
    result = subprocess.run(["streamlit run src/app.py"], shell=True, capture_output=False, text=True)
    while True:
        command = input("Type 'exit' to quit or anything else to continue: ").lower()
        if command == 'exit':
            print("Exiting the app. Goodbye!")
            break
        else:
            print(f"You typed: {command}")

if __name__ == "__main__":
    main()
